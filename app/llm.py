"""Thin wrapper over the Gemini API (google-genai): model, thinking level, retries, cost logging."""

import asyncio
import json
import logging
import time
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app import costs
from app.config import get_settings

log = logging.getLogger(__name__)

_client: genai.Client | None = None

# JSON-schema keys the Gemini schema subset may reject; they only matter for validation on our side.
_STRIP_KEYS = {"additionalProperties", "strict", "$schema"}


class LLMRefusal(RuntimeError):
    pass


def client() -> genai.Client:
    global _client
    if _client is None:
        key = get_settings().gemini_api_key or None  # None -> SDK reads GEMINI_API_KEY / GOOGLE_API_KEY
        _client = genai.Client(api_key=key)
    return _client


def clean_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: clean_schema(v) for k, v in schema.items() if k not in _STRIP_KEYS}
    if isinstance(schema, list):
        return [clean_schema(v) for v in schema]
    return schema


def _thinking(level: str) -> types.ThinkingConfig:
    return types.ThinkingConfig(thinking_level=level.upper())


async def generate(
    *,
    stage: str,
    thinking: str,
    contents: Any,
    system: str | None = None,
    tools: list[types.Tool] | None = None,
    json_schema: dict | None = None,
    max_output_tokens: int = 16000,
) -> types.GenerateContentResponse:
    """One generate_content call with house defaults; logs tokens + cost; raises on blocked output."""
    s = get_settings()
    await costs.check_budget(stage)
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_output_tokens,
        thinking_config=_thinking(thinking),
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if json_schema is not None:
        config.response_mime_type = "application/json"
        config.response_json_schema = clean_schema(json_schema)

    started = time.monotonic()
    for attempt in range(4):
        try:
            resp = await client().aio.models.generate_content(model=s.llm_model, contents=contents, config=config)
            break
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 500, 503, 504) and attempt < 3:
                await asyncio.sleep(2 * 2**attempt)
                continue
            raise
    latency = int((time.monotonic() - started) * 1000)

    usage = resp.usage_metadata
    searches = _search_query_count(resp)
    await costs.record(
        "gemini",
        stage,
        {
            "model": s.llm_model,
            "input_tokens": getattr(usage, "prompt_token_count", 0),
            "output_tokens": getattr(usage, "candidates_token_count", 0),
            "thinking_tokens": getattr(usage, "thoughts_token_count", 0),
            "search_queries": searches,
        },
        costs.llm_cost(s.llm_model, usage) + await costs.search_cost(searches),
        latency,
    )

    if resp.prompt_feedback and resp.prompt_feedback.block_reason:
        raise LLMRefusal(f"prompt blocked: {resp.prompt_feedback.block_reason}")
    if not resp.candidates:
        raise LLMRefusal("no candidates returned")
    reason = resp.candidates[0].finish_reason
    if reason in (types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT, types.FinishReason.BLOCKLIST):
        raise LLMRefusal(f"response blocked: {reason}")
    return resp


def _search_query_count(resp) -> int:
    try:
        gm = resp.candidates[0].grounding_metadata
        return len(gm.web_search_queries or []) if gm else 0
    except (AttributeError, IndexError, TypeError):
        return 0


def text_of(resp) -> str:
    """Concatenate visible text parts (skips thought summaries and function calls)."""
    if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
        return ""
    return "".join(p.text for p in resp.candidates[0].content.parts if p.text and not p.thought).strip()


async def complete_text(system: str, user: str, *, stage: str, thinking: str, max_output_tokens: int = 16000) -> str:
    resp = await generate(stage=stage, thinking=thinking, contents=user, system=system,
                          max_output_tokens=max_output_tokens)
    return text_of(resp)


async def complete_json(
    system: str, user: str, schema: dict, *, stage: str, thinking: str, max_output_tokens: int = 32000
) -> dict:
    resp = await generate(stage=stage, thinking=thinking, contents=user, system=system, json_schema=schema,
                          max_output_tokens=max_output_tokens)
    if resp.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
        raise RuntimeError(f"{stage}: output truncated at max_output_tokens")
    return json.loads(text_of(resp))


SEARCH_SYSTEM = """Answer the query using Google Search. Be factual and concise (under 200 words), include dates
for time-sensitive facts, and say plainly if the results don't answer it."""


async def grounded_search(query: str) -> dict:
    """Run a Google-Search-grounded call; returns {"answer", "sources": [{"title","url"}]}."""
    s = get_settings()
    resp = await generate(
        stage="web_search",
        thinking=s.llm_thinking_chat,
        contents=query,
        system=SEARCH_SYSTEM,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=4000,
    )
    sources = []
    gm = resp.candidates[0].grounding_metadata
    for chunk in (gm.grounding_chunks or []) if gm else []:
        if chunk.web and chunk.web.uri:
            sources.append({"title": chunk.web.title or chunk.web.domain or "source", "url": chunk.web.uri})
    sources = sources[: s.web_search_max_uses]
    await _resolve_redirects(sources)
    return {"answer": text_of(resp), "sources": sources}


async def _resolve_redirects(sources: list[dict]) -> None:
    """Grounding links are long vertexaisearch redirect URLs; swap in the real article URL when we can."""
    async with httpx.AsyncClient(timeout=5, follow_redirects=False) as http:

        async def one(src: dict) -> None:
            if "grounding-api-redirect" not in src["url"]:
                return
            try:
                r = await http.head(src["url"])
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    src["url"] = r.headers["location"]
            except httpx.HTTPError:
                pass

        await asyncio.gather(*(one(src) for src in sources))
