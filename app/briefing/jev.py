"""TypeSafe Jev (System One) HTTP client: typed Noul / Choice / Score questions in one call."""

import asyncio
import logging
import time

import httpx

from app import costs
from app.config import get_settings

log = logging.getLogger(__name__)

URL = "https://api.typesafe.ai/v1/systemone"


class JevError(RuntimeError):
    pass


def noul(instructions, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def choice(instructions, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, levels: list) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


async def system_one(state, questions: dict, *, stage: str, client: httpx.AsyncClient | None = None) -> dict:
    """POST one evaluation. Returns the `answers` map. Retries 429/529 with backoff."""
    s = get_settings()
    if not s.typesafe_api_key:
        raise JevError("TYPESAFE_API_KEY is not set")
    await costs.check_budget(stage)
    body = {"state": state, "model": s.jev_model, "questions": questions}
    headers = {"Authorization": f"Bearer {s.typesafe_api_key}"}
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        for attempt in range(4):
            started = time.monotonic()
            r = await client.post(URL, json=body, headers=headers)
            latency = int((time.monotonic() - started) * 1000)
            if r.status_code in (429, 529, 502, 503) and attempt < 3:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if r.status_code >= 400:
                raise JevError(f"Jev {r.status_code}: {r.text[:300]}")
            data = r.json()
            await costs.record("typesafe", stage, {**data.get("usage", {}), "model": data.get("model")}, 0.0, latency)
            data["answers"]["_latency_ms"] = latency
            return data["answers"]
        raise JevError("Jev retries exhausted")
    finally:
        if own:
            await client.aclose()


def normalized_score(answer: dict) -> float:
    """Score answers are 0..(levels-1); map to 0..1."""
    n = max(len(answer.get("legend", {})) - 1, 1)
    return float(answer["score"]) / n
