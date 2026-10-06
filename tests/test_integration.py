"""Integration tests against the local Postgres (docker compose up -d db) with Gemini + channels faked.

Skipped automatically if the database isn't reachable.
"""

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from zoneinfo import ZoneInfo

import pytest
from google.genai import types
from sqlalchemy import select, text

from app import llm
from app.channels.base import Button, Channel, Incoming
from app.db import engine, init_db, session_scope
from app.models import Episode, Feedback, Item, Tracker


async def _db_ok() -> bool:
    try:
        async with engine.connect() as c:
            await c.execute(text("select 1"))
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def db():
    if not await _db_ok():
        pytest.skip("database not reachable")
    await init_db()
    yield
    await engine.dispose()


class FakeChannel(Channel):
    name = "telegram"

    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    @property
    def owner(self) -> str:
        return "42"

    async def send_text(self, to, text, sources=None):
        self.sent.append(("text", text))
        return "m"

    async def send_voice(self, to, ogg_path: Path, caption=None):
        self.sent.append(("voice", str(ogg_path)))
        return "m"

    async def send_buttons(self, to, text, buttons: list[Button], sources=None):
        self.sent.append(("buttons", text + " | " + ",".join(b.id for b in buttons)))
        return "m"

    async def send_briefing_notice(self, to, label, episode_id):
        self.sent.append(("notice", f"{label}:{episode_id}"))
        return "m"

    async def send_proactive(self, to, text, sources=None):
        self.sent.append(("proactive", text))
        return "m"

    async def download_media(self, media_ref):
        return b"", "audio/ogg"

    async def answer_callback(self, _id, text=None):
        if text:
            self.sent.append(("toast", text))

    async def send_story_list(self, to, header, groups, episode_id):
        self.sent.append(("stories", (header, groups, episode_id)))
        return "m"

    async def delete_message(self, message_id):
        self.sent.append(("deleted", message_id))


def _resp(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20),
    )


class FakeModels:
    def __init__(self, script):
        self.script = script
        self.calls: list[dict] = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents) if isinstance(contents, list) else contents,
                           "config": config})
        return self.script(len(self.calls), config)


async def test_agent_tool_loop_sets_reminder(monkeypatch):
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).astimezone(ZoneInfo("Asia/Kolkata"))
    marker = f"review PR {uuid.uuid4().hex[:6]}"

    def script(n, config):
        if n == 1:  # model asks for two tools in parallel
            return _resp(
                types.Part(text="Setting that."),
                types.Part(function_call=types.FunctionCall(id="c1", name="set_reminder", args={
                    "text": marker, "repeat": "once", "date": f"{tomorrow:%Y-%m-%d}", "time": "09:00"})),
                types.Part(function_call=types.FunctionCall(id="c2", name="web_search", args={"query": "PR tips"})),
            )
        return _resp(types.Part(text="Saved: reminder tomorrow."))

    fake = FakeModels(script)
    monkeypatch.setattr(llm, "_client", NS(aio=NS(models=fake)))

    async def fake_search(query):
        return {"answer": "tips", "sources": [{"title": "Example", "url": "https://example.com/a"}]}

    monkeypatch.setattr(llm, "grounded_search", fake_search)

    from app.agent.agent import respond

    answer = await respond("remind me tomorrow to review the PR", channel="test")
    assert answer.text == "Saved: reminder tomorrow."
    assert answer.sources == [("Example", "https://example.com/a")]  # web_search sources kept separately

    first = fake.calls[0]["config"]
    assert first.thinking_config.thinking_level == "LOW"
    assert first.automatic_function_calling.disable is True
    names = {d.name for d in first.tools[0].function_declarations}
    assert {"web_search", "set_reminder", "get_story"} <= names

    # second call: history = ..., model turn (with the function calls), user turn with both function responses
    second = fake.calls[1]["contents"]
    assert second[-2].role == "model" and second[-2].parts[1].function_call.name == "set_reminder"
    responses = [p.function_response for p in second[-1].parts]
    assert [r.id for r in responses] == ["c1", "c2"]
    assert "Reminder #" in responses[0].response["result"]

    async with session_scope() as s:
        t = await s.scalar(select(Tracker).where(Tracker.kind == "reminder").order_by(Tracker.id.desc()))
        assert t.payload["text"] == marker
        await s.delete(t)


async def test_llm_json_and_schema_cleaning(monkeypatch):
    def script(n, config):
        assert config.response_mime_type == "application/json"
        assert "additionalProperties" not in json.dumps(config.response_json_schema)
        return _resp(types.Part(text='{"ok": true}'))

    monkeypatch.setattr(llm, "_client", NS(aio=NS(models=FakeModels(script))))
    out = await llm.complete_json("sys", "user", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                                  "required": ["ok"], "additionalProperties": False},
                                  stage="test", thinking="low")
    assert out == {"ok": True}


async def test_pipeline_build_play_and_feedback(monkeypatch):
    tag = uuid.uuid4().hex[:6]
    async with session_scope() as s:
        for i, sec in enumerate(["ai", "ai", "world", "fun"]):
            s.add(Item(section=sec, title=f"Test story {tag} {i}", url=f"https://t.example/{tag}/{i}",
                       summary="summary", full_text="Full article text. " * 20,
                       published_at=datetime.now(timezone.utc)))

    async def fake_json(system, user, schema, *, stage, thinking, max_output_tokens=32000):
        if stage == "rank_llm":
            ids = [int(x) for x in re.findall(r"<candidate id=(\d+)", user)]
            titles = dict(re.findall(r"<candidate id=(\d+).*?\ntitle: (.*?)\n", user, re.S))
            out = []
            for i in ids:
                ours = tag in titles.get(str(i), "")
                out.append({"id": i, "relevance": 10 if ours else 0, "substance": 10 if ours else 0,
                            "section": "ai" if ours and titles[str(i)].endswith(("0", "1")) else
                            ("world" if ours and titles[str(i)].endswith("2") else ("fun" if ours else "skip")),
                            "topic": "llms", "injection_suspected": False})
            return {"items": out}
        assert stage == "script"
        nums = re.findall(r'<article number="(\d+)"', user)
        return {"segments": [{"kind": "intro", "story_number": 0, "text": "Good morning."}]
                + [{"kind": "story", "story_number": int(n), "text": f"Story {n}."} for n in nums]
                + [{"kind": "outro", "story_number": 0, "text": "Bye."}]}

    from app.briefing import ingest, pipeline

    async def no_ingest():
        return []

    monkeypatch.setattr(llm, "complete_json", fake_json)
    monkeypatch.setattr(ingest, "ingest", no_ingest)
    ch = FakeChannel()
    monkeypatch.setattr(pipeline, "primary_channel", lambda: ch)

    ep = await pipeline.build_episode(with_audio=False)
    async with session_scope() as s:
        ep = await s.get(Episode, ep.id)
    assert ep.status == "ready" and len(ep.story_order) == 4
    assert ep.segments[0]["kind"] == "intro" and "Story 1." in ep.script

    await pipeline.deliver_notice(ep)
    assert ch.sent[-1] == ("notice", f"{ep.date:%A}:{ep.id}")

    # Tap Play -> transcript (no audio) + 4 story cards with feedback buttons
    import app.handlers as handlers

    monkeypatch.setattr(handlers, "get_channel", lambda name: ch)
    await handlers.handle(Incoming(channel="telegram", sender="42", external_id=f"cb:{tag}p", kind="button",
                                   button_id=f"play:{ep.id}"))
    lists = [v for k, v in ch.sent if k == "stories"]
    assert len(lists) == 1                                            # ONE story message, not one per story
    header, groups, ep_id = lists[0]
    assert ep_id == ep.id and sum(len(g[1]) for g in groups) == 4
    assert [n for _, stories in groups for n, *_ in stories] == [1, 2, 3, 4]

    # Duplicate delivery of the same button is ignored
    n_before = len(ch.sent)
    await handlers.handle(Incoming(channel="telegram", sender="42", external_id=f"cb:{tag}p", kind="button",
                                   button_id=f"play:{ep.id}"))
    assert len(ch.sent) == n_before

    # Thumbs-down (button on an older message) adjusts the topic weight and answers with a pop-up
    await handlers.handle(Incoming(channel="telegram", sender="42", external_id=f"cb:{tag}f", kind="button",
                                   button_id=f"fb:{ep.story_order[0]}:down", origin_message_id=555))
    assert ch.sent[-1][0] == "toast" and "weight" in ch.sent[-1][1]
    async with session_scope() as s:
        fb = await s.scalar(select(Feedback).where(Feedback.item_id == ep.story_order[0]))
        assert fb.rating == -1

    # Non-owner is ignored
    n_before = len(ch.sent)
    await handlers.handle(Incoming(channel="telegram", sender="999", external_id=f"x{tag}", kind="text", text="hi"))
    assert len(ch.sent) == n_before


async def test_telegram_clear_deletes_remembered_messages(monkeypatch):
    from app.channels.base import send_category
    from app.channels.telegram import TelegramChannel

    ch = TelegramChannel()
    calls: list[tuple[str, dict]] = []
    counter = iter(range(9000, 9100))

    async def fake_call(method, **kwargs):
        calls.append((method, kwargs.get("json", {})))
        return {"message_id": next(counter)} if method.startswith("send") else True

    monkeypatch.setattr(ch, "_call", fake_call)
    await ch.clear()  # start from a clean slate in the test DB
    calls.clear()

    token = send_category.set("briefing")
    await ch.send_text("42", "briefing notice")          # 9000 (briefing)
    send_category.reset(token)
    await ch.send_text("42", "chat answer")               # 9001 (chat)
    await ch.record_incoming(Incoming(channel="telegram", sender="42", external_id="42:77", kind="text"))

    assert await ch.clear(category="briefing") == 1
    assert calls[-1] == ("deleteMessages", {"chat_id": "42", "message_ids": [9000]})

    assert await ch.clear() == 2                          # the rest: bot answer + user's message
    assert sorted(calls[-1][1]["message_ids"]) == [77, 9001]
    assert await ch.clear() == 0                          # nothing left
    await ch.close()


async def test_due_tick_runs_task_and_repeating_reminder(monkeypatch):
    from app import trackers
    from app.channels.base import Reply
    from app.schedules import normalize

    ch = FakeChannel()
    monkeypatch.setattr(trackers, "primary_channel", lambda: ch)
    seen = {}

    async def fake_respond(text, *, channel, message_id=None, scheduled=None):
        seen["args"] = (text, scheduled)
        return Reply("Nifty 25,100 (+0.4%)", [("nseindia.com", "https://nseindia.com/x")])

    monkeypatch.setattr("app.agent.agent.respond", fake_respond)
    now = datetime.now(timezone.utc)
    local = now.astimezone(trackers.get_settings().tz)
    daily = normalize({"repeat": "daily", "time": f"{local:%H:%M}"}, local)
    task = await trackers.add_tracker("task", {"title": "Nifty close", "instruction": "Give the Nifty close.",
                                               "schedule": daily}, due_at=now - timedelta(minutes=1))
    rem = await trackers.add_tracker("reminder", {"text": "drink water", "schedule": daily},
                                     due_at=now - timedelta(minutes=1))
    stale = await trackers.add_tracker("reminder", {"text": "old", "schedule": daily},
                                       due_at=now - timedelta(hours=5))   # missed while asleep

    await trackers.due_tick()

    assert seen["args"] == ("Give the Nifty close.", "Nifty close")
    texts = [t for k, t in ch.sent if k == "proactive"]
    assert any(t.startswith("⏰ Nifty close") and "25,100" in t for t in texts)
    assert "⏰ Reminder: drink water" in texts
    assert not any("old" in t for t in texts)                             # stale repeating run skipped
    async with session_scope() as s:
        for tid in (task.id, rem.id, stale.id):
            row = await s.get(Tracker, tid)
            assert row.status == "open" and row.due_at > now                 # rescheduled, not closed
            await s.delete(row)
