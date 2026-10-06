"""Offline unit tests (no API keys, no database)."""

import hashlib
import hmac
import io
import wave

import numpy as np
import pytest

from app.briefing.dedupe import find_duplicates
from app.briefing.jev import normalized_score
from app.briefing.rank import composite, select_top
from app.channels.telegram import parse_update
from app.channels.whatsapp import _split, parse_webhook, verify_signature
from app.models import ItemScore
from app.speech.audio import chunk_text, stitch_wavs, wav_seconds


def _wav(seconds: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return buf.getvalue()


def test_chunk_text_respects_limit_and_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    chunks = chunk_text(text, 2400)
    assert all(len(c) <= 2400 for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks) == text


def test_chunk_text_splits_monster_sentence():
    text = "word " * 1500  # 7500 chars, no sentence end
    chunks = chunk_text(text, 2400)
    assert len(chunks) >= 4 and all(len(c) <= 2400 for c in chunks)


def test_stitch_wavs_adds_pauses():
    out = stitch_wavs([_wav(1.0), _wav(2.0)], [500, 500])
    assert abs(wav_seconds(out) - 4.0) < 0.01


def test_find_duplicates():
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0], dtype=np.float32)
    near_a = np.array([0.99, 0.1, 0], dtype=np.float32)
    near_a /= np.linalg.norm(near_a)
    old = np.stack([a])
    new = np.stack([b, near_a, b])
    assert find_duplicates(new, old, 0.9) == [None, 0, 1]  # third is dup of the first *new* item (index 1)


def test_whatsapp_signature():
    body = b'{"x":1}'
    sig = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    assert verify_signature("secret", body, sig)
    assert not verify_signature("secret", body, "sha256=deadbeef")
    assert not verify_signature("", body, sig)


def test_parse_whatsapp_webhook():
    payload = {"entry": [{"changes": [{"value": {
        "messages": [
            {"from": "9199", "id": "wamid.1", "type": "text", "text": {"body": "hi"}},
            {"from": "9199", "id": "wamid.2", "type": "audio", "audio": {"id": "media1"}},
            {"from": "9199", "id": "wamid.3", "type": "button", "button": {"payload": "play:3", "text": "Play"}},
            {"from": "9199", "id": "wamid.4", "type": "interactive",
             "interactive": {"type": "button_reply", "button_reply": {"id": "fb:7:up", "title": "👍"}}},
        ],
        "statuses": [{"status": "read", "id": "wamid.x"}],
    }}]}]}
    msgs, statuses = parse_webhook(payload)
    assert [m.kind for m in msgs] == ["text", "voice", "button", "button"]
    assert msgs[0].text == "hi" and msgs[1].media_ref == "media1"
    assert msgs[2].button_id == "play:3" and msgs[3].button_id == "fb:7:up"
    assert statuses[0]["status"] == "read"


def test_parse_telegram_updates():
    t = parse_update({"update_id": 1, "message": {"message_id": 5, "chat": {"id": 42}, "text": "yo"}})
    assert (t.kind, t.sender, t.text, t.external_id) == ("text", "42", "yo", "42:5")
    v = parse_update({"update_id": 2, "message": {"message_id": 6, "chat": {"id": 42}, "voice": {"file_id": "F"}}})
    assert (v.kind, v.media_ref) == ("voice", "F")
    c = parse_update({"update_id": 3, "callback_query": {"id": "Q", "data": "deep:2",
                                                          "message": {"chat": {"id": 42}}}})
    assert (c.kind, c.button_id, c.external_id) == ("button", "deep:2", "cb:Q")


def test_split_long_text():
    parts = _split("a" * 9000, 4000)
    assert [len(p) for p in parts] == [4000, 4000, 1000]


def test_jev_normalized_score():
    assert normalized_score({"score": 2.0, "legend": {"0": "a", "1": "b", "2": "c"}}) == 1.0
    assert normalized_score({"score": 1.5, "legend": {str(i): "x" for i in range(4)}}) == 0.5


def test_select_top_quotas_and_injection():
    def row(i, sec, comp, inj=0.0):
        return ItemScore(item_id=i, run_id="r", ranker="llm", relevance=comp, substance=comp,
                         section=sec, composite=comp, injection=inj)

    rows = [row(1, "ai", 0.9), row(2, "ai", 0.8, inj=0.9), row(3, "ai", 0.7), row(4, "ai", 0.6),
            row(5, "ai", 0.55), row(6, "ai", 0.5), row(7, "world", 0.2), row(8, "world", 0.6),
            row(9, "fun", 0.1), row(10, "skip", 0.99)]
    picked = select_top(rows, quotas={"ai": 4, "world": 3, "fun": 1})
    assert picked["ai"] == [1, 3, 4, 5]          # injection-flagged item 2 dropped
    assert picked["world"] == [8]                # below MIN_COMPOSITE dropped
    assert picked["fun"] == [9]                  # fun has no floor
    assert composite(1, 1, 1) > composite(1, 1, 0)


def test_select_top_caps_per_source():
    rows = [ItemScore(item_id=i, run_id="r", ranker="llm", relevance=1, substance=1, section="ai",
                      composite=1 - i / 100, injection=0.0) for i in range(1, 7)]
    sources = {1: 10, 2: 10, 3: 10, 4: 20, 5: 30, 6: 40}  # items 1-3 from the same source
    assert select_top(rows, sources, {"ai": 4})["ai"] == [1, 2, 4, 5]


def test_select_top_excludes_topics():
    rows = [ItemScore(item_id=1, run_id="r", ranker="llm", relevance=1, substance=1, section="world",
                      topic="sports", composite=0.9, injection=0.0),
            ItemScore(item_id=2, run_id="r", ranker="llm", relevance=1, substance=1, section="world",
                      topic="geopolitics", composite=0.8, injection=0.0)]
    assert select_top(rows, quotas={"world": 2}, exclude_topics=["sports"])["world"] == [2]


def _prefs(**kw):
    p = {"weekday_time": "07:30", "weekend_time": "09:00", "minutes": 8, "sections": {"ai": 3},
         "exclude_topics": [], "focus": "", "paused": False, "interests": "x"}
    p.update(kw)
    return p


def test_scheduler_decide():
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    from app.scheduler import decide

    tz = ZoneInfo("Asia/Kolkata")
    wed = lambda h, m: datetime(2026, 9, 30, h, m, tzinfo=tz)  # a Wednesday
    sat = lambda h, m: datetime(2026, 10, 3, h, m, tzinfo=tz)
    utc_now = datetime.now(timezone.utc)

    assert decide(wed(6, 59), _prefs(), [], 30) is None
    assert decide(wed(7, 0), _prefs(), [], 30) == "build"
    assert decide(wed(7, 10), _prefs(), [("building", utc_now)], 30) is None           # build in progress
    assert decide(wed(7, 30), _prefs(), [("ready", utc_now)], 30) == "deliver"
    assert decide(wed(7, 45), _prefs(), [], 30) == "build_and_deliver"                 # woke up late
    assert decide(wed(7, 45), _prefs(), [("sent", utc_now)], 30) is None
    assert decide(wed(13, 0), _prefs(), [], 30) is None                                # past the 4h grace
    assert decide(wed(7, 10), _prefs(), [("failed", utc_now)], 30) == "build"          # one retry
    assert decide(wed(7, 10), _prefs(), [("failed", utc_now)] * 2, 30) is None
    assert decide(wed(7, 10), _prefs(paused=True), [], 30) is None
    assert decide(sat(7, 30), _prefs(), [], 30) is None                                # weekend time is 09:00
    assert decide(sat(8, 30), _prefs(), [], 30) == "build"
    started = wed(7, 0).astimezone(timezone.utc)
    assert decide(wed(7, 10), _prefs(), [("building", started)], 30) is None           # 10 min old: still running
    assert decide(wed(7, 29), _prefs(), [("building", started - timedelta(hours=1))], 30) == "build"  # crashed


def test_prefs_validate():
    from app.prefs import validate

    clean, errors = validate({"weekday_time": "8:15", "minutes": 5, "sections": {"india": 3, "sports": 2},
                              "exclude_topics": ["Sports", "crypto"]})
    assert clean["weekday_time"] == "08:15" and clean["minutes"] == 5
    assert clean["sections"] == {"india": 3} and clean["exclude_topics"] == ["crypto", "sports"]
    assert any("sports" in e for e in errors)
    _, errors = validate({"weekday_time": "25:00", "minutes": 99})
    assert len(errors) == 2


def test_select_top_skips_same_story_from_other_outlet():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.77, (1 - 0.77**2) ** 0.5], dtype=np.float32)  # cos(a, b) = 0.77
    c = np.array([0.0, 1.0], dtype=np.float32)
    rows = [ItemScore(item_id=i, run_id="r", ranker="llm", relevance=1, substance=1, section="ai",
                      composite=1 - i / 10, injection=0.0) for i in (1, 2, 3)]
    assert select_top(rows, quotas={"ai": 3}, vectors={1: a, 2: b, 3: c})["ai"] == [1, 3]


def test_telegram_html_sources():
    from app.channels.telegram import links_html, to_html

    assert to_html("a < b & **c**") == "a &lt; b &amp; <b>c</b>"
    out = links_html([("thehindu.com", "https://www.thehindu.com/x?a=1&b=2"), ("NDTV", "https://ndtv.com/y")])
    assert out == ('Sources: <a href="https://www.thehindu.com/x?a=1&amp;b=2">thehindu.com</a> · '
                   '<a href="https://ndtv.com/y">NDTV</a>')


def test_schedules_next_run():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from app.schedules import ScheduleError, next_run, normalize

    tz = ZoneInfo("Asia/Kolkata")
    wed_17 = datetime(2026, 9, 30, 17, 0, tzinfo=tz)   # Wednesday 5pm
    fri_19 = datetime(2026, 10, 2, 19, 0, tzinfo=tz)   # Friday 7pm

    wk = normalize({"repeat": "weekdays", "time": "18:00"}, wed_17)
    assert next_run(wk, wed_17) == datetime(2026, 9, 30, 18, 0, tzinfo=tz)        # later today
    assert next_run(wk, fri_19) == datetime(2026, 10, 5, 18, 0, tzinfo=tz)        # skips the weekend -> Monday

    weekly = normalize({"repeat": "weekly", "time": "7:00", "days": ["Sunday", "wed"]}, wed_17)
    assert weekly["days"] == ["wed", "sun"]
    assert next_run(weekly, wed_17) == datetime(2026, 10, 4, 7, 0, tzinfo=tz)     # next Sunday

    once = normalize({"repeat": "once", "time": "09:00"}, wed_17)                 # no date -> tomorrow 9am
    assert once["date"] == "2026-10-01"
    first = next_run(once, wed_17)
    assert first == datetime(2026, 10, 1, 9, 0, tzinfo=tz)
    assert next_run(once, first, last_run=first) is None                          # one-off is finished

    hourly = normalize({"repeat": "every_n_hours", "every_hours": 3}, wed_17)
    assert next_run(hourly, wed_17, last_run=wed_17) == datetime(2026, 9, 30, 20, 0, tzinfo=tz)

    with pytest.raises(ScheduleError):
        normalize({"repeat": "every_n_hours", "every_hours": 0}, wed_17)          # no "every 5 minutes"
    with pytest.raises(ScheduleError):
        normalize({"repeat": "weekly", "time": "09:00", "days": []}, wed_17)
    with pytest.raises(ScheduleError):
        normalize({"repeat": "daily", "time": "25:00"}, wed_17)


def _pool_test_crash():
    """Module-level so it's importable/picklable under spawn -- simulates a worker hard-crashing."""
    import os

    os._exit(1)


def _pool_test_add_one(x):
    return x + 1


async def test_extraction_pool_recovers_from_a_real_crashed_worker():
    """The actual recovery mechanism, end to end: a real subprocess hard-exits (same symptom as the SIGABRT
    we hit in production), BrokenProcessPool surfaces instead of killing us, and a fresh pool still works."""
    import asyncio

    from concurrent.futures.process import BrokenProcessPool

    from app.briefing import ingest

    await ingest._reset_pool()
    try:
        pool = await ingest._get_pool()
        loop = asyncio.get_running_loop()
        with pytest.raises(BrokenProcessPool):
            await loop.run_in_executor(pool, _pool_test_crash)

        await ingest._reset_pool()  # what _extract_fulltext does on this exception
        pool2 = await ingest._get_pool()
        assert pool2 is not pool
        result = await loop.run_in_executor(pool2, _pool_test_add_one, 41)
        assert result == 42
    finally:
        await ingest._reset_pool()


async def test_extract_fulltext_contains_a_broken_pool(monkeypatch):
    """_extract_fulltext's own exception handling: a BrokenProcessPool must become "", not propagate, and
    must trigger a pool reset so the next call gets a fresh one."""
    import concurrent.futures
    from concurrent.futures.process import BrokenProcessPool

    from app.briefing import ingest

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body>hello</body></html>"

    class FakeClient:
        async def get(self, url, follow_redirects=True):
            return FakeResponse()

    class BrokenExecutor:
        def submit(self, func, *args):
            fut = concurrent.futures.Future()
            fut.set_exception(BrokenProcessPool("worker died"))
            return fut

    class WorkingExecutor:
        def submit(self, func, *args):
            fut = concurrent.futures.Future()
            fut.set_result(func(*args))
            return fut

    reset_calls = []
    real_reset = ingest._reset_pool

    async def spy_reset():
        reset_calls.append(1)
        await real_reset()

    pools = iter([BrokenExecutor(), WorkingExecutor()])
    monkeypatch.setattr(ingest, "_get_pool", lambda: _async_result(next(pools)))
    monkeypatch.setattr(ingest, "_reset_pool", spy_reset)

    result = await ingest._extract_fulltext(FakeClient(), "https://example.test/crashy")
    assert result == ""
    assert reset_calls == [1]

    result = await ingest._extract_fulltext(FakeClient(), "https://example.test/fine")
    assert result == "hello"


async def _async_result(value):
    return value


