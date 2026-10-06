"""Client-side tools for the assistant. v1 tools are read-only or write only to your own trackers/memory."""

import json
import logging
from datetime import datetime, timedelta, timezone

import httpx
import trafilatura
from google.genai import types
from sqlalchemy import select

from app import llm, trackers
from app.briefing.embed import embed
from app.briefing.pipeline import episode_stories, latest_episode
from app.briefing.rank import TOPICS
from app.prefs import SECTIONS as BRIEF_SECTIONS
from app.prefs import describe, get_prefs, time_for, update_prefs
from app.schedules import DAYS, REPEATS, ScheduleError, next_run, normalize
from app.schedules import describe as describe_schedule
from app.config import get_settings
from app.db import session_scope
from app.models import Episode, Item, Tracker

log = logging.getLogger(__name__)


def _tool(name: str, description: str, props: dict, required: list[str] | None = None) -> types.FunctionDeclaration:
    req = list(props) if required is None else required
    params = {"type": "object", "properties": props, "required": req} if props else None
    return types.FunctionDeclaration(name=name, description=description, parameters_json_schema=params)


S = {"type": "string"}
I = {"type": "integer"}
SCHED = {"repeat": {"type": "string", "enum": REPEATS,
                      "description": "once | daily | weekdays (Mon-Fri) | weekly (on `days`) | every_n_hours"},
           "time": {**S, "description": "24h HH:MM local time (not needed for every_n_hours)"},
           "date": {**S, "description": "YYYY-MM-DD, only for repeat=once; omit for the next occurrence of `time`"},
           "days": {"type": "array", "items": {"type": "string", "enum": DAYS}, "description": "for weekly"},
           "every_hours": {**I, "description": "for every_n_hours (1-168)"}}

TOOLS = [
    _tool("web_search", "Search the web (Google) for current or factual information. Returns a short answer "
          "with source links. Use it whenever facts may have changed or you are unsure.", {"query": S}),
    _tool("read_url", "Fetch a web page and return its main text (for links the user shares or a story URL).",
          {"url": S}),
    _tool("get_briefing_settings", "Show the daily briefing settings: delivery time, length, section mix, skipped "
          "topics, extra focus, and the interests used for ranking.", {}),
    _tool("update_briefing_settings",
          "Change the daily briefing. Include ONLY the fields the user wants to change. Use this (not "
          "remember_fact) for anything about the briefing: time, length, what to include/skip, pausing.",
          {"weekday_time": {**S, "description": "Mon-Fri delivery time, 24h HH:MM, e.g. 08:15"},
           "weekend_time": {**S, "description": "Sat-Sun delivery time, 24h HH:MM"},
           "minutes": {**I, "description": "target length in minutes (2-20)"},
           "sections": {"type": "object",
                        "description": "stories per section to set (0 removes a section); unspecified sections "
                                       "keep their current count",
                        "properties": {k: I for k in BRIEF_SECTIONS}},
           "exclude_topics": {"type": "array", "items": {"type": "string", "enum": TOPICS},
                              "description": "the FULL list of topics to skip (replaces the current list)"},
           "focus": {**S, "description": "free-text extra instruction for the briefing, e.g. 'more on Indian "
                                         "startups'; empty string clears it"},
           "paused": {"type": "boolean", "description": "true pauses daily briefings, false resumes"},
           "interests": {**S, "description": "full updated description of the user's interests for ranking"}},
          required=[]),
    _tool("get_story", "Get the full stored text, source and link of a story from the latest briefing by its "
          "number (e.g. 'story 2' -> 2). Use this before answering questions about a briefing story.",
          {"number": I}),
    _tool("search_past_stories", "Semantic search over stories ingested in the last 30 days (including ones not "
          "chosen for a briefing). Returns titles, links and snippets.", {"query": S}),
    _tool("remember_fact", "Save a fact or preference the user explicitly stated or confirmed (e.g. briefing "
          "time, interests, preferred tone). Never store health, finance or identity-document details.",
          {"key": {**S, "description": "short snake_case key, e.g. briefing_time"}, "value": S}),
    _tool("list_memory", "List everything stored about the user.", {}),
    _tool("forget_fact", "Delete a stored fact by key.", {"key": S}),
    _tool("add_to_list", "Add an item to a named list such as 'reading' or 'todo'.",
          {"list": S, "item": S}),
    _tool("show_list", "Show open items on a named list. Use an empty string for all lists.", {"list": S}),
    _tool("watch_topic", "Create a topic watch: alert the user when new ingested news matches.",
          {"query": {**S, "description": "what to watch for, e.g. 'Sarvam releases a new model'"},
           "keywords": {"type": "array", "items": S,
                        "description": "optional words that must appear in the headline; [] for none"}}),
    _tool("set_reminder", "Send the user a FIXED reminder text at a time, once or repeating. For anything that "
          "needs fresh information at that time (prices, scores, weather, news), use schedule_task instead.",
          {"text": S, **SCHED}, required=["text", "repeat"]),
    _tool("schedule_task",
          "Run an instruction later (once or repeating) and send the user the fresh answer each time, e.g. "
          "'every weekday at 6pm send me the Nifty and Sensex close'. Write `instruction` as a standalone request "
          "that will make sense when run later without this conversation.",
          {"title": {**S, "description": "short label, e.g. 'Nifty/Sensex close'"},
           "instruction": {**S, "description": "what to do at run time, e.g. 'Give today's Nifty 50 and Sensex "
                                               "closing levels and % change, with a one-line reason.'"},
           **SCHED}, required=["title", "instruction", "repeat"]),
    _tool("update_schedule", "Change, pause or resume an existing reminder or scheduled task by id (from "
          "list_trackers). Include only fields that change; to change timing send the full new schedule.",
          {"id": I, "paused": {"type": "boolean"}, "instruction": S, "title": S, "text": S, **SCHED},
          required=["id"]),
    _tool("list_trackers", "List open trackers with ids and schedules. kind: 'list', 'watch', 'reminder', 'task' "
          "or '' for all.", {"kind": S}),
    _tool("close_tracker", "Mark a tracker done (completed list item) or cancelled (delete a reminder, task or "
          "watch).", {"id": I, "status": {"type": "string", "enum": ["done", "cancelled"]}}),
    _tool("adjust_preference", "Change how much a topic is favoured in future briefings, e.g. 'less crypto'.",
          {"topic": {"type": "string", "enum": TOPICS}, "direction": {"type": "string", "enum": ["more", "less"]}}),
    _tool("rate_story", "Record the user's rating of a briefing story (thumbs up/down) with an optional note.",
          {"number": I, "rating": {"type": "string", "enum": ["up", "down"]}, "note": S}),
]


def _tz():
    return get_settings().tz


async def _get_story(number: int) -> str:
    ep = await latest_episode()
    if not ep:
        return "No briefing exists yet."
    for n, item, source in await episode_stories(ep):
        if n == number:
            return json.dumps({"number": n, "title": item.title, "source": source, "url": item.url,
                               "published_at": item.published_at.isoformat() if item.published_at else None,
                               "text": (item.full_text or item.summary)[:8000]})
    return f"Story {number} not found; the latest briefing ({ep.date}) has {len(ep.story_order or [])} stories."


async def _search_past(query: str) -> str:
    vec = (await embed([query]))[0].tolist()
    since = datetime.now(timezone.utc) - timedelta(days=30)
    async with session_scope() as s:
        rows = (await s.scalars(
            select(Item).where(Item.embedding.is_not(None), Item.fetched_at >= since)
            .order_by(Item.embedding.cosine_distance(vec)).limit(6)
        )).all()
    return json.dumps([{"title": it.title, "url": it.url, "date": str(it.fetched_at.date()),
                        "snippet": (it.summary or it.full_text)[:300]} for it in rows])


async def _update_briefing(args: dict) -> str:
    prefs, errors = await update_prefs(args)
    now = datetime.now(_tz())
    today_at = time_for(prefs, now.date())
    async with session_scope() as s:
        sent_today = await s.scalar(
            select(Episode.id).where(Episode.date == now.date(), Episode.status == "sent").limit(1)
        )
    if prefs["paused"]:
        nxt = "Briefings are paused."
    elif today_at > f"{now:%H:%M}" and not sent_today:
        nxt = f"Next briefing: today at {today_at}."
    else:
        tomorrow = (now + timedelta(days=1)).date()
        nxt = f"Next briefing: tomorrow at {time_for(prefs, tomorrow)}."
    msg = f"Updated. {nxt}\n{describe(prefs)}"
    if errors:
        msg += "\nNot applied: " + "; ".join(errors)
    return msg


MAX_OPEN_TASKS = 15


async def _create_scheduled(kind: str, payload: dict, args: dict) -> str:
    now = datetime.now(_tz())
    try:
        sched = normalize(args, now)
    except ScheduleError as e:
        return f"Could not schedule: {e}. Ask the user to clarify."
    if kind == "task" and len(await trackers.list_trackers("task")) >= MAX_OPEN_TASKS:
        return f"The user already has {MAX_OPEN_TASKS} scheduled tasks; ask them to delete one first."
    nxt = next_run(sched, now)
    if not nxt:
        return "That time is in the past; ask the user for a future time."
    t = await trackers.add_tracker(kind, {**payload, "schedule": sched}, due_at=nxt)
    label = "Reminder" if kind == "reminder" else "Task"
    return f"{label} #{t.id} scheduled {describe_schedule(sched)}. First run: {nxt:%a %d %b %H:%M}."


async def _update_scheduled(args: dict) -> str:
    async with session_scope() as s:
        t = await s.get(Tracker, int(args["id"]))
        if not t or t.kind not in ("reminder", "task") or t.status != "open":
            return "No open reminder or task with that id."
        payload = dict(t.payload)
        now = datetime.now(_tz())
        for key in ("instruction", "title", "text"):
            if args.get(key):
                payload[key] = args[key]
        if args.get("repeat"):
            try:
                payload["schedule"] = normalize(args, now)
            except ScheduleError as e:
                return f"Could not update: {e}."
        if args.get("paused") is not None:
            payload["paused"] = bool(args["paused"])
        sched = payload.get("schedule")
        if sched and (args.get("repeat") or args.get("paused") is False):
            nxt = next_run(sched, now)
            if not nxt:
                return "That time is in the past; ask the user for a future time."
            t.due_at = nxt
        t.payload = payload
        await s.flush()
        return "Updated: " + _fmt_tracker(t)


async def _read_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Only http(s) URLs are supported."
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (personal-assistant)"}) as client:
        r = await client.get(url)
    if r.status_code >= 400:
        return f"Fetch failed: HTTP {r.status_code}"
    text = trafilatura.extract(r.text) if "html" in r.headers.get("content-type", "") else r.text
    return (text or "No readable text found.")[:8000]


def _fmt_tracker(t) -> str:
    p = t.payload or {}
    if t.kind in ("reminder", "task"):
        sched = describe_schedule(p["schedule"]) if p.get("schedule") else "once"
        nxt = "paused" if p.get("paused") else f"next {t.due_at.astimezone(_tz()):%a %d %b %H:%M}"
        what = f"reminder: {p.get('text')}" if t.kind == "reminder" else f"task '{p.get('title')}': {p.get('instruction')}"
        return f"#{t.id} {what} ({sched}; {nxt})"
    if t.kind == "watch":
        return f"#{t.id} watch: {p.get('query')}"
    return f"#{t.id} [{p.get('list')}] {p.get('item')}"


async def run_tool(name: str, args: dict, *, source_message_id: int | None = None) -> str:
    match name:
        case "web_search":
            return json.dumps(await llm.grounded_search(args["query"]))
        case "read_url":
            return await _read_url(args["url"])
        case "get_briefing_settings":
            return describe(await get_prefs())
        case "update_briefing_settings":
            return await _update_briefing(args)
        case "get_story":
            return await _get_story(int(args["number"]))
        case "search_past_stories":
            return await _search_past(args["query"])
        case "remember_fact":
            return await trackers.remember(args["key"], args["value"], source_message_id)
        case "list_memory":
            facts = await trackers.list_facts()
            return "\n".join(f"{f.key}: {f.value}" for f in facts) or "Nothing stored yet."
        case "forget_fact":
            return "Deleted." if await trackers.forget(args["key"]) else f"No fact named {args['key']}."
        case "add_to_list":
            t = await trackers.add_tracker("list", {"list": args["list"].lower(), "item": args["item"]})
            return f"Saved #{t.id} to {args['list']} list."
        case "show_list":
            rows = await trackers.list_trackers("list", args["list"] or None)
            return "\n".join(_fmt_tracker(t) for t in rows) or "That list is empty."
        case "watch_topic":
            t = await trackers.add_tracker("watch", {"query": args["query"], "keywords": args.get("keywords") or []})
            return f"Watching (#{t.id}): {args['query']}"
        case "set_reminder":
            return await _create_scheduled("reminder", {"text": args["text"]}, args)
        case "schedule_task":
            return await _create_scheduled(
                "task", {"title": args["title"], "instruction": args["instruction"]}, args
            )
        case "update_schedule":
            return await _update_scheduled(args)
        case "list_trackers":
            rows = await trackers.list_trackers(args["kind"] or None)
            return "\n".join(_fmt_tracker(t) for t in rows) or "No open trackers."
        case "close_tracker":
            ok = await trackers.set_tracker_status(int(args["id"]), args["status"])
            return "Updated." if ok else "No tracker with that id."
        case "adjust_preference":
            w = await trackers.adjust_topic(args["topic"], 0.2 if args["direction"] == "more" else -0.2)
            return f"Topic '{args['topic']}' weight is now {w:+.1f}."
        case "rate_story":
            ep = await latest_episode()
            order = (ep.story_order if ep else None) or []
            n = int(args["number"])
            if not 1 <= n <= len(order):
                return "No such story in the latest briefing."
            return await trackers.record_story_feedback(
                order[n - 1], 1 if args["rating"] == "up" else -1, args.get("note", "")
            )
    return f"Unknown tool {name}"
