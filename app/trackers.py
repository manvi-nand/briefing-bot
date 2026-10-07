"""Memory facts, trackers (lists / topic watches / reminders), feedback weights. Shared by agent + scheduler."""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import delete, func, select

from app import costs
from app.briefing.embed import embed
from app.channels import primary_channel
from app.config import get_settings
from app.db import session_scope
from app.models import Feedback, Item, MemoryFact, Tracker, TopicWeight

log = logging.getLogger(__name__)

WATCH_THRESHOLD = 0.62
FEEDBACK_STEP = 0.1


# ---------------------------------------------------------------- memory facts

async def remember(key: str, value: str, source_message_id: int | None = None) -> str:
    key = key.strip().lower().replace(" ", "_")[:100]
    async with session_scope() as s:
        fact = await s.scalar(select(MemoryFact).where(MemoryFact.key == key))
        if fact:
            fact.value = value
        else:
            s.add(MemoryFact(key=key, value=value, source_message_id=source_message_id))
    return f"Saved: {key.replace('_', ' ')} = {value}"


async def list_facts() -> list[MemoryFact]:
    async with session_scope() as s:
        return list((await s.scalars(select(MemoryFact).order_by(MemoryFact.key))).all())


async def forget(key: str) -> bool:
    key = key.strip().lower().replace(" ", "_")
    async with session_scope() as s:
        res = await s.execute(delete(MemoryFact).where(MemoryFact.key == key))
    return res.rowcount > 0


# ---------------------------------------------------------------- trackers

async def add_tracker(kind: str, payload: dict, due_at: datetime | None = None) -> Tracker:
    async with session_scope() as s:
        t = Tracker(kind=kind, payload=payload, due_at=due_at)
        s.add(t)
        await s.flush()
        return t


async def list_trackers(kind: str | None = None, list_name: str | None = None, include_done: bool = False) -> list[Tracker]:
    async with session_scope() as s:
        q = select(Tracker).order_by(Tracker.created_at)
        if kind:
            q = q.where(Tracker.kind == kind)
        if not include_done:
            q = q.where(Tracker.status == "open")
        rows = list((await s.scalars(q)).all())
    if list_name:
        rows = [r for r in rows if (r.payload or {}).get("list", "").lower() == list_name.lower()]
    return rows


async def set_tracker_status(tracker_id: int, status: str) -> bool:
    async with session_scope() as s:
        t = await s.get(Tracker, tracker_id)
        if not t:
            return False
        t.status = status
    return True


MAX_TASK_RUNS_PER_DAY = 20  # guard against runaway schedules ("every hour" x many tasks)
STALE_AFTER = timedelta(hours=3)  # a repeating run missed by more than this (laptop asleep) is skipped


async def _task_runs_today() -> int:
    from app.models import UsageEvent

    start = datetime.now(get_settings().tz).replace(hour=0, minute=0, second=0, microsecond=0)
    async with session_scope() as s:
        return await s.scalar(
            select(func.count()).where(UsageEvent.stage == "task_run", UsageEvent.created_at >= start)
        ) or 0


async def _run_task(t: Tracker) -> None:
    from app import costs
    from app.agent.agent import respond
    from app.models import Message

    ch = primary_channel()
    title = t.payload.get("title") or "Scheduled task"
    if await _task_runs_today() >= MAX_TASK_RUNS_PER_DAY:
        await ch.send_proactive(ch.owner, f"⏸ Skipped '{title}': the daily limit of {MAX_TASK_RUNS_PER_DAY} "
                                          "scheduled task runs was reached.")
        return
    try:
        answer = await respond(t.payload["instruction"], channel=ch.name, scheduled=title)
    except costs.BudgetExceeded:
        await ch.send_proactive(ch.owner, f"⏸ Skipped '{title}': today's budget is used up.")
        return
    await costs.record("pipeline", "task_run", {"tracker_id": t.id}, 0.0)
    await ch.send_proactive(ch.owner, f"⏰ {title}\n\n{answer.text}", answer.sources)
    async with session_scope() as s:  # so follow-ups like "tell me more" have context
        s.add(Message(channel=ch.name, direction="out", type="text", text=f"[{title}] {answer.text}"))


async def due_tick() -> int:
    """Run reminders and scheduled tasks whose time has come; reschedule repeating ones. Runs every minute."""
    from app.schedules import next_run

    now = datetime.now(timezone.utc)
    local_now = now.astimezone(get_settings().tz)
    async with session_scope() as s:
        due = list((await s.scalars(
            select(Tracker).where(Tracker.kind.in_(["reminder", "task"]), Tracker.status == "open",
                                  Tracker.due_at <= now)
        )).all())
    due = [t for t in due if not (t.payload or {}).get("paused")]
    ch = primary_channel()
    for t in due:
        sched = (t.payload or {}).get("schedule")
        repeating = bool(sched) and sched["repeat"] != "once"
        stale = repeating and now - t.due_at > STALE_AFTER
        try:
            if stale:
                log.info("skipping missed run of tracker %s (due %s)", t.id, t.due_at)
            elif t.kind == "reminder":
                await ch.send_proactive(ch.owner, f"⏰ Reminder: {t.payload.get('text', '')}")
            else:
                await _run_task(t)
        except Exception:
            log.exception("tracker %s failed", t.id)
        async with session_scope() as s:
            row = await s.get(Tracker, t.id)
            if not row:
                continue
            nxt = next_run(sched, local_now, last_run=local_now) if repeating else None
            payload = dict(row.payload)
            if not stale:
                payload["runs"] = payload.get("runs", 0) + 1
                payload["last_run"] = now.isoformat()
            row.payload = payload
            if nxt:
                row.due_at = nxt
            else:
                row.status = "done"
    return len(due)


async def check_watches(item_ids: list[int] | None = None) -> int:
    """Alert when newly ingested items match an open topic watch (embedding similarity + keywords)."""
    watches = await list_trackers("watch")
    if not watches:
        return 0
    since = datetime.now(timezone.utc) - timedelta(hours=26)
    async with session_scope() as s:
        q = select(Item).where(Item.embedding.is_not(None), Item.duplicate_of.is_(None))
        q = q.where(Item.id.in_(item_ids)) if item_ids else q.where(Item.fetched_at >= since)
        items = list((await s.scalars(q)).all())
    if not items:
        return 0
    item_vecs = np.array([it.embedding for it in items], dtype=np.float32)
    watch_vecs = await embed([w.payload["query"] for w in watches])
    sent = 0
    ch = primary_channel()
    for w, wv in zip(watches, watch_vecs, strict=True):
        seen = set(w.payload.get("alerted_item_ids", []))
        keywords = [k.lower() for k in w.payload.get("keywords", [])]
        sims = item_vecs @ wv
        hits = []
        for it, sim in zip(items, sims, strict=True):
            if it.id in seen:
                continue
            kw = any(k in it.title.lower() for k in keywords) if keywords else True
            if sim >= WATCH_THRESHOLD and kw:
                hits.append(it)
        if not hits:
            continue
        lines = "\n".join(f"• {it.title}\n  {it.url}" for it in hits[:3])
        await ch.send_proactive(ch.owner, f"👀 Watch '{w.payload['query']}':\n{lines}")
        async with session_scope() as s:
            t = await s.get(Tracker, w.id)
            t.payload = {**t.payload, "alerted_item_ids": list(seen | {it.id for it in hits})}
        sent += 1
    return sent


# ---------------------------------------------------------------- feedback / ranking weights

async def adjust_topic(topic: str, delta: float) -> float:
    async with session_scope() as s:
        tw = await s.get(TopicWeight, topic)
        if not tw:
            tw = TopicWeight(topic=topic, weight=0.0)
            s.add(tw)
        tw.weight = max(-1.0, min(1.0, tw.weight + delta))
        tw.updated_at = datetime.now(timezone.utc)
        return tw.weight


async def record_story_feedback(item_id: int, rating: int, note: str = "") -> str:
    async with session_scope() as s:
        item = await s.get(Item, item_id)
        s.add(Feedback(item_id=item_id, rating=rating, note=note))
        topic = item.topic if item else None
    if topic and rating:
        w = await adjust_topic(topic, FEEDBACK_STEP * rating)
        return f"Noted. Topic '{topic}' weight is now {w:+.1f}."
    return "Noted, thanks."


async def weekly_summary() -> str:
    since = datetime.now(timezone.utc) - timedelta(days=7)
    async with session_scope() as s:
        ups = await s.scalar(select(func.count()).where(Feedback.created_at >= since, Feedback.rating > 0))
        downs = await s.scalar(select(func.count()).where(Feedback.created_at >= since, Feedback.rating < 0))
        changed = list(
            (await s.scalars(select(TopicWeight).where(TopicWeight.updated_at >= since).order_by(TopicWeight.weight.desc()))).all()
        )
        notes = list((await s.scalars(select(Feedback.note).where(Feedback.created_at >= since, Feedback.note != ""))).all())
    week_cost = sum(c for _, _, c in await costs.cost_report(7))
    lines = [f"📊 Weekly personalization summary: {ups or 0} 👍, {downs or 0} 👎.",
             f"💰 Spent ${week_cost:.2f} this week (~${week_cost / 7:.2f}/day)."]
    if changed:
        lines.append("Topic weights changed this week:")
        lines += [f"• {tw.topic}: {tw.weight:+.1f}" for tw in changed]
    if notes:
        lines.append("Your notes: " + "; ".join(notes[:5]))
    if len(lines) == 2:  # just the summary + cost lines, nothing else happened
        lines.append("No changes this week.")
    return "\n".join(lines)


async def send_weekly_summary() -> None:
    ch = primary_channel()
    await ch.send_proactive(ch.owner, await weekly_summary())
