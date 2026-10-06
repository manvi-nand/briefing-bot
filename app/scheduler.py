"""In-process scheduler. A one-minute tick reads the user's briefing prefs (changeable by chat), so a new time
takes effect immediately without restarting. Also: reminders, topic watches, weekly summary."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app import trackers
from app.briefing import dedupe, ingest, pipeline
from app.channels import primary_channel
from app.config import get_settings
from app.db import session_scope
from app.models import Episode
from app.prefs import get_prefs, time_for

log = logging.getLogger(__name__)

GRACE = timedelta(hours=4)  # if the laptop/server was asleep at briefing time, still send within this window
STALE_BUILD = timedelta(minutes=30)
MAX_FAILED_BUILDS = 2

# Shared with the manual /build command (app/handlers.py) so the two can never run concurrently.
build_lock = asyncio.Lock()


def decide(now: datetime, prefs: dict, todays: list[tuple[str, datetime]], lead_minutes: int) -> str | None:
    """What the tick should do now: 'build', 'deliver', 'build_and_deliver' or None.

    `now` is local time; `todays` is [(status, created_at_utc)] for today's episodes.
    """
    if prefs["paused"]:
        return None
    hh, mm = map(int, time_for(prefs, now.date()).split(":"))
    deliver_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    build_at = deliver_at - timedelta(minutes=lead_minutes)
    statuses = [st for st, _ in todays]

    if "sent" in statuses:
        return None
    if "ready" in statuses:
        return "deliver" if deliver_at <= now < deliver_at + GRACE else None
    fresh_build = any(st == "building" and now.astimezone(timezone.utc) - ts < STALE_BUILD for st, ts in todays)
    if fresh_build or statuses.count("failed") >= MAX_FAILED_BUILDS:
        return None
    if build_at <= now < deliver_at:
        return "build"
    if deliver_at <= now < deliver_at + GRACE:
        return "build_and_deliver"
    return None


async def briefing_tick() -> None:
    if build_lock.locked():
        return
    async with build_lock:
        s = get_settings()
        now = datetime.now(s.tz)
        prefs = await get_prefs()
        async with session_scope() as sess:
            eps = list((await sess.scalars(
                select(Episode).where(Episode.date == now.date()).order_by(Episode.created_at)
            )).all())
        action = decide(now, prefs, [(e.status, e.created_at) for e in eps], s.briefing_build_lead_minutes)
        if not action:
            return
        log.info("briefing tick: %s", action)
        try:
            if action == "deliver":
                ready = [e for e in eps if e.status == "ready"][-1]
                await pipeline.deliver_notice(ready)
                return
            ep = await pipeline.build_episode()
            if action == "build_and_deliver":
                await pipeline.deliver_notice(ep)
        except Exception as e:
            log.exception("scheduled briefing failed")
            ch = primary_channel()
            await ch.send_proactive(ch.owner, f"Today's briefing failed to build: {e}")


async def watch_job() -> None:
    """Light ingest every few hours so topic watches fire during the day too."""
    try:
        new_ids = await ingest.ingest()
        await dedupe.embed_and_dedupe(new_ids)
        await trackers.check_watches(new_ids)
    except Exception:
        log.exception("watch job failed")


def build_scheduler() -> AsyncIOScheduler:
    s = get_settings()
    sched = AsyncIOScheduler(timezone=s.tz)
    sched.add_job(briefing_tick, "interval", minutes=1, id="briefing", coalesce=True, max_instances=1)
    sched.add_job(trackers.due_tick, "interval", minutes=1, id="reminders", coalesce=True, max_instances=1)
    sched.add_job(watch_job, "cron", hour="10,14,18,22", minute=5, id="watches", coalesce=True)
    sched.add_job(pipeline.cleanup_audio, "cron", hour=3, minute=30, id="cleanup_audio", coalesce=True)
    sched.add_job(trackers.send_weekly_summary, "cron", day_of_week="sun", hour=19, minute=0, id="weekly")
    log.info("scheduler started (briefing time comes from chat-editable prefs; see /settings)")
    return sched
