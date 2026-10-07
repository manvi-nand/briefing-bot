"""External uptime alerting: pings a 'dead man's switch' URL (e.g. Healthchecks.io) on a schedule.

The app can't notify you if the whole process is down -- it has nothing left to send a message with. So this
pings OUT to an external monitor instead: as long as pings keep arriving on schedule, the monitor stays quiet.
If they stop (process crashed, container down, host unreachable), the monitor notices the silence itself and
alerts you by email/SMS/whatever you configured there -- outside Telegram, so it still reaches you.

Setup (free): create an account at healthchecks.io, add a check for "alive" (expect a ping every ~10 minutes)
and optionally one for "briefing" (expect one ping a day, around your briefing time), and put their ping URLs
in HEARTBEAT_URL / BRIEFING_HEARTBEAT_URL. Leave either blank to disable it -- nothing else here depends on it.
"""

import logging

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


async def ping(kind: str = "alive") -> None:
    """Fire-and-forget; a monitoring ping must never be able to break anything else."""
    s = get_settings()
    url = {"alive": s.heartbeat_url, "briefing": s.briefing_heartbeat_url}.get(kind, "")
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.get(url)
    except Exception:
        log.debug("heartbeat ping (%s) failed; will retry next cycle", kind)
