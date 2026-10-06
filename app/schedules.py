"""Schedules for reminders and scheduled tasks: once, daily, weekdays, weekly (chosen days) or every N hours.

A reminder sends fixed text. A task re-runs an instruction through the assistant (web search etc.) each time and
sends the fresh answer, e.g. "every weekday at 6pm send me the Nifty and Sensex close".
"""

import re
from datetime import date, datetime, time, timedelta

REPEATS = ["once", "daily", "weekdays", "weekly", "every_n_hours"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
MIN_INTERVAL_HOURS = 1


class ScheduleError(ValueError):
    pass


def normalize(sched: dict, now: datetime) -> dict:
    """Validate a schedule from the model/tool call; returns a clean dict. `now` is tz-aware local time."""
    repeat = (sched.get("repeat") or "once").lower()
    if repeat not in REPEATS:
        raise ScheduleError(f"repeat must be one of {', '.join(REPEATS)}")
    out: dict = {"repeat": repeat}

    if repeat == "every_n_hours":
        hours = int(sched.get("every_hours") or 0)
        if hours < MIN_INTERVAL_HOURS or hours > 168:
            raise ScheduleError(f"every_hours must be between {MIN_INTERVAL_HOURS} and 168")
        out["every_hours"] = hours
        return out

    t = str(sched.get("time") or "").strip()
    m = TIME_RE.match(t)
    if not m:
        raise ScheduleError("time must be HH:MM in 24-hour format")
    out["time"] = f"{int(m.group(1)):02d}:{m.group(2)}"

    if repeat == "once":
        d = str(sched.get("date") or "").strip()
        if d:
            try:
                date.fromisoformat(d)
            except ValueError as e:
                raise ScheduleError("date must be YYYY-MM-DD") from e
            out["date"] = d
        else:  # no date: the next occurrence of that time
            candidate = datetime.combine(now.date(), _t(out["time"]), tzinfo=now.tzinfo)
            out["date"] = (now.date() if candidate > now else now.date() + timedelta(days=1)).isoformat()
    if repeat == "weekly":
        days = [str(d).lower()[:3] for d in (sched.get("days") or [])]
        bad = [d for d in days if d not in DAYS]
        if not days or bad:
            raise ScheduleError(f"weekly needs days from {', '.join(DAYS)}")
        out["days"] = sorted(set(days), key=DAYS.index)
    return out


def _t(hhmm: str) -> time:
    hh, mm = hhmm.split(":")
    return time(int(hh), int(mm))


def next_run(sched: dict, after: datetime, last_run: datetime | None = None) -> datetime | None:
    """First run strictly after `after` (tz-aware local). None when a one-off has already happened."""
    tz = after.tzinfo
    repeat = sched["repeat"]
    if repeat == "every_n_hours":
        step = timedelta(hours=sched["every_hours"])
        base = last_run.astimezone(tz) if last_run else after
        nxt = base + step if last_run else after + timedelta(minutes=1)
        while nxt <= after:
            nxt += step
        return nxt.replace(second=0, microsecond=0)

    at = _t(sched["time"])
    if repeat == "once":
        when = datetime.combine(date.fromisoformat(sched["date"]), at, tzinfo=tz)
        return when if when > after and last_run is None else None

    for offset in range(0, 8):
        d = after.date() + timedelta(days=offset)
        candidate = datetime.combine(d, at, tzinfo=tz)
        if candidate <= after:
            continue
        wd = d.weekday()
        if repeat == "daily":
            return candidate
        if repeat == "weekdays" and wd < 5:
            return candidate
        if repeat == "weekly" and DAYS[wd] in sched["days"]:
            return candidate
    return None


def describe(sched: dict) -> str:
    r = sched["repeat"]
    if r == "every_n_hours":
        return f"every {sched['every_hours']}h"
    if r == "once":
        return f"once on {date.fromisoformat(sched['date']):%a %d %b} at {sched['time']}"
    if r == "daily":
        return f"daily at {sched['time']}"
    if r == "weekdays":
        return f"weekdays at {sched['time']}"
    return f"every {', '.join(d.capitalize() for d in sched['days'])} at {sched['time']}"
