"""Briefing preferences the user can change by chat (time, length, section mix, topics). Stored in Postgres.

The scheduler and the pipeline read these on every run, so changes take effect without a restart.
"""

import re
from copy import deepcopy
from datetime import date, datetime

from app.config import get_settings
from app.db import session_scope
from app.models import BriefingPrefs

# Section key -> (spoken title, what belongs there). Order here is the default briefing order.
SECTIONS: dict[str, tuple[str, str]] = {
    "ai": ("AI and new models", "AI, machine learning, new models, AI products, AI research and AI companies"),
    "india": ("India", "Indian politics, policy, government, economy and major national events"),
    "world": ("World", "International news, geopolitics, conflicts, elections and diplomacy outside India"),
    "business": ("Business and markets", "Companies, startups, funding, markets, economy and finance"),
    "science": ("Science and tech", "Science, space, health research, climate, and non-AI technology"),
    "fun": ("Something light", "Quirky, delightful, wonder, space pictures, history ('on this day') items"),
}
DEFAULT_SECTIONS = {"ai": 3, "india": 2, "world": 2, "business": 1, "science": 1, "fun": 1}
MAX_STORIES = 16
TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def defaults() -> dict:
    s = get_settings()
    return {
        "weekday_time": s.briefing_time,
        "weekend_time": s.briefing_time,
        "minutes": s.briefing_minutes,
        "sections": dict(DEFAULT_SECTIONS),
        "exclude_topics": [],
        "focus": "",
        "paused": False,
        "interests": s.user_interests,
    }


async def get_prefs() -> dict:
    async with session_scope() as s:
        row = await s.get(BriefingPrefs, "default")
        stored = dict(row.value) if row else {}
    prefs = defaults()
    prefs.update({k: v for k, v in stored.items() if k in prefs})
    return prefs


def validate(changes: dict) -> tuple[dict, list[str]]:
    """Return (clean_changes, errors). Unknown / invalid fields are reported, not silently applied."""
    clean: dict = {}
    errors: list[str] = []
    for key in ("weekday_time", "weekend_time"):
        if changes.get(key):
            v = str(changes[key]).strip()
            m = TIME_RE.match(v)
            if m:
                clean[key] = f"{int(m.group(1)):02d}:{m.group(2)}"
            else:
                errors.append(f"{key} must be HH:MM in 24-hour time, got {v!r}")
    if changes.get("minutes") is not None:
        m = int(changes["minutes"])
        if 2 <= m <= 20:
            clean["minutes"] = m
        else:
            errors.append("minutes must be between 2 and 20")
    if changes.get("sections"):
        secs = {}
        for k, v in dict(changes["sections"]).items():
            if k not in SECTIONS:
                errors.append(f"unknown section {k!r}; valid: {', '.join(SECTIONS)}")
                continue
            secs[k] = max(0, min(6, int(v)))
        clean["sections"] = secs
    if changes.get("exclude_topics") is not None:
        clean["exclude_topics"] = sorted({str(t).strip().lower() for t in changes["exclude_topics"] if str(t).strip()})
    if changes.get("focus") is not None:
        clean["focus"] = str(changes["focus"]).strip()[:500]
    if changes.get("paused") is not None:
        clean["paused"] = bool(changes["paused"])
    if changes.get("interests"):
        clean["interests"] = str(changes["interests"]).strip()[:1000]
    return clean, errors


async def update_prefs(changes: dict) -> tuple[dict, list[str]]:
    clean, errors = validate(changes)
    prefs = await get_prefs()
    if "sections" in clean:  # merge counts into the existing mix, keeping section order
        merged = dict(prefs["sections"])
        merged.update(clean.pop("sections"))
        if 0 < sum(merged.values()) <= MAX_STORIES:
            prefs["sections"] = {k: merged[k] for k in SECTIONS if merged.get(k, 0) > 0}
        else:
            errors.append(f"total stories must be between 1 and {MAX_STORIES}")
    prefs.update(clean)
    async with session_scope() as s:
        row = await s.get(BriefingPrefs, "default")
        if row:
            row.value = deepcopy(prefs)
        else:
            s.add(BriefingPrefs(key="default", value=deepcopy(prefs)))
    return prefs, errors


def time_for(prefs: dict, day: date) -> str:
    return prefs["weekend_time"] if day.weekday() >= 5 else prefs["weekday_time"]


def describe(prefs: dict) -> str:
    mix = ", ".join(f"{SECTIONS[k][0]} {n}" for k, n in prefs["sections"].items() if n)
    lines = [
        f"Weekdays at {prefs['weekday_time']}, weekends at {prefs['weekend_time']} ({get_settings().timezone})"
        + (" — PAUSED" if prefs["paused"] else ""),
        f"Length: about {prefs['minutes']} minutes, {sum(prefs['sections'].values())} stories",
        f"Mix: {mix}",
    ]
    if prefs["exclude_topics"]:
        lines.append("Skipping topics: " + ", ".join(prefs["exclude_topics"]))
    if prefs["focus"]:
        lines.append(f"Extra focus: {prefs['focus']}")
    lines.append(f"About you (used for ranking): {prefs['interests']}")
    return "\n".join(lines)
