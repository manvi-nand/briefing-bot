"""Usage logging and the per-day budget cap (PRD risk: 'Runaway cost')."""

import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select

from app.config import get_settings
from app.db import session_scope
from app.models import UsageEvent

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output incl. thinking, cached input). Gemini 3.8 Flash promo pricing runs to
# 2026-12-31 and doubles on 2027-01-01 -- update then.
LLM_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
}
SEARCH_PER_QUERY = 14.0 / 1000  # Google Search grounding after the free allowance
FREE_SEARCHES_PER_MONTH = 5000
INR_PER_USD = 85.0
SARVAM_TTS_PER_CHAR = (30.0 / 10_000) / INR_PER_USD
SARVAM_STT_PER_SEC = (30.0 / 3600) / INR_PER_USD


class BudgetExceeded(RuntimeError):
    pass


def llm_cost(model: str, usage, search_queries: int = 0) -> float:
    pin, pout, pcache = LLM_PRICES.get(model, LLM_PRICES["gemini-3.8-flash"])
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    cached = getattr(usage, "cached_content_token_count", 0) or 0
    out = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
    tool_prompt = getattr(usage, "tool_use_prompt_token_count", 0) or 0
    cost = ((prompt - cached + tool_prompt) * pin + cached * pcache + out * pout) / 1_000_000
    return cost + search_queries * SEARCH_PER_QUERY


async def search_cost(queries: int) -> float:
    """Price grounding queries, counting this month's free allowance (from our own usage log)."""
    if not queries:
        return 0.0
    now = datetime.now(get_settings().tz)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    async with session_scope() as s:
        used = await s.scalar(
            select(func.coalesce(func.sum(UsageEvent.units["search_queries"].as_integer()), 0))
            .where(UsageEvent.provider == "gemini", UsageEvent.created_at >= month_start)
        )
    billable = max(0, (used or 0) + queries - FREE_SEARCHES_PER_MONTH)
    return min(billable, queries) * SEARCH_PER_QUERY


async def record(provider: str, stage: str, units: dict, cost_usd: float, latency_ms: int | None = None) -> None:
    try:
        async with session_scope() as s:
            s.add(UsageEvent(provider=provider, stage=stage, units=units, cost_usd=cost_usd, latency_ms=latency_ms))
    except Exception:  # never let accounting break the main flow
        log.exception("failed to record usage")


async def spent_today() -> float:
    settings = get_settings()
    now = datetime.now(settings.tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    async with session_scope() as s:
        total = await s.scalar(
            select(func.coalesce(func.sum(UsageEvent.cost_usd), 0.0)).where(UsageEvent.created_at >= start)
        )
    return float(total or 0.0)


async def check_budget(stage: str) -> None:
    cap = get_settings().daily_budget_usd
    if cap <= 0:
        return
    spent = await spent_today()
    if spent >= cap:
        raise BudgetExceeded(f"Daily budget ${cap:.2f} reached (spent ${spent:.2f}); blocked stage '{stage}'.")


async def cost_report(days: int = 7) -> list[tuple[str, str, float]]:
    since = datetime.now(get_settings().tz) - timedelta(days=days)
    async with session_scope() as s:
        rows = await s.execute(
            select(UsageEvent.provider, UsageEvent.stage, func.sum(UsageEvent.cost_usd))
            .where(UsageEvent.created_at >= since)
            .group_by(UsageEvent.provider, UsageEvent.stage)
            .order_by(func.sum(UsageEvent.cost_usd).desc())
        )
        return [(p, st, float(c or 0)) for p, st, c in rows.all()]
