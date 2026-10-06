"""Ranking eval (PRD eval 1): label candidates daily, then compare Jev vs the LLM ranker.

1) `python -m app.cli label` each day for ~2 weeks: y/n "would have wanted this".
2) `python -m app.cli eval-ranking` re-scores the labelled items with both rankers and reports
   precision@N, missed wanted stories, latency and cost.
"""

import json
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.briefing import rank
from app.config import get_settings
from app.db import session_scope
from app.models import Item, ItemScore, Source, UsageEvent

LABELS = get_settings().data_dir / "labels.jsonl"


def _load_labels() -> dict[int, bool]:
    if not LABELS.exists():
        return {}
    out = {}
    for line in LABELS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["item_id"]] = row["want"]
    return out


async def label_today() -> None:
    done = _load_labels()
    since = datetime.now(timezone.utc) - timedelta(hours=30)
    async with session_scope() as s:
        rows = (await s.execute(
            select(Item, Source.name).join(Source, Source.id == Item.source_id, isouter=True)
            .where(Item.fetched_at >= since, Item.duplicate_of.is_(None))
            .order_by(Item.section, Item.id)
        )).all()
    todo = [(it, src) for it, src in rows if it.id not in done]
    print(f"{len(todo)} unlabelled candidates. y = would want it, n = no, s = skip, q = quit")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("a") as f:
        for it, src in todo:
            ans = input(f"\n[{it.section}] {it.title}\n  ({src}) {it.url}\n> ").strip().lower()
            if ans == "q":
                break
            if ans in {"y", "n"}:
                f.write(json.dumps({"item_id": it.id, "want": ans == "y", "date": str(datetime.now().date())}) + "\n")


async def _cost_since(provider: str, start: datetime) -> float:
    async with session_scope() as s:
        v = await s.scalar(select(func.coalesce(func.sum(UsageEvent.cost_usd), 0.0))
                           .where(UsageEvent.provider == provider, UsageEvent.created_at >= start))
    return float(v or 0)


def _report(name: str, scored: list[rank.Scored], labels: dict[int, bool], n: int, seconds: float, cost: float) -> None:
    ordered = sorted(scored, key=lambda x: rank.composite(x.relevance, x.substance, 0.0), reverse=True)
    top = [x.item_id for x in ordered if x.section != "skip"][:n]
    wanted = {i for i, w in labels.items() if w}
    hits = sum(1 for i in top if labels.get(i))
    missed = wanted - set(top)
    per_1k = cost / max(len(scored), 1) * 1000
    print(f"\n== {name} ==")
    print(f"precision@{n}: {hits / max(n, 1):.2f}   wanted stories missed: {len(missed)}/{len(wanted)}")
    print(f"latency: {seconds:.1f}s for {len(scored)} items   cost: ${cost:.4f} (${per_1k:.2f} per 1,000 items)")


async def compare(days: int = 14) -> None:
    labels = _load_labels()
    if not labels:
        print("No labels yet. Run `python -m app.cli label` first.")
        return
    async with session_scope() as s:
        rows = (await s.execute(
            select(Item, Source.name).join(Source, Source.id == Item.source_id, isouter=True)
            .where(Item.id.in_(list(labels)),
                   Item.fetched_at >= datetime.now(timezone.utc) - timedelta(days=days))
        )).all()
    cands = [(it, src or "") for it, src in rows]
    n = sum(1 for w in labels.values() if w) or 8
    n = min(n, len(cands))
    print(f"{len(cands)} labelled items, {sum(labels.get(it.id, False) for it, _ in cands)} wanted; N={n}")

    start, t0 = datetime.now(timezone.utc), time.monotonic()
    llm_scored = await rank.rank_llm(cands)
    _report("LLM ranker", llm_scored, labels, n, time.monotonic() - t0, await _cost_since("gemini", start))

    if get_settings().typesafe_api_key:
        start, t0 = datetime.now(timezone.utc), time.monotonic()
        jev_scored, low_conf = await rank.rank_jev(cands)
        _report("Jev", jev_scored, labels, n, time.monotonic() - t0, await _cost_since("typesafe", start))
        print(f"Jev low-confidence/failed (would fall back to LLM): {len(low_conf)}")
        lat = sorted(x.latency_ms for x in jev_scored if x.latency_ms)
        if lat:
            print(f"Jev per-call latency p50 {lat[len(lat) // 2]} ms, p95 {lat[int(len(lat) * 0.95) - 1]} ms "
                  f"(PRD gate: p95 < 500 ms)")
    else:
        print("\n(TYPESAFE_API_KEY not set: skipping Jev)")


async def side_by_side() -> None:
    """Rank today's candidate pool with both rankers and print each one's briefing picks side by side."""
    cands = await rank.candidates(skip_recent=False)
    by_id = {it.id: (it, src) for it, src in cands}
    print(f"{len(cands)} candidates\n")

    start, t0 = datetime.now(timezone.utc), time.monotonic()
    llm_scored = await rank.rank_llm(cands)
    llm_secs, llm_cost = time.monotonic() - t0, await _cost_since("gemini", start)

    start, t0 = datetime.now(timezone.utc), time.monotonic()
    jev_scored, fallback = await rank.rank_jev(cands)
    jev_secs, jev_cost = time.monotonic() - t0, await _cost_since("typesafe", start)
    # what the real pipeline does: low-confidence/failed Jev items go to the LLM ranker
    fb_ids = {it.id for it, _ in fallback}
    combined = jev_scored + [x for x in llm_scored if x.item_id in fb_ids]

    src_of = {it.id: it.source_id for it, _ in cands}

    def picks(scored: list[rank.Scored]) -> dict[str, list[int]]:
        rows = [ItemScore(item_id=x.item_id, run_id="cmp", ranker=x.ranker, relevance=x.relevance,
                          substance=x.substance, section=x.section, composite=rank.composite(x.relevance, x.substance, 0),
                          injection=x.injection) for x in scored]
        return rank.select_top(rows, src_of)

    a, b = picks(llm_scored), picks(combined)
    for sec in ["ai", "world", "fun"]:
        print(f"== {sec.upper()} ==")
        both = set(a[sec]) & set(b[sec])
        for label, ids in (("LLM", a[sec]), ("Jev", b[sec])):
            print(f"  {label}:")
            for i in ids:
                it, src = by_id[i]
                mark = "=" if i in both else "*"
                print(f"    {mark} [{src}] {it.title[:95]}")
        print()

    overlap = sum(len(set(a[s]) & set(b[s])) for s in a)
    total = sum(len(a[s]) for s in a)
    lat = sorted(x.latency_ms for x in jev_scored if x.latency_ms)
    print(f"overlap: {overlap}/{total} picks identical  (= in both, * only in that ranker)")
    print(f"LLM ranker: {llm_secs:.1f}s total, ${llm_cost:.4f}")
    print(f"Jev: {jev_secs:.1f}s total, {len(jev_scored)} confident, {len(fallback)} sent to LLM fallback, "
          f"tokens logged under 'typesafe' (cost not priced)")
    if lat:
        print(f"Jev per-call latency: p50 {lat[len(lat) // 2]} ms, p95 {lat[max(int(len(lat) * 0.95) - 1, 0)]} ms")
