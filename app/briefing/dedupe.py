"""Embed new items and mark near-duplicates (stage 1 of the two-stage filter)."""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select

from app.briefing.embed import embed, item_text
from app.config import get_settings
from app.db import session_scope
from app.models import Item

log = logging.getLogger(__name__)


def find_duplicates(new_vecs: np.ndarray, old_vecs: np.ndarray, threshold: float) -> list[int | None]:
    """For each new vector return the index (into old+preceding new) of its duplicate, or None.

    Indices < len(old_vecs) refer to old items; larger indices refer to earlier new items
    (offset by len(old_vecs)).
    """
    result: list[int | None] = []
    pool = old_vecs if len(old_vecs) else np.zeros((0, new_vecs.shape[1]), dtype=np.float32)
    for i, v in enumerate(new_vecs):
        if len(pool):
            sims = pool @ v
            j = int(np.argmax(sims))
            result.append(j if sims[j] >= threshold else None)
        else:
            result.append(None)
        pool = np.vstack([pool, v[None, :]])
    return result


async def embed_and_dedupe(new_ids: list[int]) -> int:
    if not new_ids:
        return 0
    threshold = get_settings().dedupe_threshold
    since = datetime.now(timezone.utc) - timedelta(days=3)
    async with session_scope() as s:
        new_items = (await s.scalars(select(Item).where(Item.id.in_(new_ids)).order_by(Item.id))).all()
        old_items = (
            await s.scalars(
                select(Item).where(
                    Item.fetched_at >= since, Item.embedding.is_not(None), Item.id.not_in(new_ids)
                )
            )
        ).all()
        vecs = await embed([item_text(it.title, it.summary or it.full_text) for it in new_items])
        old_vecs = (
            np.array([it.embedding for it in old_items], dtype=np.float32)
            if old_items
            else np.zeros((0, vecs.shape[1]), dtype=np.float32)
        )
        dup_idx = find_duplicates(vecs, old_vecs, threshold)
        ids_pool = [it.id for it in old_items] + [it.id for it in new_items]
        dups = 0
        for it, v, d in zip(new_items, vecs, dup_idx, strict=True):
            it.embedding = v.tolist()
            if d is not None:
                target = ids_pool[d]
                # point at the canonical (non-duplicate) item
                it.duplicate_of = target
                dups += 1
    log.info("dedupe: %d/%d new items were near-duplicates", dups, len(new_ids))
    return dups
