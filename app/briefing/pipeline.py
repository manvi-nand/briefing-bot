"""Daily briefing: ingest -> dedupe -> rank -> script -> TTS -> deliver."""

import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select

from app import costs
from app.briefing import dedupe, ingest, rank, script
from app.channels import Button, primary_channel
from app import heartbeat
from app.channels.base import send_category
from app.config import get_settings
from app.db import session_scope
from app.models import Episode, Item, ItemScore, Message, Source
from app.prefs import get_prefs
from app.speech import audio, sarvam

log = logging.getLogger(__name__)

async def latest_episode() -> Episode | None:
    async with session_scope() as s:
        return await s.scalar(
            select(Episode).where(Episode.status.in_(["ready", "sent"])).order_by(Episode.created_at.desc()).limit(1)
        )


async def episode_stories(ep: Episode) -> list[tuple[int, Item, str]]:
    """[(story_number, item, source_name)] in briefing order."""
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(Item, Source.name)
                .join(Source, Source.id == Item.source_id, isouter=True)
                .where(Item.id.in_(ep.story_order or []))
            )
        ).all()
    by_id = {it.id: (it, name or "") for it, name in rows}
    return [(n, *by_id[i]) for n, i in enumerate(ep.story_order or [], start=1) if i in by_id]


async def build_episode(today: date | None = None, *, ranker: str | None = None, with_audio: bool = True) -> Episode:
    settings = get_settings()
    today = today or datetime.now(settings.tz).date()
    run_id = f"{today:%Y%m%d}-{uuid.uuid4().hex[:6]}"
    t0 = time.monotonic()
    prefs = await get_prefs()

    async with session_scope() as s:
        ep = Episode(date=today, run_id=run_id, language=settings.sarvam_tts_language, status="building")
        s.add(ep)
    try:
        new_ids = await ingest.ingest()
        await dedupe.embed_and_dedupe(new_ids)
        try:
            from app.trackers import check_watches

            await check_watches(new_ids)
        except Exception:
            log.exception("topic watch check failed")
        rows = await rank.score_candidates(run_id, ranker, prefs["interests"])
        async with session_scope() as s:
            meta = (await s.execute(
                select(Item.id, Item.source_id, Item.embedding).where(Item.id.in_([r.item_id for r in rows]))
            )).all()
        src_of = {i: src for i, src, _ in meta}
        vectors = {i: np.array(emb, dtype=np.float32) for i, _, emb in meta if emb is not None}
        picked = rank.select_top(rows, src_of, prefs["sections"], prefs["exclude_topics"], vectors)

        order: list[tuple[str, int]] = [(sec, i) for sec in prefs["sections"] for i in picked.get(sec, [])]
        if not order:
            raise RuntimeError("no stories passed ranking")
        async with session_scope() as s:
            items = {it.id: it for it in (await s.scalars(select(Item).where(Item.id.in_([i for _, i in order])))).all()}
            sources = {src.id: src.name for src in (await s.scalars(select(Source))).all()}
        stories = [
            (n, sec, items[i], sources.get(items[i].source_id, "")) for n, (sec, i) in enumerate(order, start=1)
        ]

        segments = await script.write_script(stories, today, prefs)
        full_script = "\n\n".join(seg["text"] for seg in segments)

        audio_path, seconds = None, None
        if with_audio:
            wav, seconds = await sarvam.synthesize([seg["text"] for seg in segments], stage="tts_briefing")
            out = settings.audio_dir / f"briefing-{today:%Y-%m-%d}-{run_id[-6:]}.ogg"
            await audio.to_ogg_opus(wav, out)
            audio_path = str(out)

        async with session_scope() as s:
            ep = await s.scalar(select(Episode).where(Episode.run_id == run_id))
            ep.segments = segments
            ep.script = full_script
            ep.story_order = [i for _, i in order]
            ep.audio_path = audio_path
            ep.audio_seconds = seconds
            ep.status = "ready"
        log.info("episode %s ready: %d stories, %.0fs audio, built in %.0fs",
                 run_id, len(order), seconds or 0, time.monotonic() - t0)
        await costs.record("pipeline", "build_episode", {"run_id": run_id, "stories": len(order)}, 0.0,
                           int((time.monotonic() - t0) * 1000))
        return ep
    except Exception:
        async with session_scope() as s:
            ep = await s.scalar(select(Episode).where(Episode.run_id == run_id))
            ep.status = "failed"
        raise


async def deliver_notice(ep: Episode) -> None:
    ch = primary_channel()
    label = f"{ep.date:%A}"
    try:  # keep the chat tidy: remove the previous briefing's notice, audio and story cards
        removed = await ch.clear(category="briefing", before=datetime.now(timezone.utc))
        if removed:
            log.info("removed %d old briefing messages", removed)
    except Exception:
        log.exception("briefing cleanup failed")
    token = send_category.set("briefing")
    try:
        msg_id = await ch.send_briefing_notice(ch.owner, label, ep.id)
    finally:
        send_category.reset(token)
    async with session_scope() as s:
        s.add(Message(channel=ch.name, external_id=msg_id, direction="out", type="template",
                      text=f"briefing notice {label}", episode_id=ep.id))
        e = await s.get(Episode, ep.id)
        e.status, e.sent_at = "sent", datetime.now(timezone.utc)
    await heartbeat.ping("briefing")


async def play_episode(episode_id: int | None = None) -> None:
    """Send the audio + numbered story cards with feedback buttons."""
    ch = primary_channel()
    async with session_scope() as s:
        ep = await s.get(Episode, episode_id) if episode_id else None
    ep = ep or await latest_episode()
    if not ep:
        await ch.send_text(ch.owner, "No briefing is ready yet.")
        return

    token = send_category.set("briefing")
    try:
        await _send_episode(ch, ep)
    finally:
        send_category.reset(token)


async def _send_episode(ch, ep: Episode) -> None:
    stories = await episode_stories(ep)
    if ep.audio_path:
        await ch.send_voice(ch.owner, Path(ep.audio_path))
    else:
        await ch.send_text(ch.owner, ep.script or "(empty briefing)")

    async with session_scope() as s:
        e = await s.get(Episode, ep.id)
        e.played_at = e.played_at or datetime.now(timezone.utc)
        if e.status == "ready":
            e.status, e.sent_at = "sent", e.sent_at or datetime.now(timezone.utc)
        s.add(Message(channel=ch.name, direction="out", type="audio", text="briefing audio", episode_id=ep.id))

    await ch.send_story_list(ch.owner, f"🗞 {ep.date:%A} briefing · {len(stories)} stories",
                             await story_groups(ep, stories), ep.id)


async def story_groups(ep: Episode, stories: list[tuple[int, Item, str]]):
    """Group the episode's stories under their section titles, in briefing order."""
    from app.prefs import SECTIONS

    async with session_scope() as s:
        rows = (await s.execute(
            select(ItemScore.item_id, ItemScore.section).where(ItemScore.run_id == ep.run_id)
        )).all()
    section_of = {i: sec for i, sec in rows}
    groups: list[tuple[str, list]] = []
    for n, item, source in stories:
        title = SECTIONS.get(section_of.get(item.id, ""), ("Stories", ""))[0]
        if not groups or groups[-1][0] != title:
            groups.append((title, []))
        groups[-1][1].append((n, item.title, source or "source", item.url))
    return groups


AUDIO_RETENTION_DAYS = 14


async def cleanup_audio() -> int:
    """Delete generated briefing audio older than AUDIO_RETENTION_DAYS. Runs on a daily schedule."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=AUDIO_RETENTION_DAYS)
    removed = 0
    for f in get_settings().audio_dir.glob("*.ogg"):
        try:
            if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            log.exception("could not remove old audio file %s", f)
    if removed:
        log.info("cleanup_audio: removed %d file(s) older than %d days", removed, AUDIO_RETENTION_DAYS)
    return removed


async def run_daily() -> None:
    """Scheduler entrypoint: build then send the notice."""
    try:
        ep = await build_episode()
    except Exception as e:
        log.exception("briefing build failed")
        ch = primary_channel()
        await ch.send_proactive(ch.owner, f"Today's briefing failed to build: {e}")
        return
    await deliver_notice(ep)
