"""Ingest stories from RSS, Hacker News and Wikipedia 'On this day'; extract full article text."""

import asyncio
import logging
import multiprocessing
import re
from calendar import timegm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta, timezone
from html import unescape

import feedparser
import httpx
import trafilatura
import yaml
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.db import session_scope
from app.models import Item, Source

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (personal-briefing-bot; +https://example.invalid)"}
MAX_AGE = timedelta(hours=36)
FULLTEXT_CHARS = 12_000


def _clean(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


async def sync_sources() -> list[dict]:
    """Upsert config/sources.yaml into the sources table; returns the config list."""
    cfg = yaml.safe_load(get_settings().sources_file.read_text())["sources"]
    async with session_scope() as s:
        for src in cfg:
            stmt = insert(Source).values(
                name=src["name"], url=src["url"], type=src["type"], section=src["section"],
                active=src.get("active", True),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[Source.url],
                set_={"name": stmt.excluded.name, "type": stmt.excluded.type,
                      "section": stmt.excluded.section, "active": stmt.excluded.active},
            )
            await s.execute(stmt)
        # sources removed from the YAML stop being fetched
        await s.execute(
            update(Source).where(Source.url.not_in([src["url"] for src in cfg])).values(active=False)
        )
    return cfg


async def _fetch_rss(client: httpx.AsyncClient, src: dict) -> list[dict]:
    r = await client.get(src["url"])
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    now = datetime.now(timezone.utc)
    out = []
    for e in feed.entries[: src.get("limit", 10) * 2]:
        link = e.get("link")
        if not link:
            continue
        ts = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime.fromtimestamp(timegm(ts), tz=timezone.utc) if ts else None
        if published and now - published > MAX_AGE:
            continue
        summary = _clean(e.get("summary", "") or "")
        content = ""
        if e.get("content"):
            content = _clean(e.content[0].get("value", ""))
        out.append(
            {"title": _clean(e.get("title", "")), "url": link, "published_at": published,
             "summary": summary[:1500], "full_text": content[:FULLTEXT_CHARS]}
        )
        if len(out) >= src.get("limit", 10):
            break
    return out


async def _fetch_hn(client: httpx.AsyncClient, src: dict) -> list[dict]:
    r = await client.get(src["url"])
    r.raise_for_status()
    out = []
    for hit in r.json().get("hits", [])[: src.get("limit", 30)]:
        url = hit.get("url")
        if not url:  # Ask HN etc.
            continue
        out.append(
            {"title": hit.get("title", ""), "url": url,
             "published_at": datetime.fromtimestamp(hit.get("created_at_i", 0), tz=timezone.utc),
             "summary": f"Hacker News: {hit.get('points', 0)} points, {hit.get('num_comments', 0)} comments.",
             "full_text": ""}
        )
    return out


async def _fetch_onthisday(client: httpx.AsyncClient, src: dict) -> list[dict]:
    today = datetime.now(get_settings().tz)
    r = await client.get(f"{src['url']}/{today:%m}/{today:%d}")
    r.raise_for_status()
    out = []
    for ev in r.json().get("selected", [])[: src.get("limit", 3)]:
        page = (ev.get("pages") or [{}])[0]
        page_url = page.get("content_urls", {}).get("desktop", {}).get("page", "https://en.wikipedia.org")
        text = f"On this day in {ev.get('year')}: {ev.get('text', '')}"
        out.append(
            {"title": text[:300], "url": f"{page_url}?otd={today:%Y-%m-%d}", "published_at": None,
             "summary": text, "full_text": f"{text}\n\n{page.get('extract', '')}"}
        )
    return out


FETCHERS = {"rss": _fetch_rss, "hn": _fetch_hn, "onthisday": _fetch_onthisday}

# trafilatura/lxml has occasionally hard-crashed (SIGABRT) on specific pages, which would take the whole
# app down if run in a thread. Running it in a subprocess means a crash there only breaks that one
# extraction; the pool recreates itself and ingest carries on with the items it already has.
_EXTRACT_TIMEOUT = 15
_pool: ProcessPoolExecutor | None = None
_pool_lock = asyncio.Lock()


def _extract_worker(html: str) -> str:
    return trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True) or ""


async def _get_pool() -> ProcessPoolExecutor:
    global _pool
    async with _pool_lock:
        if _pool is None:
            # "spawn" explicitly (not "fork"): our app is multi-threaded (asyncio + library threads), and
            # fork()ing a multi-threaded process can deadlock the child if another thread held a lock (e.g.
            # malloc's) at fork time. spawn starts a clean interpreter instead -- slightly slower to boot,
            # but immune to that class of bug, and behaves the same in dev (macOS) and prod (Linux).
            ctx = multiprocessing.get_context("spawn")
            _pool = ProcessPoolExecutor(max_workers=3, mp_context=ctx)
        return _pool


async def _reset_pool() -> None:
    global _pool
    async with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


async def _extract_fulltext(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, follow_redirects=True)
        if r.status_code >= 400 or "html" not in r.headers.get("content-type", ""):
            return ""
    except Exception:
        return ""
    try:
        pool = await _get_pool()
        loop = asyncio.get_running_loop()
        text = await asyncio.wait_for(loop.run_in_executor(pool, _extract_worker, r.text), timeout=_EXTRACT_TIMEOUT)
        return (text or "")[:FULLTEXT_CHARS]
    except BrokenProcessPool:
        log.warning("extraction worker crashed on %s; recreating the worker pool", url)
        await _reset_pool()
        return ""
    except TimeoutError:
        log.warning("extraction timed out on %s", url)
        return ""
    except Exception:
        return ""


async def ingest() -> list[int]:
    """Fetch all active sources; insert new items; return ids of newly inserted items."""
    cfg = await sync_sources()
    async with session_scope() as s:
        sources = {src.url: src for src in (await s.scalars(select(Source).where(Source.active))).all()}

    async with httpx.AsyncClient(timeout=25, headers=UA, follow_redirects=True) as client:

        async def one(src_cfg: dict) -> list[tuple[dict, dict]]:
            if src_cfg["url"] not in sources:
                return []
            try:
                items = await FETCHERS[src_cfg["type"]](client, src_cfg)
                return [(src_cfg, it) for it in items]
            except Exception as e:
                log.warning("source %s failed: %s", src_cfg["name"], e)
                return []

        batches = await asyncio.gather(*(one(c) for c in cfg))
        found = [pair for batch in batches for pair in batch]

        # Skip URLs we already have.
        urls = [it["url"] for _, it in found]
        async with session_scope() as s:
            existing = set((await s.scalars(select(Item.url).where(Item.url.in_(urls)))).all()) if urls else set()
        fresh = [(c, it) for c, it in found if it["url"] not in existing]
        fresh = list({it["url"]: (c, it) for c, it in fresh}.values())

        sem = asyncio.Semaphore(8)

        async def fill(it: dict) -> None:
            if len(it["full_text"]) >= 800:
                return
            async with sem:
                text = await _extract_fulltext(client, it["url"])
            if len(text) > len(it["full_text"]):
                it["full_text"] = text

        await asyncio.gather(*(fill(it) for _, it in fresh))

    new_ids: list[int] = []
    async with session_scope() as s:
        for src_cfg, it in fresh:
            src = sources[src_cfg["url"]]
            stmt = (
                insert(Item)
                .values(source_id=src.id, section=src.section, title=it["title"][:1000], url=it["url"],
                        published_at=it["published_at"], summary=it["summary"],
                        full_text=it["full_text"] or it["summary"])
                .on_conflict_do_nothing(index_elements=[Item.url])
                .returning(Item.id)
            )
            new_id = await s.scalar(stmt)
            if new_id:
                new_ids.append(new_id)
    log.info("ingest: %d candidates, %d new", len(found), len(new_ids))
    return new_ids
