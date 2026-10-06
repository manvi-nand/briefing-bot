"""Telegram Bot API channel: fallback if WhatsApp rules change, and the easiest channel for local dev."""

import asyncio
import html
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from datetime import datetime

import httpx
from sqlalchemy import select

from app.channels.base import STORY_HINT, Button, Channel, Incoming, Source, send_category, source_label
from app.config import get_settings
from app.db import session_scope
from app.models import ChatMessageRef, MessageSources

log = logging.getLogger(__name__)


def to_html(text: str) -> str:
    """Escape for Telegram HTML; keep **bold** from the model as real bold."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def links_html(sources: list[Source], prefix: str = "Sources: ") -> str:
    seen, parts = set(), []
    for title, url in sources:
        label = source_label(title, url)
        if url in seen:
            continue
        seen.add(url)
        parts.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>')
    return prefix + " · ".join(parts) if parts else ""


def sources_list_html(sources: list[Source]) -> str:
    seen, lines = set(), []
    for title, url in sources:
        if url in seen:
            continue
        seen.add(url)
        lines.append(f'• <a href="{html.escape(url, quote=True)}">{html.escape(source_label(title, url))}</a>')
    return "\n".join(lines)


def _sources_keyboard(n: int, expanded: bool) -> dict:
    label = "🔼 Hide sources" if expanded else f"🔗 Sources ({n})"
    return {"inline_keyboard": [[{"text": label, "callback_data": "src:toggle"}]]}


def parse_update(update: dict) -> Incoming | None:
    if cq := update.get("callback_query"):
        chat = (cq.get("message") or {}).get("chat", {})
        return Incoming(
            channel="telegram",
            sender=str(chat.get("id") or cq.get("from", {}).get("id")),
            external_id=f"cb:{cq['id']}",
            kind="button",
            button_id=cq.get("data"),
            raw=update,
            origin_message_id=(cq.get("message") or {}).get("message_id"),
        )
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    base = dict(
        channel="telegram",
        sender=str(msg["chat"]["id"]),
        external_id=f"{msg['chat']['id']}:{msg['message_id']}",
        raw=update,
    )
    if "voice" in msg:
        return Incoming(kind="voice", media_ref=msg["voice"]["file_id"], **base)
    if "audio" in msg:
        return Incoming(kind="voice", media_ref=msg["audio"]["file_id"], **base)
    if "text" in msg:
        return Incoming(kind="text", text=msg["text"], **base)
    return Incoming(kind="text", text="[unsupported message type]", **base)


class TelegramChannel(Channel):
    name = "telegram"

    def __init__(self) -> None:
        s = get_settings()
        self.s = s
        self.api = f"https://api.telegram.org/bot{s.telegram_bot_token}"
        self.file_api = f"https://api.telegram.org/file/bot{s.telegram_bot_token}"
        self.http = httpx.AsyncClient(timeout=70)
        self._offset = 0

    @property
    def owner(self) -> str:
        return self.s.telegram_chat_id

    async def _call(self, method: str, **kwargs) -> dict:
        r = await self.http.post(f"{self.api}/{method}", **kwargs)
        data = r.json()
        if not data.get("ok"):
            log.error("Telegram %s failed: %s", method, data)
            raise RuntimeError(f"Telegram {method} failed: {data.get('description')}")
        return data["result"]

    async def send_text(self, to: str, text: str, sources: list[Source] | None = None) -> str | None:
        """Plain HTML text. With sources, the last chunk gets a collapsed '🔗 Sources (N)' button."""
        last = None
        chunks = [text[i : i + 3500] for i in range(0, len(text), 3500)] or [""]
        for n, chunk in enumerate(chunks):
            body = to_html(chunk) or "🔗 Sources for that answer"
            payload = {"chat_id": to, "text": body, "parse_mode": "HTML", "link_preview_options": {"is_disabled": True}}
            with_sources = bool(sources) and n == len(chunks) - 1
            if with_sources:
                payload["reply_markup"] = _sources_keyboard(len(sources), expanded=False)
            res = await self._call("sendMessage", json=payload)
            last = str(res["message_id"])
            await self._remember(to, res["message_id"])
            if with_sources:
                async with session_scope() as s:
                    s.add(MessageSources(chat_id=str(to), message_id=res["message_id"], body_html=body,
                                         sources=[list(x) for x in sources]))
        return last

    async def toggle_sources(self, message_id: int) -> None:
        async with session_scope() as s:
            row = await s.scalar(select(MessageSources).where(MessageSources.message_id == message_id)
                                 .order_by(MessageSources.id.desc()).limit(1))
            if not row:
                return
            row.expanded = not row.expanded
            text = row.body_html
            if row.expanded:
                text += "\n\n" + sources_list_html([tuple(x) for x in row.sources])
            chat_id, n, expanded = row.chat_id, len(row.sources), row.expanded
        await self._call("editMessageText", json={
            "chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
            "reply_markup": _sources_keyboard(n, expanded=expanded),
        })

    async def send_story_list(self, to: str, header: str, groups: list[tuple[str, list[tuple[int, str, str, str]]]],
                              episode_id: int) -> str | None:
        """One tidy message: headlines grouped by section, each a link. No buttons; the user just replies."""
        lines = [f"<b>{html.escape(header)}</b>"]
        for sec, stories in groups:
            lines.append(f"\n<b>{html.escape(sec)}</b>")
            for n, title, source, url in stories:
                lines.append(f'{n}. <a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
                             f" · <i>{html.escape(source)}</i>")
        lines.append(f"\n💬 <i>{html.escape(STORY_HINT)}</i>")
        res = await self._call("sendMessage", json={
            "chat_id": to, "text": "\n".join(lines), "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        })
        await self._remember(to, res["message_id"])
        return str(res["message_id"])

    async def delete_message(self, message_id: int) -> None:
        try:
            await self._call("deleteMessage", json={"chat_id": self.owner, "message_id": message_id})
        except RuntimeError:
            pass

    async def send_voice(self, to: str, ogg_path: Path, caption: str | None = None) -> str | None:
        with ogg_path.open("rb") as f:
            res = await self._call(
                "sendVoice",
                data={"chat_id": to, **({"caption": caption[:1000]} if caption else {})},
                files={"voice": (ogg_path.name, f, "audio/ogg")},
            )
        await self._remember(to, res["message_id"])
        return str(res["message_id"])

    async def send_buttons(
        self, to: str, text: str, buttons: list[Button], sources: list[Source] | None = None
    ) -> str | None:
        keyboard = [[{"text": b.title, "callback_data": b.id[:64]} for b in buttons]]
        body = to_html(text) + (f"\n{links_html(sources, prefix='Read: ')}" if sources else "")
        res = await self._call(
            "sendMessage",
            json={
                "chat_id": to,
                "text": body,
                "parse_mode": "HTML",
                "reply_markup": {"inline_keyboard": keyboard},
                "link_preview_options": {"is_disabled": True},
            },
        )
        await self._remember(to, res["message_id"])
        return str(res["message_id"])

    async def _remember(self, chat_id: str, message_id: int, category: str | None = None) -> None:
        try:
            async with session_scope() as s:
                s.add(ChatMessageRef(channel="telegram", chat_id=str(chat_id), message_id=int(message_id),
                                     category=category or send_category.get()))
        except Exception:
            log.exception("could not record telegram message id")

    async def record_incoming(self, inc: Incoming) -> None:
        if inc.external_id.startswith("cb:"):
            return  # button taps aren't messages
        chat_id, _, message_id = inc.external_id.partition(":")
        if message_id.isdigit():
            await self._remember(chat_id, int(message_id), "chat")

    async def clear(self, category: str | None = None, before: datetime | None = None) -> int:
        """Delete remembered messages. Telegram only allows deleting messages < 48h old; older ones are
        skipped by Telegram (clear those with the app's 'Clear history')."""
        async with session_scope() as s:
            q = select(ChatMessageRef).where(ChatMessageRef.channel == "telegram", ChatMessageRef.deleted.is_(False))
            if category:
                q = q.where(ChatMessageRef.category == category)
            if before:
                q = q.where(ChatMessageRef.created_at < before)
            refs = list((await s.scalars(q)).all())
            by_chat: dict[str, list[int]] = {}
            for r in refs:
                by_chat.setdefault(r.chat_id, []).append(r.message_id)
            for chat_id, ids in by_chat.items():
                for i in range(0, len(ids), 100):  # deleteMessages takes up to 100 ids
                    batch = ids[i : i + 100]
                    try:
                        await self._call("deleteMessages", json={"chat_id": chat_id, "message_ids": batch})
                    except Exception:  # e.g. some too old: fall back to one by one, ignoring failures
                        for mid in batch:
                            try:
                                await self._call("deleteMessage", json={"chat_id": chat_id, "message_id": mid})
                            except Exception:
                                pass
            for r in refs:
                r.deleted = True
        return len(refs)

    async def send_briefing_notice(self, to: str, label: str, episode_id: int) -> str | None:
        return await self.send_buttons(
            to, f"Your {label} briefing is ready 🎧", [Button(id=f"play:{episode_id}", title="▶️ Play")]
        )

    async def send_proactive(self, to: str, text: str, sources: list[Source] | None = None) -> str | None:
        return await self.send_text(to, text, sources)

    async def download_media(self, media_ref: str) -> tuple[bytes, str]:
        info = await self._call("getFile", json={"file_id": media_ref})
        r = await self.http.get(f"{self.file_api}/{info['file_path']}")
        r.raise_for_status()
        return r.content, "audio/ogg"

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        """Stops the button's loading spinner; `text` shows as a small pop-up instead of a new message."""
        try:
            await self._call("answerCallbackQuery",
                             json={"callback_query_id": callback_id, **({"text": text[:200]} if text else {})})
        except Exception:
            pass

    async def set_webhook(self, url: str, secret: str) -> None:
        await self._call("setWebhook", json={"url": url, "secret_token": secret, "drop_pending_updates": True})

    async def poll_forever(self, handler: Callable[[Incoming], Awaitable[None]]) -> None:
        """Long-poll getUpdates (local dev: no public URL needed)."""
        await self._call("deleteWebhook", json={"drop_pending_updates": False})
        log.info("Telegram long polling started")
        while True:
            try:
                updates = await self._call(
                    "getUpdates", json={"offset": self._offset, "timeout": 50, "allowed_updates": ["message", "callback_query"]}
                )
                for u in updates:
                    self._offset = u["update_id"] + 1
                    inc = parse_update(u)
                    if inc:
                        asyncio.create_task(handler(inc))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Telegram polling error; retrying in 5s")
                await asyncio.sleep(5)

    async def close(self) -> None:
        await self.http.aclose()
