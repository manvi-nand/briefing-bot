"""WhatsApp Business Platform (Cloud API, direct from Meta)."""

import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from app.channels.base import Button, Channel, Incoming, Source, plain_sources
from app.config import get_settings
from app.db import session_scope
from app.models import Message

log = logging.getLogger(__name__)

WINDOW = timedelta(hours=24)


def verify_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    """Validate X-Hub-Signature-256 = 'sha256=' + HMAC_SHA256(app_secret, raw_body)."""
    if not app_secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def parse_webhook(payload: dict) -> tuple[list[Incoming], list[dict]]:
    """Extract inbound messages and delivery statuses from a Cloud API webhook payload."""
    messages: list[Incoming] = []
    statuses: list[dict] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            statuses.extend(value.get("statuses", []))
            for m in value.get("messages", []):
                mtype = m.get("type")
                base = dict(channel="whatsapp", sender=m.get("from", ""), external_id=m.get("id", ""), raw=m)
                if mtype == "text":
                    messages.append(Incoming(kind="text", text=m["text"]["body"], **base))
                elif mtype == "audio":
                    messages.append(Incoming(kind="voice", media_ref=m["audio"]["id"], **base))
                elif mtype == "button":  # quick reply on a template
                    b = m.get("button", {})
                    messages.append(
                        Incoming(kind="button", button_id=b.get("payload"), text=b.get("text", ""), **base)
                    )
                elif mtype == "interactive":
                    inter = m.get("interactive", {})
                    reply = inter.get("button_reply") or inter.get("list_reply") or {}
                    messages.append(
                        Incoming(kind="button", button_id=reply.get("id"), text=reply.get("title", ""), **base)
                    )
                elif mtype == "reaction":
                    r = m.get("reaction", {})
                    messages.append(
                        Incoming(
                            kind="button",
                            button_id=f"react:{r.get('message_id')}:{r.get('emoji', '')}",
                            text=r.get("emoji", ""),
                            **base,
                        )
                    )
                else:
                    messages.append(Incoming(kind="text", text=f"[unsupported message type: {mtype}]", **base))
    return messages, statuses


class WhatsAppChannel(Channel):
    name = "whatsapp"

    def __init__(self) -> None:
        s = get_settings()
        self.s = s
        self.base = f"https://graph.facebook.com/{s.whatsapp_api_version}"
        self.http = httpx.AsyncClient(
            timeout=60, headers={"Authorization": f"Bearer {s.whatsapp_access_token}"}
        )

    @property
    def owner(self) -> str:
        return self.s.my_whatsapp_number

    async def _send(self, to: str, body: dict) -> str | None:
        payload = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to, **body}
        r = await self.http.post(f"{self.base}/{self.s.whatsapp_phone_number_id}/messages", json=payload)
        if r.status_code >= 400:
            log.error("WhatsApp send failed %s: %s", r.status_code, r.text[:1000])
            r.raise_for_status()
        msgs = r.json().get("messages") or [{}]
        return msgs[0].get("id")

    async def send_text(self, to: str, text: str, sources: list[Source] | None = None) -> str | None:
        last = None
        for part in _split(plain_sources(text, sources), 4000):  # Cloud API text body limit is 4096
            last = await self._send(to, {"type": "text", "text": {"body": part, "preview_url": False}})
        return last

    async def upload_media(self, path: Path, mime: str) -> str:
        with path.open("rb") as f:
            r = await self.http.post(
                f"{self.base}/{self.s.whatsapp_phone_number_id}/media",
                data={"messaging_product": "whatsapp", "type": mime},
                files={"file": (path.name, f, mime)},
            )
        r.raise_for_status()
        return r.json()["id"]

    async def send_voice(self, to: str, ogg_path: Path, caption: str | None = None) -> str | None:
        media_id = await self.upload_media(ogg_path, "audio/ogg")
        audio: dict = {"id": media_id}
        if self.s.whatsapp_audio_as_voice:
            audio["voice"] = True
        try:
            msg_id = await self._send(to, {"type": "audio", "audio": audio})
        except httpx.HTTPStatusError:
            if "voice" not in audio:
                raise
            audio.pop("voice")  # older API versions reject the flag; send as a plain audio file
            msg_id = await self._send(to, {"type": "audio", "audio": audio})
        if caption:  # audio messages can't carry captions
            await self.send_text(to, caption)
        return msg_id

    async def send_buttons(
        self, to: str, text: str, buttons: list[Button], sources: list[Source] | None = None
    ) -> str | None:
        text = plain_sources(text, sources)
        return await self._send(
            to,
            {
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": text[:1024]},
                    "action": {
                        "buttons": [
                            {"type": "reply", "reply": {"id": b.id[:256], "title": b.title[:20]}}
                            for b in buttons[:3]
                        ]
                    },
                },
            },
        )

    async def _template(self, to: str, name: str, body_params: list[str], button_payloads: list[str]) -> str | None:
        components: list[dict] = []
        if body_params:
            components.append(
                # template params may not contain newlines/tabs or 4+ consecutive spaces
                {"type": "body", "parameters": [
                    {"type": "text", "text": " | ".join(line.strip() for line in p.splitlines() if line.strip())[:1000]}
                    for p in body_params
                ]}
            )
        for i, payload in enumerate(button_payloads):
            components.append(
                {
                    "type": "button",
                    "sub_type": "quick_reply",
                    "index": str(i),
                    "parameters": [{"type": "payload", "payload": payload}],
                }
            )
        return await self._send(
            to,
            {
                "type": "template",
                "template": {
                    "name": name,
                    "language": {"code": self.s.whatsapp_template_language},
                    "components": components,
                },
            },
        )

    async def window_open(self) -> bool:
        """True if the user messaged us in the last 24h (free-form messages allowed)."""
        async with session_scope() as s:
            last = await s.scalar(
                select(Message.created_at)
                .where(Message.channel == "whatsapp", Message.direction == "in")
                .order_by(Message.created_at.desc())
                .limit(1)
            )
        return bool(last and datetime.now(timezone.utc) - last < WINDOW - timedelta(minutes=5))

    async def send_briefing_notice(self, to: str, label: str, episode_id: int) -> str | None:
        return await self._template(to, self.s.whatsapp_briefing_template, [label], [f"play:{episode_id}"])

    async def send_proactive(self, to: str, text: str, sources: list[Source] | None = None) -> str | None:
        text = plain_sources(text, sources)
        if await self.window_open():
            return await self.send_text(to, text)
        return await self._template(to, self.s.whatsapp_alert_template, [text[:900]], [])

    async def download_media(self, media_ref: str) -> tuple[bytes, str]:
        meta = await self.http.get(f"{self.base}/{media_ref}")
        meta.raise_for_status()
        info = meta.json()
        r = await self.http.get(info["url"])
        r.raise_for_status()
        return r.content, info.get("mime_type", "audio/ogg")

    async def mark_read(self, external_id: str) -> None:
        try:
            await self.http.post(
                f"{self.base}/{self.s.whatsapp_phone_number_id}/messages",
                json={"messaging_product": "whatsapp", "status": "read", "message_id": external_id},
            )
        except httpx.HTTPError:
            pass

    async def close(self) -> None:
        await self.http.aclose()


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit and cur:
            parts.append(cur)
            cur = ""
        while len(para) > limit:
            parts.append(para[:limit])
            para = para[limit:]
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts
