"""Channel-agnostic messaging interface (PRD: keep agent logic portable across WhatsApp/Telegram)."""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# What outgoing messages are tagged as, so cleanup can target them ("briefing" or "chat").
send_category: ContextVar[str] = ContextVar("send_category", default="chat")


@dataclass
class Incoming:
    channel: str
    sender: str
    external_id: str
    kind: str  # text | voice | button
    text: str = ""
    media_ref: str | None = None
    button_id: str | None = None
    raw: dict = field(default_factory=dict)
    origin_message_id: int | None = None  # for button taps: the message the button belongs to


Source = tuple[str, str]  # (title, url)

STORY_HINT = 'Ask me about any story, e.g. "more on story 3", "more like story 2", or "less like story 5".'


def source_label(title: str, url: str) -> str:
    """Short display name: the site's domain without 'www.' (grounding titles are usually just domains)."""
    host = urlparse(url).netloc.removeprefix("www.")
    if not title or "." in title and " " not in title:
        return host or title or "source"
    return title if len(title) <= 40 else host


def plain_sources(text: str, sources: list[Source] | None) -> str:
    """Fallback rendering for channels without hyperlinks: numbered bare URLs."""
    if not sources:
        return text
    lines = "\n".join(f"{i}. {source_label(t, u)}: {u}" for i, (t, u) in enumerate(sources, 1))
    return f"{text}\n\nSources:\n{lines}" if text else f"Sources:\n{lines}"


@dataclass
class Reply:
    text: str
    sources: list[Source] = field(default_factory=list)

    def __str__(self) -> str:
        return plain_sources(self.text, self.sources)


@dataclass
class Button:
    id: str  # <= 256 chars; our payload, e.g. "fb:12:up"
    title: str  # WhatsApp limit: 20 chars


class Channel(ABC):
    name: str

    @property
    @abstractmethod
    def owner(self) -> str:
        """The one allowed user's address on this channel."""

    @abstractmethod
    async def send_text(self, to: str, text: str, sources: list[Source] | None = None) -> str | None: ...

    @abstractmethod
    async def send_voice(self, to: str, ogg_path: Path, caption: str | None = None) -> str | None: ...

    @abstractmethod
    async def send_buttons(
        self, to: str, text: str, buttons: list[Button], sources: list[Source] | None = None
    ) -> str | None: ...

    @abstractmethod
    async def send_briefing_notice(self, to: str, label: str, episode_id: int) -> str | None:
        """'Your briefing is ready' with a Play button (a template on WhatsApp)."""

    @abstractmethod
    async def send_proactive(self, to: str, text: str, sources: list[Source] | None = None) -> str | None:
        """Bot-initiated message (reminder/alert). Must work outside WhatsApp's 24h window."""

    @abstractmethod
    async def download_media(self, media_ref: str) -> tuple[bytes, str]:
        """Return (bytes, mime_type) for an inbound media reference."""

    async def mark_read(self, external_id: str) -> None:  # optional
        return None

    async def send_story_list(self, to: str, header: str, groups: list[tuple[str, list[tuple[int, str, str, str]]]],
                              episode_id: int) -> str | None:
        """One message listing the briefing's stories. groups = [(section title, [(n, title, source, url)])].
        Default: plain text with links (channels with in-place editing override this)."""
        lines = [header]
        for sec, stories in groups:
            lines.append(f"\n{sec}")
            lines += [f"{n}. {title} ({source}) {url}" for n, title, source, url in stories]
        lines.append(f"\n💬 {STORY_HINT}")
        return await self.send_text(to, "\n".join(lines))

    async def toggle_sources(self, message_id: int) -> None:
        return None

    async def delete_message(self, message_id: int) -> None:
        return None

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        return None

    async def record_incoming(self, inc: Incoming) -> None:  # optional: remember ids for cleanup
        return None

    async def clear(self, category: str | None = None, before: datetime | None = None) -> int:
        """Delete remembered messages (optionally one category / older than `before`). Returns count."""
        return 0

    async def close(self) -> None:
        return None
