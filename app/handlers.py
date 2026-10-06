"""Inbound message handling: allow-list, dedupe, voice -> text, buttons, agent, reply in the same mode."""

import logging
import re
import uuid

from sqlalchemy.exc import IntegrityError

from app import costs, trackers
from app.agent.agent import respond
from app.briefing import pipeline
from app.channels import Channel, Incoming, get_channel
from app.channels.base import Reply
from app.config import get_settings
from app.db import session_scope
from app.models import Message
from app.speech import audio, sarvam

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")

HELP = """Hi! I'm your briefing assistant.
- Every morning I send your audio briefing.
- Ask me anything by text or voice note.
- "go deeper on story 2", "remind me Friday 10am to review the PR", "add X to my reading list",
  "tell me when Sarvam releases a new model", "less crypto", "what do you know about me?"
- "send my briefing at 8:15 on weekdays", "make it 5 minutes", "more India news", "skip sports"
- "every weekday at 6pm send me the Nifty and Sensex close", "every Monday 9am remind me to submit my timesheet",
  "what tasks do I have?", "pause the Nifty one", "delete task 12"
Commands: /settings (briefing time & mix), /play (latest briefing), /build (build a fresh one now), /cost (spend this week),\n/clear (delete messages from the last 48 hours; older ones: Telegram's 'Clear history')"""


async def _save(channel: str, direction: str, mtype: str, text: str, external_id: str | None = None,
                media_ref: str | None = None) -> int | None:
    try:
        async with session_scope() as s:
            m = Message(channel=channel, external_id=external_id, direction=direction, type=mtype, text=text,
                        media_ref=media_ref)
            s.add(m)
            await s.flush()
            return m.id
    except IntegrityError:
        return None  # duplicate webhook delivery


async def _set_text(message_id: int, text: str) -> None:
    async with session_scope() as s:
        m = await s.get(Message, message_id)
        if m:
            m.text = text


async def reply(ch: Channel, answer: Reply, *, as_voice: bool) -> None:
    to = ch.owner
    if as_voice:
        spoken = URL_RE.sub("", answer.text.replace("**", "")).strip()
        try:
            wav, _ = await sarvam.synthesize([spoken], stage="tts_reply")
            path = get_settings().audio_dir / f"reply-{uuid.uuid4().hex[:10]}.ogg"
            await audio.to_ogg_opus(wav, path)
            await ch.send_voice(to, path)
            path.unlink(missing_ok=True)
            if answer.sources:
                await ch.send_text(to, "", answer.sources)
        except Exception:
            log.exception("voice reply failed; sending text")
            await ch.send_text(to, answer.text, answer.sources)
    else:
        await ch.send_text(to, answer.text, answer.sources)
    await _save(ch.name, "out", "voice" if as_voice else "text", answer.text)


async def _handle_button(ch: Channel, inc: Incoming) -> tuple[str | None, str | None]:
    """Returns (text to route to the agent or None if fully handled, pop-up text for the tap)."""
    bid = inc.button_id or ""
    origin = inc.origin_message_id
    if bid.startswith("play:"):
        ep_id = bid.split(":", 1)[1]
        await pipeline.play_episode(int(ep_id) if ep_id.isdigit() else None)
        if origin:
            await ch.delete_message(origin)  # the "briefing ready" notice has done its job
        return None, None
    if bid == "src:toggle":
        if origin:
            await ch.toggle_sources(origin)
        return None, None
    if bid.startswith("fb:"):
        parts = bid.split(":")  # fb:<item>:<up|down> (buttons on older messages)
        msg = await trackers.record_story_feedback(int(parts[1]), 1 if parts[2] == "up" else -1)
        return None, ("👍 " if parts[2] == "up" else "👎 ") + msg
    if bid.startswith("deep:"):
        n = bid.split(":", 1)[1]
        return f"Go deeper on story {n}.", f"Looking into story {n}…"
    if bid.startswith("react:"):
        return None, None  # emoji reactions on arbitrary messages: ignore for now
    return inc.text or bid or None, None


async def _command(ch: Channel, text: str) -> bool:
    cmd = text.strip().lower().split()[0] if text.strip() else ""
    if cmd in {"/start", "/help", "help"}:
        await ch.send_text(ch.owner, HELP)
    elif cmd == "/play":
        await pipeline.play_episode()
    elif cmd == "/build":
        from app.scheduler import build_lock

        if build_lock.locked():
            await ch.send_text(ch.owner, "A briefing is already building; I'll let you know when it's ready.")
        else:
            await ch.send_text(ch.owner, "Building a fresh briefing; this takes a few minutes…")
            async with build_lock:
                ep = await pipeline.build_episode()
            await pipeline.play_episode(ep.id)
    elif cmd == "/settings":
        from app.prefs import describe, get_prefs

        await ch.send_text(ch.owner, "Your briefing settings:\n" + describe(await get_prefs())
                           + "\n\nChange anything by just telling me, e.g. \"send it at 8:15 on weekdays\".")
    elif cmd == "/clear":
        n = await ch.clear()
        await ch.send_text(ch.owner, f"🧹 Cleared {n} messages. Anything older than 48 hours has to be removed "
                                     "with Telegram's 'Clear history'.")
    elif cmd == "/cost":
        rows = await costs.cost_report(7)
        total = sum(c for _, _, c in rows)
        lines = [f"Last 7 days: ${total:.2f} (today ${await costs.spent_today():.2f})"]
        lines += [f"- {p}/{st}: ${c:.3f}" for p, st, c in rows[:10]]
        await ch.send_text(ch.owner, "\n".join(lines))
    else:
        return False
    return True


async def handle(inc: Incoming) -> None:
    ch = get_channel(inc.channel)

    if not ch.owner:
        if inc.channel == "telegram" and inc.kind == "text":
            await ch.send_text(inc.sender, f"Your chat id is {inc.sender}. Put TELEGRAM_CHAT_ID={inc.sender} in .env "
                                           "and restart.")
        log.warning("%s owner not configured; ignoring message from %s", inc.channel, inc.sender)
        return
    if inc.sender != ch.owner:
        log.warning("ignoring message from non-owner %s on %s", inc.sender, inc.channel)
        return

    msg_id = await _save(inc.channel, "in", inc.kind, inc.text, inc.external_id, inc.media_ref)
    if msg_id is None:
        return
    await ch.record_incoming(inc)
    await ch.mark_read(inc.external_id)

    try:
        text = inc.text
        as_voice = False
        if inc.kind == "button":
            toast = None
            try:
                text, toast = await _handle_button(ch, inc)
            finally:
                if inc.external_id.startswith("cb:"):  # always stop the button's spinner
                    await ch.answer_callback(inc.external_id[3:], toast)
            if text is None:
                return
            await _set_text(msg_id, text)
        elif inc.kind == "voice":
            data, mime = await ch.download_media(inc.media_ref)
            text = await sarvam.transcribe(data, mime=mime)
            if not text:
                await ch.send_text(ch.owner, "Sorry, I couldn't make out that voice note.")
                return
            await _set_text(msg_id, text)
            as_voice = True
        elif await _command(ch, text):
            return

        answer = await respond(text, channel=inc.channel, message_id=msg_id)
        await reply(ch, answer, as_voice=as_voice)
    except costs.BudgetExceeded as e:
        await ch.send_text(ch.owner, f"⚠️ {e} Raise DAILY_BUDGET_USD to continue today.")
    except Exception as e:
        log.exception("handling message failed")
        await ch.send_text(ch.owner, f"Sorry, something went wrong: {type(e).__name__}")
