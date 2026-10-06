"""Sarvam AI speech: Bulbul TTS and Saaras STT (REST)."""

import asyncio
import base64
import logging
import time

import httpx

from app import costs
from app.config import get_settings
from app.speech.audio import chunk_text, stitch_wavs, to_wav_segments, wav_seconds

log = logging.getLogger(__name__)

BASE = "https://api.sarvam.ai"
TTS_MAX_CHARS = 2400  # bulbul:v3 accepts up to 2,500 characters per request


def _headers() -> dict:
    key = get_settings().sarvam_api_key
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set")
    return {"api-subscription-key": key}


async def tts_chunk(text: str, *, language: str | None = None, client: httpx.AsyncClient) -> bytes:
    s = get_settings()
    body = {
        "text": text,
        "language_code": language or s.sarvam_tts_language,
        "model": s.sarvam_tts_model,
        "speaker": s.sarvam_tts_speaker,
        "pace": s.sarvam_tts_pace,
        "speech_sample_rate": 24000,
        "output_audio_codec": "wav",
    }
    for attempt in range(3):
        r = await client.post(f"{BASE}/text-to-speech", json=body, headers=_headers())
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            await asyncio.sleep(2**attempt)
            continue
        r.raise_for_status()
        return base64.b64decode(r.json()["audios"][0])
    raise RuntimeError("unreachable")


async def synthesize(segments: list[str], *, language: str | None = None, stage: str = "tts") -> tuple[bytes, float]:
    """TTS each segment (chunked at sentence boundaries) and stitch with pauses.

    Longer pause between segments, short pause between chunks of the same segment.
    Returns (wav_bytes, seconds).
    """
    await costs.check_budget(stage)
    started = time.monotonic()
    pieces: list[tuple[str, int]] = []
    for seg in segments:
        chunks = chunk_text(seg, TTS_MAX_CHARS)
        for i, c in enumerate(chunks):
            pieces.append((c, 900 if i == len(chunks) - 1 else 250))
    if not pieces:
        raise ValueError("nothing to synthesize")

    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=120) as client:

        async def one(text: str) -> bytes:
            async with sem:
                return await tts_chunk(text, language=language, client=client)

        wavs = await asyncio.gather(*(one(t) for t, _ in pieces))

    audio = stitch_wavs(list(wavs), [p for _, p in pieces])
    chars = sum(len(t) for t, _ in pieces)
    await costs.record(
        "sarvam", stage, {"chars": chars, "requests": len(pieces)},
        chars * costs.SARVAM_TTS_PER_CHAR, int((time.monotonic() - started) * 1000),
    )
    return audio, wav_seconds(audio)


async def _stt_request(client: httpx.AsyncClient, audio: bytes, filename: str, mime: str) -> str:
    s = get_settings()
    r = await client.post(
        f"{BASE}/speech-to-text",
        headers=_headers(),
        files={"file": (filename, audio, mime)},
        data={"model": s.sarvam_stt_model, "language_code": s.sarvam_stt_language},
    )
    r.raise_for_status()
    return (r.json().get("transcript") or "").strip()


async def transcribe(audio: bytes, *, filename: str = "voice.ogg", mime: str = "audio/ogg") -> str:
    """Transcribe a voice note. The REST API handles ~30 s per request, so longer clips are split."""
    await costs.check_budget("stt")
    started = time.monotonic()
    segments = await to_wav_segments(audio, segment_seconds=28)
    seconds = sum(wav_seconds(seg) for seg in segments)
    async with httpx.AsyncClient(timeout=120) as client:
        parts = await asyncio.gather(
            *(_stt_request(client, seg, f"part{i}.wav", "audio/wav") for i, seg in enumerate(segments))
        )
    await costs.record(
        "sarvam", "stt", {"seconds": round(seconds, 1), "requests": len(segments)},
        seconds * costs.SARVAM_STT_PER_SEC, int((time.monotonic() - started) * 1000),
    )
    return " ".join(p for p in parts if p).strip()
