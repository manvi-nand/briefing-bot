"""Text chunking for TTS and WAV stitching / OGG-Opus encoding (needs ffmpeg on PATH)."""

import asyncio
import io
import re
import shutil
import wave
from pathlib import Path

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 2400) -> list[str]:
    """Split text into chunks <= max_chars, breaking only at sentence boundaries when possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:  # a single monster sentence: split on commas/spaces
            cut = sentence.rfind(", ", 0, max_chars)
            if cut <= 0:
                cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head, sentence = sentence[: cut + 1].strip(), sentence[cut + 1 :].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def stitch_wavs(wavs: list[bytes], pause_ms: list[int] | int = 400) -> bytes:
    """Concatenate WAV byte blobs (same format) inserting silence after each one."""
    if not wavs:
        raise ValueError("no audio to stitch")
    pauses = pause_ms if isinstance(pause_ms, list) else [pause_ms] * len(wavs)
    out = io.BytesIO()
    params = None
    with wave.open(out, "wb") as dst:
        for blob, pause in zip(wavs, pauses, strict=False):
            with wave.open(io.BytesIO(blob), "rb") as src:
                p = src.getparams()
                if params is None:
                    params = p
                    dst.setnchannels(p.nchannels)
                    dst.setsampwidth(p.sampwidth)
                    dst.setframerate(p.framerate)
                elif (p.nchannels, p.sampwidth, p.framerate) != (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                ):
                    raise ValueError("WAV chunks have mismatched formats")
                dst.writeframes(src.readframes(src.getnframes()))
            silent_frames = int(params.framerate * pause / 1000)
            dst.writeframes(b"\x00" * silent_frames * params.sampwidth * params.nchannels)
    return out.getvalue()


def wav_seconds(blob: bytes) -> float:
    with wave.open(io.BytesIO(blob), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def to_wav_segments(audio: bytes, segment_seconds: int = 28) -> list[bytes]:
    """Decode any audio (e.g. OGG voice note) to 16 kHz mono WAV and split into short segments."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH (brew install ffmpeg)")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(audio)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {err.decode(errors='ignore')[:500]}")
    # ffmpeg writes a streaming header with unknown sizes; re-read frames robustly.
    data_start = out.find(b"data") + 8
    pcm = out[data_start:]
    frame_bytes = 2  # 16-bit mono
    per_segment = 16000 * segment_seconds * frame_bytes
    segments = []
    for i in range(0, len(pcm), per_segment):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm[i : i + per_segment])
        segments.append(buf.getvalue())
    return segments


async def to_ogg_opus(wav_bytes: bytes, out_path: Path, bitrate: str = "48k") -> Path:
    """Encode WAV -> OGG/Opus mono, the format WhatsApp/Telegram show as a voice note."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH (brew install ffmpeg)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", bitrate,
        "-application", "voip", str(out_path),
        stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate(wav_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[:500]}")
    return out_path
