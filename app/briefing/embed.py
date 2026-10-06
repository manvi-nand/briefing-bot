"""Local embeddings via fastembed (ONNX, no API key) + cosine helpers."""

import asyncio
from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embedding_model)


def _embed_sync(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


async def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, get_settings().embedding_dim), dtype=np.float32)
    return await asyncio.to_thread(_embed_sync, texts)


def item_text(title: str, summary: str) -> str:
    return f"{title}. {summary[:400]}"
