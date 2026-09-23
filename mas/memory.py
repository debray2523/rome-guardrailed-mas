"""Persistent cross-session memory: Mem0 OSS + SQLite history + Chroma vectors.

Token discipline: the graph never re-sends raw conversation history. Before
planning it retrieves only the top-k (default 3) semantically relevant facts
for the user; after a completed run it stores the exchange, which Mem0
condenses into short facts (infer=True).
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any

from . import config

os.environ.setdefault("MEM0_TELEMETRY", "False")


def mem0_config(data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = Path(data_dir or config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        "llm": {
            "provider": "groq",
            "config": {"model": config.GROQ_MODEL, "temperature": 0.0, "max_tokens": 1000},
        },
        "embedder": {
            "provider": "huggingface",  # local sentence-transformers, no API cost
            "config": {"model": config.EMBED_MODEL, "embedding_dims": config.EMBED_DIMS},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "rome_mas_memory", "path": str(data_dir / "chroma")},
        },
        "history_db_path": str(data_dir / "mem0_history.db"),  # SQLite
    }


class HashingEmbedder:
    """Offline, deterministic bag-of-words embedder (384 dims) used only by the
    offline demo and tests, so persistence can be shown without downloads or keys."""

    dims = config.EMBED_DIMS

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        vec = [0.0] * self.dims
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(t, memory_action) for t in texts]


class MemoryStore:
    """Thin wrapper so the graph depends on two verbs: recall() and remember()."""

    def __init__(self, mem0_memory: Any, infer: bool = True):
        self._m = mem0_memory
        self.infer = infer  # False in offline mode: store raw text, no LLM call

    @classmethod
    def local(cls, data_dir: Path | None = None) -> "MemoryStore":
        from mem0 import Memory

        return cls(Memory.from_config(mem0_config(data_dir)), infer=True)

    @classmethod
    def offline(cls, data_dir: Path | None = None) -> "MemoryStore":
        """Same Mem0 + SQLite + Chroma stack, but with the HashingEmbedder and no LLM use."""
        from mem0 import Memory

        cfg = mem0_config(Path(data_dir or config.DATA_DIR / "offline"))
        # Providers only need to construct; they are never called in offline mode.
        cfg["embedder"] = {"provider": "openai", "config": {"api_key": "offline", "embedding_dims": 384}}
        cfg["llm"]["config"]["api_key"] = "offline"
        mem = Memory.from_config(cfg)
        mem.embedding_model = HashingEmbedder()
        return cls(mem, infer=False)

    # sync API --------------------------------------------------------------
    def recall_sync(self, user_id: str, query: str, k: int = config.MEMORY_TOP_K) -> list[str]:
        res = self._m.search(query, filters={"user_id": user_id}, top_k=k, threshold=0.0)
        return [r["memory"] for r in res.get("results", [])][:k]

    def remember_sync(self, user_id: str, messages: list[dict[str, str]]) -> None:
        self._m.add(messages, user_id=user_id, infer=self.infer)

    def all(self, user_id: str) -> list[str]:
        res = self._m.get_all(filters={"user_id": user_id})
        return [r["memory"] for r in res.get("results", [])]

    # async API used by graph nodes ---------------------------------------
    async def recall(self, user_id: str, query: str, k: int = config.MEMORY_TOP_K) -> list[str]:
        return await asyncio.to_thread(self.recall_sync, user_id, query, k)

    async def remember(self, user_id: str, messages: list[dict[str, str]]) -> None:
        await asyncio.to_thread(self.remember_sync, user_id, messages)
