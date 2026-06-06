"""Pluggable model providers and built-in deterministic fakes.

Model access is intentionally swappable so the library runs fully offline in tests
and demos, yet can be wired to any real backend in production.

- An :data:`LLM` is any ``Callable[[str], str]`` — given a prompt, return text.
- An :data:`Embedder` is any ``Callable[[str], list[float]]`` — given text, return a
  fixed-length vector.

The fakes here are *real* implementations (a hash-based embedder and a templated
echo LLM), not stubs: they are deterministic and dependency-free, so behaviour is
reproducible without any network access or API keys.

Part of AI Customer Support Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

__all__ = [
    "LLM",
    "Embedder",
    "FakeEmbedder",
    "FakeLLM",
    "echo_llm",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class LLM(Protocol):
    """A callable that turns a prompt into a text completion."""

    def __call__(self, prompt: str) -> str:  # pragma: no cover - protocol definition
        ...


@runtime_checkable
class Embedder(Protocol):
    """A callable that turns text into a fixed-length embedding vector."""

    def __call__(self, text: str) -> list[float]:  # pragma: no cover - protocol definition
        ...


def _tokenize(text: str) -> list[str]:
    """Lower-case word/number tokenisation shared across providers."""
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic, offline embedder built from token hashes.

    Each token is hashed into one bucket of a fixed-length vector using a bag-of-words
    scheme, then the vector is L2-normalised. Because identical tokens map to identical
    buckets, texts that share vocabulary point in similar directions — so cosine
    similarity behaves meaningfully without any trained model or network call.

    Args:
        dim: Length of the produced vectors. Must be positive.
    """

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim

    def __call__(self, text: str) -> list[float]:
        """Embed ``text`` into an L2-normalised vector of length ``self.dim``."""
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            vec[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class FakeLLM:
    """Deterministic, offline LLM that phrases an answer from supplied context.

    It does not invent content: it returns the retrieved answer verbatim, optionally
    wrapped in a fixed, friendly template. This keeps the demo and tests reproducible
    while exercising the same call site a real model would occupy.

    Args:
        prefix: Text prepended to the answer (set to ``""`` to echo verbatim).
    """

    def __init__(self, prefix: str = "Thanks for reaching out! ") -> None:
        self.prefix = prefix

    def __call__(self, prompt: str) -> str:
        """Return a phrased completion for ``prompt``.

        The bot passes prompts of the form ``"Answer: <text>"``; the embedded answer is
        extracted and returned with the configured prefix. Any other prompt is echoed.
        """
        marker = "Answer:"
        idx = prompt.rfind(marker)
        answer = prompt[idx + len(marker) :].strip() if idx != -1 else prompt.strip()
        return f"{self.prefix}{answer}"


def echo_llm(prompt: str) -> str:
    """Trivial :data:`LLM` that returns the prompt unchanged.

    Useful as a minimal default when no phrasing is desired.
    """
    return prompt
