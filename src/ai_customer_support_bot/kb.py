"""Knowledge base of FAQ entries with term-overlap and embedding retrieval.

A :class:`KnowledgeBase` holds ``(question, answer)`` entries and finds the best match
for a user query. Two scoring modes are supported:

- **Term overlap** (default): Jaccard similarity over token sets — fast, dependency-free,
  and good for short FAQ questions.
- **Embedding cosine**: pass a pluggable :data:`~ai_customer_support_bot.providers.Embedder`
  and matches are ranked by genuine cosine similarity of the embedding vectors.

Both modes return a similarity score in ``[0.0, 1.0]`` so a caller can threshold on
confidence.

Part of AI Customer Support Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ai_customer_support_bot.providers import Embedder

__all__ = ["FAQEntry", "KnowledgeBase", "RetrievalResult"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Return the set of lower-cased word/number tokens in ``text``."""
    return set(_TOKEN_RE.findall(text.lower()))


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity of two equal-length vectors.

    Returns ``0.0`` if either vector has zero magnitude.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot: float = sum(x * y for x, y in zip(a, b, strict=True))
    na: float = sum(x * x for x in a) ** 0.5
    nb: float = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    # Clamp to [0, 1] to absorb floating-point error (e.g. 1.0000000000000002).
    cosine: float = dot / (na * nb)
    return max(0.0, min(1.0, cosine))


class FAQEntry(BaseModel):
    """A single frequently-asked-question entry."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class RetrievalResult(BaseModel):
    """The best-matching FAQ entry for a query and its similarity score."""

    entry: FAQEntry
    score: float = Field(..., ge=0.0, le=1.0)


class KnowledgeBase:
    """A searchable collection of FAQ entries.

    Args:
        entries: Optional initial entries to load.
        embedder: Optional pluggable embedder. When provided, retrieval uses cosine
            similarity of embeddings; otherwise term-overlap (Jaccard) is used.
    """

    def __init__(
        self,
        entries: list[FAQEntry] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._entries: list[FAQEntry] = []
        self._embedder = embedder
        self._embeddings: list[list[float]] = []
        for entry in entries or []:
            self.add(entry)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[FAQEntry]:
        """Return a shallow copy of the stored entries."""
        return list(self._entries)

    def add(self, entry: FAQEntry) -> None:
        """Add an FAQ entry, pre-computing its embedding when an embedder is set."""
        self._entries.append(entry)
        if self._embedder is not None:
            self._embeddings.append(self._embedder(entry.question))

    def add_qa(self, question: str, answer: str) -> None:
        """Convenience wrapper to add an entry from raw strings."""
        self.add(FAQEntry(question=question, answer=answer))

    def _score(self, query: str, index: int) -> float:
        """Score ``query`` against the entry at ``index``."""
        if self._embedder is not None:
            return _cosine(self._embedder(query), self._embeddings[index])
        q_tokens = _tokens(query)
        e_tokens = _tokens(self._entries[index].question)
        if not q_tokens or not e_tokens:
            return 0.0
        union = q_tokens | e_tokens
        return len(q_tokens & e_tokens) / len(union)

    def retrieve(self, query: str) -> RetrievalResult | None:
        """Return the best-matching entry for ``query``, or ``None`` if KB is empty.

        Args:
            query: The user's question.

        Returns:
            A :class:`RetrievalResult` with the top entry and its similarity score,
            or ``None`` when the knowledge base contains no entries.
        """
        if not self._entries:
            return None
        best_index = 0
        best_score = self._score(query, 0)
        for index in range(1, len(self._entries)):
            score = self._score(query, index)
            if score > best_score:
                best_score = score
                best_index = index
        return RetrievalResult(entry=self._entries[best_index], score=best_score)
