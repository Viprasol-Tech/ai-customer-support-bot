"""Tests for the knowledge base retrieval logic."""

from __future__ import annotations

import pytest

from ai_customer_support_bot.kb import FAQEntry, KnowledgeBase
from ai_customer_support_bot.providers import FakeEmbedder


def _kb(embedder: FakeEmbedder | None = None) -> KnowledgeBase:
    kb = KnowledgeBase(embedder=embedder)
    kb.add_qa("How do I reset my password?", "Open Settings and reset it.")
    kb.add_qa("What are your support hours?", "9am-6pm IST, Mon-Fri.")
    kb.add_qa("How do I get a refund?", "Refunds take 5-7 business days.")
    return kb


def test_add_and_len() -> None:
    kb = _kb()
    assert len(kb) == 3
    kb.add(FAQEntry(question="New?", answer="Yes."))
    assert len(kb) == 4


def test_empty_kb_returns_none() -> None:
    assert KnowledgeBase().retrieve("anything") is None


def test_term_overlap_retrieves_right_entry() -> None:
    kb = _kb()
    result = kb.retrieve("how can I reset my password please")
    assert result is not None
    assert "reset" in result.entry.answer.lower()
    assert result.score > 0.0


def test_embedding_cosine_retrieves_right_entry() -> None:
    kb = _kb(embedder=FakeEmbedder())
    result = kb.retrieve("I want a refund for my order")
    assert result is not None
    assert "refund" in result.entry.answer.lower()


def test_score_bounds() -> None:
    kb = _kb(embedder=FakeEmbedder())
    result = kb.retrieve("support hours today")
    assert result is not None
    assert 0.0 <= result.score <= 1.0


def test_identical_query_scores_high_with_embedder() -> None:
    kb = _kb(embedder=FakeEmbedder())
    result = kb.retrieve("How do I reset my password?")
    assert result is not None
    assert result.entry.question == "How do I reset my password?"
    assert result.score == pytest.approx(1.0)


def test_entries_property_is_copy() -> None:
    kb = _kb()
    entries = kb.entries
    entries.clear()
    assert len(kb) == 3
