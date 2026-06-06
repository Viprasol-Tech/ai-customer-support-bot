"""Tests for the confidence-gated SupportBot."""

from __future__ import annotations

import pytest

from ai_customer_support_bot.bot import SupportBot
from ai_customer_support_bot.kb import KnowledgeBase
from ai_customer_support_bot.providers import FakeLLM


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add_qa("How do I reset my password?", "Open Settings and reset it.")
    kb.add_qa("What are your support hours?", "9am-6pm IST, Mon-Fri.")
    return kb


def test_close_query_is_answered() -> None:
    bot = SupportBot(_kb(), threshold=0.2)
    response = bot.answer("how do I reset my password")
    assert response.answered is True
    assert response.escalated is False
    assert "reset" in response.answer.lower()


def test_unknown_query_escalates() -> None:
    bot = SupportBot(_kb(), threshold=0.3)
    response = bot.answer("can you ship a kangaroo to the moon")
    assert response.escalated is True
    assert response.answered is False
    assert response.answer == ""
    assert response.query == "can you ship a kangaroo to the moon"


def test_empty_kb_escalates() -> None:
    bot = SupportBot(KnowledgeBase(), threshold=0.3)
    response = bot.answer("anything at all")
    assert response.escalated is True
    assert response.score == 0.0


def test_threshold_boundary_answers_at_exact_score() -> None:
    kb = _kb()
    result = kb.retrieve("totally unrelated gibberish xyzzy")
    assert result is not None
    # At threshold 0.0, even a zero score meets the inclusive boundary.
    bot_at_score = SupportBot(kb, threshold=result.score)
    response = bot_at_score.answer("totally unrelated gibberish xyzzy")
    assert response.answered is True


def test_threshold_just_above_score_escalates() -> None:
    kb = _kb()
    query = "reset password instructions for my locked account today"
    bot_probe = SupportBot(kb, threshold=0.0)
    score = bot_probe.answer(query).score
    assert 0.0 < score < 1.0
    bot = SupportBot(kb, threshold=score + 0.01)
    response = bot.answer(query)
    assert response.escalated is True


def test_llm_phrases_answer() -> None:
    bot = SupportBot(_kb(), threshold=0.2, llm=FakeLLM(prefix="Hi! "))
    response = bot.answer("how do I reset my password")
    assert response.answer.startswith("Hi! ")
    assert "Open Settings" in response.answer


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        SupportBot(_kb(), threshold=1.5)
