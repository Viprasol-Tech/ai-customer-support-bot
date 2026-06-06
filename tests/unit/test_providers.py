"""Tests for the pluggable providers and deterministic fakes."""

from __future__ import annotations

import pytest

from ai_customer_support_bot.providers import Embedder, FakeEmbedder, FakeLLM, echo_llm


def test_fake_embedder_is_deterministic() -> None:
    emb = FakeEmbedder(dim=64)
    assert emb("hello world") == emb("hello world")


def test_fake_embedder_dim_and_norm() -> None:
    emb = FakeEmbedder(dim=32)
    vec = emb("reset my password")
    assert len(vec) == 32
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0)


def test_fake_embedder_empty_text_zero_vector() -> None:
    emb = FakeEmbedder(dim=16)
    assert emb("") == [0.0] * 16


def test_fake_embedder_rejects_bad_dim() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


def test_fake_embedder_satisfies_protocol() -> None:
    emb: Embedder = FakeEmbedder()
    assert isinstance(emb, Embedder)


def test_fake_llm_extracts_answer() -> None:
    llm = FakeLLM(prefix="Hello! ")
    assert llm("Answer: do the thing") == "Hello! do the thing"


def test_echo_llm_round_trip() -> None:
    assert echo_llm("ping") == "ping"
