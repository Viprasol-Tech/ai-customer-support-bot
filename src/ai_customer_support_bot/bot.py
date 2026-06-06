"""SupportBot — answer from the knowledge base or escalate on low confidence.

The bot retrieves the closest FAQ entry for a query. If the similarity score meets the
configured ``threshold`` it answers (optionally phrasing the answer through a pluggable
LLM); otherwise it returns an *escalation* result carrying the original query so a human
agent can take over.

Part of AI Customer Support Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai_customer_support_bot.kb import FAQEntry, KnowledgeBase
from ai_customer_support_bot.providers import LLM

__all__ = ["BotResponse", "SupportBot"]


class BotResponse(BaseModel):
    """Outcome of handling a single user query.

    Attributes:
        query: The original user query.
        answered: ``True`` if the bot answered, ``False`` if it escalated.
        answer: The answer text (empty when escalated).
        score: Similarity score of the best-matched entry in ``[0.0, 1.0]``.
        matched: The matched FAQ entry, if any.
        escalated: ``True`` when the query was handed off to a human.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    answered: bool
    answer: str = ""
    score: float = Field(0.0, ge=0.0, le=1.0)
    matched: FAQEntry | None = None
    escalated: bool = False


class SupportBot:
    """A confidence-gated FAQ support bot.

    Args:
        kb: The knowledge base to answer from.
        threshold: Minimum similarity score (inclusive) required to answer. Queries
            scoring below this are escalated. Must be in ``[0.0, 1.0]``.
        llm: Optional pluggable LLM used to phrase the final answer. When ``None``,
            the matched answer is returned verbatim.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        threshold: float = 0.3,
        llm: LLM | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.kb = kb
        self.threshold = threshold
        self._llm = llm

    def _phrase(self, answer: str) -> str:
        """Phrase ``answer`` through the LLM when configured, else return it verbatim."""
        if self._llm is None:
            return answer
        return self._llm(f"Answer: {answer}")

    def answer(self, query: str) -> BotResponse:
        """Handle ``query``: answer if confident enough, otherwise escalate.

        Args:
            query: The user's question.

        Returns:
            A :class:`BotResponse`. When the best match meets ``threshold`` it is
            answered; otherwise ``escalated`` is ``True`` and ``answer`` is empty.
        """
        result = self.kb.retrieve(query)
        if result is None or result.score < self.threshold:
            score = result.score if result is not None else 0.0
            matched = result.entry if result is not None else None
            return BotResponse(
                query=query,
                answered=False,
                answer="",
                score=score,
                matched=matched,
                escalated=True,
            )
        return BotResponse(
            query=query,
            answered=True,
            answer=self._phrase(result.entry.answer),
            score=result.score,
            matched=result.entry,
            escalated=False,
        )
