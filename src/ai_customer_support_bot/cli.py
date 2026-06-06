"""Command-line interface for AI Customer Support Bot.

``ai-customer-support-bot demo`` loads a small FAQ knowledge base and shows the bot
answering a known question and escalating an unknown one — fully offline, using the
built-in deterministic fake providers (no API keys, no network).

Part of AI Customer Support Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import typer
from rich.console import Console

from ai_customer_support_bot import __version__
from ai_customer_support_bot.bot import BotResponse, SupportBot
from ai_customer_support_bot.kb import KnowledgeBase
from ai_customer_support_bot.providers import FakeEmbedder, FakeLLM

app = typer.Typer(add_completion=False, help="AI Customer Support Bot — by Viprasol Tech.")
console = Console()


def _sample_kb(embedder: FakeEmbedder | None) -> KnowledgeBase:
    """Build a small demonstration knowledge base."""
    kb = KnowledgeBase(embedder=embedder)
    kb.add_qa("How do I reset my password?", "Open Settings > Security and click 'Reset password'.")
    kb.add_qa(
        "What are your support hours?", "Our team is available 9am-6pm IST, Monday to Friday."
    )
    kb.add_qa(
        "How do I get a refund?", "Refunds are issued within 5-7 business days to the source."
    )
    kb.add_qa(
        "Where can I download an invoice?",
        "Invoices live under Billing > Invoices in your account.",
    )
    return kb


def _render(response: BotResponse) -> None:
    """Print a single bot response to the console (ASCII-only)."""
    console.print(f"Query:  [bold]{response.query}[/]")
    if response.escalated:
        console.print(f"Result: [yellow]ESCALATE to human[/] (score {response.score:.2f})")
    else:
        console.print(f"Result: [green]ANSWER[/] (score {response.score:.2f})")
        console.print(f"Answer: {response.answer}")
    console.print("")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ai-customer-support-bot [bold cyan]{__version__}[/] -- by Viprasol Tech")


@app.command()
def demo(
    embeddings: bool = typer.Option(
        False, "--embeddings", help="Use the fake cosine embedder instead of term overlap."
    ),
    threshold: float = typer.Option(0.3, help="Confidence threshold to answer vs escalate."),
) -> None:
    """Answer a known question and escalate an unknown one, fully offline."""
    embedder = FakeEmbedder() if embeddings else None
    kb = _sample_kb(embedder)
    bot = SupportBot(kb, threshold=threshold, llm=FakeLLM())

    mode = "embedding cosine" if embeddings else "term overlap"
    console.print(
        f"[bold]AI Customer Support Bot[/] demo -- retrieval: {mode}, threshold {threshold:.2f}\n"
    )

    _render(bot.answer("I forgot my password, how can I reset it?"))
    _render(bot.answer("Do you ship hardware to Antarctica overnight?"))


if __name__ == "__main__":
    app()
