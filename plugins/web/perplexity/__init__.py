"""Perplexity Sonar web search plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.perplexity.provider import PerplexityWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(PerplexityWebSearchProvider())
