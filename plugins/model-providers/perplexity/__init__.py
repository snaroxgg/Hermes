"""Perplexity AI provider profile.

Perplexity exposes an OpenAI-compatible chat completions endpoint.
Sonar models have real-time web search built in.
API docs: https://docs.perplexity.ai
"""

from providers import register_provider
from providers.base import ProviderProfile

perplexity = ProviderProfile(
    name="perplexity",
    aliases=("pplx",),
    env_vars=("PERPLEXITY_API_KEY",),
    display_name="Perplexity AI",
    description="Perplexity AI — Sonar models with real-time web search",
    signup_url="https://www.perplexity.ai/settings/api",
    fallback_models=(
        "sonar-pro",
        "sonar",
        "sonar-reasoning-pro",
        "sonar-reasoning",
        "sonar-deep-research",
        "r1-1776",
    ),
    base_url="https://api.perplexity.ai",
    supports_vision=True,
)

register_provider(perplexity)
