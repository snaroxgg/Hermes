"""Perplexity Sonar web search provider.

Uses the Perplexity chat completions API with Sonar models — every response
includes real-time web citations, so we extract them as search results.

Config keys::

    web:
      search_backend: "perplexity"
      backend: "perplexity"

Env vars::

    PERPLEXITY_API_KEY=pplx-...   # https://www.perplexity.ai/settings/api
    PERPLEXITY_SEARCH_MODEL=...   # optional, default: sonar-pro
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sonar-pro"
_BASE_URL = "https://api.perplexity.ai"


class PerplexityWebSearchProvider(WebSearchProvider):
    """Perplexity Sonar — real-time web search via chat completions API."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity Sonar"

    def is_available(self) -> bool:
        return bool(os.getenv("PERPLEXITY_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
        if not api_key:
            return {
                "success": False,
                "error": (
                    "PERPLEXITY_API_KEY not set. "
                    "Get your key at https://www.perplexity.ai/settings/api"
                ),
            }

        model = os.getenv("PERPLEXITY_SEARCH_MODEL", _DEFAULT_MODEL).strip()

        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx is required (pip install httpx)"}

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Answer the query concisely "
                        "and include relevant facts. Use web search results."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 1024,
            "return_citations": True,
            "return_related_questions": False,
        }

        try:
            resp = httpx.post(
                f"{_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            return {"success": False, "error": f"Perplexity API error: {exc.response.status_code} {exc.response.text[:300]}"}
        except Exception as exc:
            return {"success": False, "error": f"Perplexity request failed: {exc}"}

        # Extract the answer text
        answer = ""
        choices = data.get("choices", [])
        if choices:
            answer = choices[0].get("message", {}).get("content", "")

        # Extract citations as search results
        citations: List[str] = data.get("citations", [])
        web_results = []
        for i, url in enumerate(citations[:limit]):
            web_results.append({
                "title": f"Source {i + 1}",
                "url": url,
                "description": "",
                "position": i + 1,
            })

        # If no citations, return the answer itself as the single result
        if not web_results:
            web_results.append({
                "title": "Perplexity Answer",
                "url": "",
                "description": answer,
                "position": 1,
            })
        else:
            # Prepend the answer as the first item so the agent sees it directly
            web_results.insert(0, {
                "title": "Perplexity Answer",
                "url": f"{_BASE_URL}/search?q={query.replace(' ', '+')}",
                "description": answer,
                "position": 0,
            })

        logger.info("Perplexity search '%s': %d citations", query, len(citations))
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity Sonar",
            "badge": "paid",
            "tag": "Real-time web search with AI-synthesized answers and citations.",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key (pplx-...)",
                    "url": "https://www.perplexity.ai/settings/api",
                },
                {
                    "key": "PERPLEXITY_SEARCH_MODEL",
                    "prompt": "Sonar model (default: sonar-pro)",
                    "url": "",
                    "optional": True,
                },
            ],
        }
