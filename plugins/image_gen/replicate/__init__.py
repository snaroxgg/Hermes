"""Replicate image generation backend.

Wraps a curated catalog of Replicate's top text-to-image models as an
:class:`ImageGenProvider`. The default — ``nano-banana-2`` (Google's flagship
Gemini image model) — is the single "best" image model requested at setup.

Model selection precedence (first hit wins):
1. ``model=`` kwarg from the tool call
2. ``REPLICATE_IMAGE_MODEL`` env var
3. ``image_gen.replicate.model`` in ``config.yaml``
4. :data:`DEFAULT_MODEL`

Replicate returns an HTTPS ``replicate.delivery`` URL; we materialise the bytes
into ``$HERMES_HOME/cache/images/`` so the gateway has a stable local path
(those URLs expire within the hour).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# ``ref`` is the Replicate ``owner/name`` slug (latest version is auto-resolved).

_MODELS: Dict[str, Dict[str, Any]] = {
    "nano-banana-2": {
        "ref": "google/nano-banana-2",
        "display": "Nano Banana 2 (Google)",
        "speed": "~10-20s",
        "strengths": "Google's flagship Gemini image model — best overall quality, conversational, strong text rendering.",
        "price": "see replicate.com/google/nano-banana-2",
    },
    "flux-2-pro": {
        "ref": "black-forest-labs/flux-2-pro",
        "display": "FLUX 2 Pro",
        "speed": "~10-20s",
        "strengths": "Black Forest Labs flagship — photoreal detail, prompt adherence, editing.",
        "price": "see replicate.com/black-forest-labs/flux-2-pro",
    },
    "imagen-4-ultra": {
        "ref": "google/imagen-4-ultra",
        "display": "Imagen 4 Ultra (Google)",
        "speed": "~10-20s",
        "strengths": "Highest-fidelity Imagen 4 tier; excels at fine detail and typography.",
        "price": "see replicate.com/google/imagen-4-ultra",
    },
    "nano-banana": {
        "ref": "google/nano-banana",
        "display": "Nano Banana (Google)",
        "speed": "~8-15s",
        "strengths": "Gemini 2.5 image editing model; great for instruction-based edits.",
        "price": "see replicate.com/google/nano-banana",
    },
}

DEFAULT_MODEL = "nano-banana-2"

# Hermes' three coarse aspect buckets → Replicate's ratio enum.
_ASPECT_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


# ---------------------------------------------------------------------------
# Config / model resolution
# ---------------------------------------------------------------------------


def _load_replicate_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        rep = section.get("replicate") if isinstance(section, dict) else None
        return rep if isinstance(rep, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen.replicate config: %s", exc)
        return {}


def _resolve_model(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    candidates: List[Optional[str]] = [
        explicit,
        os.environ.get("REPLICATE_IMAGE_MODEL"),
        _load_replicate_config().get("model"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip() in _MODELS:
            cid = c.strip()
            return cid, _MODELS[cid]
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ReplicateImageGenProvider(ImageGenProvider):
    """Replicate multi-model image generation backend."""

    @property
    def name(self) -> str:
        return "replicate"

    @property
    def display_name(self) -> str:
        return "Replicate"

    def is_available(self) -> bool:
        try:
            from tools.replicate_common import replicate_available

            return replicate_available()
        except Exception:  # noqa: BLE001 — never break the picker
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mid,
                "display": meta["display"],
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
                "price": meta.get("price", ""),
            }
            for mid, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Replicate",
            "badge": "paid",
            "tag": "Nano Banana 2, FLUX 2 Pro, Imagen 4 Ultra — text-to-image",
            "env_vars": [
                {
                    "key": "REPLICATE_API_TOKEN",
                    "prompt": "Replicate API token",
                    "url": "https://replicate.com/account/api-tokens",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from tools.replicate_common import ReplicateError, first_url, run_model

        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id, meta = _resolve_model(kwargs.get("model"))

        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_input",
                provider="replicate",
                model=model_id,
                prompt="",
                aspect_ratio=aspect,
            )

        model_input: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": _ASPECT_MAP.get(aspect, "16:9"),
        }
        if kwargs.get("seed") is not None:
            model_input["seed"] = kwargs["seed"]

        try:
            output = run_model(meta["ref"], model_input, timeout=300)
        except ReplicateError as exc:
            logger.warning("Replicate image gen failed (%s): %s", model_id, exc)
            return error_response(
                error=str(exc),
                error_type="api_error",
                provider="replicate",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        url = first_url(output)
        if not url:
            return error_response(
                error="Replicate returned no image URL",
                error_type="empty_response",
                provider="replicate",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            image_ref = str(save_url_image(url, prefix=f"replicate_{model_id}"))
        except Exception as exc:  # noqa: BLE001 — fall back to the bare URL
            logger.warning("Could not cache Replicate image %s (%s); using URL.", url, exc)
            image_ref = url

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="replicate",
            extra={"replicate_ref": meta["ref"], "source_url": url},
        )


def register(ctx: Any) -> None:
    """Plugin entry point — wire ``ReplicateImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(ReplicateImageGenProvider())
