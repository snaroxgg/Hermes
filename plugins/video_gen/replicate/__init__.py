"""Replicate video generation backend.

Curated catalog of Replicate's top video models exposed as a
:class:`VideoGenProvider`. The default — ``veo-3.1`` (Google Veo 3.1, native
audio, highest fidelity) — is the single "best" video model requested at setup.

One unified surface covers text-to-video and image-to-video: when ``image_url``
is supplied the payload routes to the family's image-input field (which differs
per model — Veo/Seedance use ``image``, Kling uses ``start_image``, Hailuo uses
``first_frame_image``); otherwise it's a plain text-to-video run.

Selection precedence (first hit wins):
1. ``model=`` arg from the tool call
2. ``REPLICATE_VIDEO_MODEL`` env var
3. ``video_gen.replicate.model`` in ``config.yaml``
4. ``video_gen.model`` in ``config.yaml`` (when it names one of our families)
5. :data:`DEFAULT_MODEL`
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Family catalog
# ---------------------------------------------------------------------------
# Per-family capability sheet drives which keys land in the request payload —
# keys a family doesn't advertise are dropped before send.
#
#   ref            : Replicate owner/name slug
#   image_key      : input field name for image-to-video (None = no i2v)
#   aspect_ratios  : tuple of accepted ratios (None = endpoint decides)
#   resolutions    : tuple of accepted resolutions (None = endpoint decides)
#   durations      : tuple of accepted durations OR (min, max) range
#   audio          : True if generate_audio is supported
#   negative       : True if negative_prompt is supported
#   seed           : True if seed is supported

_FAMILIES: Dict[str, Dict[str, Any]] = {
    "veo-3.1": {
        "ref": "google/veo-3.1",
        "display": "Veo 3.1 (Google)",
        "speed": "~60-180s",
        "price": "premium",
        "strengths": "Google flagship. Cinematic, native audio, strong prompt adherence. Best overall.",
        "tier": "premium",
        "image_key": "image",
        "aspect_ratios": ("16:9", "9:16"),
        "resolutions": ("720p", "1080p"),
        "durations": (4, 6, 8),
        "audio": True,
        "negative": True,
        "seed": True,
    },
    "kling-v2.5": {
        "ref": "kwaivgi/kling-v2.5-turbo-pro",
        "display": "Kling v2.5 Turbo Pro",
        "speed": "~60-180s",
        "price": "premium",
        "strengths": "Pro-level motion + prompt adherence. 5s/10s clips.",
        "tier": "premium",
        "image_key": "start_image",
        "aspect_ratios": ("16:9", "9:16", "1:1"),
        "resolutions": None,
        "durations": (5, 10),
        "audio": False,
        "negative": True,
        "seed": False,
    },
    "seedance-1-pro": {
        "ref": "bytedance/seedance-1-pro",
        "display": "Seedance 1 Pro (ByteDance)",
        "speed": "~60-150s",
        "price": "premium",
        "strengths": "Text & image-to-video up to 1080p, flexible aspect ratios.",
        "tier": "premium",
        "image_key": "image",
        "aspect_ratios": ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"),
        "resolutions": ("480p", "720p", "1080p"),
        "durations": (3, 12),
        "audio": False,
        "negative": False,
        "seed": True,
    },
    "hailuo-2.3": {
        "ref": "minimax/hailuo-2.3",
        "display": "Hailuo 2.3 (MiniMax)",
        "speed": "~60-150s",
        "price": "premium",
        "strengths": "High-fidelity motion, expressive characters.",
        "tier": "premium",
        "image_key": "first_frame_image",
        "aspect_ratios": None,
        "resolutions": ("768p", "1080p"),
        "durations": (6, 10),
        "audio": False,
        "negative": False,
        "seed": False,
    },
}

DEFAULT_MODEL = "veo-3.1"


def _is_duration_range(durations: Any) -> bool:
    """A 2-tuple of ints with a gap > 1 is treated as ``(min, max)``."""
    return (
        isinstance(durations, tuple)
        and len(durations) == 2
        and all(isinstance(d, int) for d in durations)
        and durations[1] - durations[0] > 1
    )


def _clamp_duration(family: Dict[str, Any], duration: Optional[int]) -> Optional[int]:
    durations = family.get("durations")
    if not durations:
        return None
    if duration is None:
        return durations[-1] if not _is_duration_range(durations) else durations[1]
    if _is_duration_range(durations):
        lo, hi = durations
        return max(lo, min(hi, duration))
    if duration in durations:
        return duration
    return min(durations, key=lambda d: abs(d - duration))


# ---------------------------------------------------------------------------
# Config / family resolution
# ---------------------------------------------------------------------------


def _load_video_gen_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load video_gen config: %s", exc)
        return {}


def _resolve_family(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    cfg = _load_video_gen_section()
    rep_cfg = cfg.get("replicate") if isinstance(cfg.get("replicate"), dict) else {}
    candidates: List[Optional[str]] = [
        explicit,
        os.environ.get("REPLICATE_VIDEO_MODEL"),
        rep_cfg.get("model") if isinstance(rep_cfg, dict) else None,
        cfg.get("model") if isinstance(cfg.get("model"), str) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip() in _FAMILIES:
            fid = c.strip()
            return fid, _FAMILIES[fid]
    return DEFAULT_MODEL, _FAMILIES[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _build_payload(
    family: Dict[str, Any],
    *,
    prompt: str,
    image_url: Optional[str],
    duration: Optional[int],
    aspect_ratio: str,
    resolution: str,
    negative_prompt: Optional[str],
    audio: Optional[bool],
    seed: Optional[int],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": prompt}

    if image_url and family.get("image_key"):
        payload[family["image_key"]] = image_url

    if family.get("aspect_ratios") and aspect_ratio in family["aspect_ratios"]:
        payload["aspect_ratio"] = aspect_ratio

    if family.get("resolutions") and resolution in family["resolutions"]:
        payload["resolution"] = resolution

    clamped = _clamp_duration(family, duration)
    if clamped is not None:
        payload["duration"] = clamped

    if family.get("audio") and audio is not None:
        payload["generate_audio"] = bool(audio)

    if family.get("negative") and negative_prompt:
        payload["negative_prompt"] = negative_prompt

    if family.get("seed") and seed is not None:
        payload["seed"] = seed

    return payload


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ReplicateVideoGenProvider(VideoGenProvider):
    """Replicate multi-family video generation backend (text & image-to-video)."""

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
        out: List[Dict[str, Any]] = []
        for fid, meta in _FAMILIES.items():
            modalities = ["text"]
            if meta.get("image_key"):
                modalities.append("image")
            out.append({
                "id": fid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
                "tier": meta.get("tier", "premium"),
                "modalities": modalities,
            })
        return out

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Replicate",
            "badge": "paid",
            "tag": "Veo 3.1, Kling v2.5, Seedance 1 Pro, Hailuo 2.3 — text-to-video & image-to-video",
            "env_vars": [
                {
                    "key": "REPLICATE_API_TOKEN",
                    "prompt": "Replicate API token",
                    "url": "https://replicate.com/account/api-tokens",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
            "resolutions": ["480p", "720p", "768p", "1080p"],
            "max_duration": 12,
            "min_duration": 3,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from tools.replicate_common import ReplicateError, first_url, run_model

        prompt = (prompt or "").strip()
        family_id, family = _resolve_family(model)
        image_url_norm = (image_url or "").strip() or None
        modality_used = "image" if image_url_norm else "text"

        if not prompt:
            return error_response(
                error="prompt is required.",
                error_type="missing_prompt",
                provider="replicate",
                model=family_id,
                prompt=prompt,
            )

        if image_url_norm and not family.get("image_key"):
            return error_response(
                error=(
                    f"Replicate family '{family_id}' has no image-to-video support. "
                    f"Pick a different model via `hermes tools` → Video Generation."
                ),
                error_type="modality_unsupported",
                provider="replicate",
                model=family_id,
                prompt=prompt,
            )

        payload = _build_payload(
            family,
            prompt=prompt,
            image_url=image_url_norm,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            negative_prompt=negative_prompt,
            audio=audio,
            seed=seed,
        )

        try:
            output = run_model(family["ref"], payload, timeout=900)
        except ReplicateError as exc:
            logger.warning("Replicate video gen failed (%s): %s", family_id, exc)
            return error_response(
                error=str(exc),
                error_type="api_error",
                provider="replicate",
                model=family_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        url = first_url(output)
        if not url:
            return error_response(
                error="Replicate returned no video URL",
                error_type="empty_response",
                provider="replicate",
                model=family_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        return success_response(
            video=url,
            model=family_id,
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio if "aspect_ratio" in payload else "",
            duration=int(payload["duration"]) if "duration" in payload else 0,
            provider="replicate",
            extra={"replicate_ref": family["ref"]},
        )


def register(ctx: Any) -> None:
    """Plugin entry point — wire ``ReplicateVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(ReplicateVideoGenProvider())
