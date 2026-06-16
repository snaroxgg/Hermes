"""Shared helpers for the Replicate-backed image/video generation plugins.

Replicate exposes a single uniform REST surface: ``POST /v1/models/{owner}/{name}/predictions``
runs a model's latest version. With the ``Prefer: wait`` header the call blocks
server-side until the prediction finishes (or ~60s elapse), at which point we
fall back to polling ``urls.get`` until the prediction reaches a terminal state.

Both the ``replicate`` image-gen and video-gen plugins delegate here so token
resolution and the run/poll loop live in exactly one place (mirrors the role
``tools/fal_common.py`` plays for the FAL plugins).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

REPLICATE_API_BASE = "https://api.replicate.com/v1"

# Replicate's own CLI/SDK reads REPLICATE_API_TOKEN; accept REPLICATE_API_KEY
# too since several hosted-model dashboards label it that way.
_TOKEN_ENV_VARS = ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY")


class ReplicateError(RuntimeError):
    """Raised for any Replicate API / prediction failure."""


def resolve_replicate_token() -> str:
    """Return the configured Replicate API token, or '' when unset.

    Checks ``os.environ`` first (populated by ``reload_env()`` at startup),
    then falls back to reading ``$HERMES_HOME/.env`` directly via Hermes'
    config helper — so the token resolves even in contexts where the .env
    hasn't yet been hydrated into the process environment.
    """
    for var in _TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    try:
        from hermes_cli.config import get_env_value

        for var in _TOKEN_ENV_VARS:
            val = get_env_value(var)
            if val and val.strip():
                return val.strip()
    except Exception:  # noqa: BLE001 — config helper optional; never hard-fail
        pass
    return ""


def replicate_available() -> bool:
    """True when a Replicate API token is configured."""
    return bool(resolve_replicate_token())


def run_model(
    model_ref: str,
    model_input: Dict[str, Any],
    *,
    token: Optional[str] = None,
    timeout: float = 600.0,
    poll_interval: float = 3.0,
) -> Any:
    """Run ``model_ref`` (``owner/name``) with ``model_input`` and return its output.

    Uses ``Prefer: wait`` for a fast synchronous path, then polls the prediction
    until it reaches a terminal status or ``timeout`` seconds elapse. Raises
    :class:`ReplicateError` on any non-success outcome.
    """
    token = token or resolve_replicate_token()
    if not token:
        raise ReplicateError(
            "REPLICATE_API_TOKEN is not set. Add it to your environment / "
            "$HERMES_HOME/.env (get one at https://replicate.com/account/api-tokens)."
        )

    create_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        # Block server-side until the prediction finishes (or ~60s pass).
        "Prefer": "wait",
    }
    poll_headers = {"Authorization": f"Bearer {token}"}

    url = f"{REPLICATE_API_BASE}/models/{model_ref}/predictions"
    try:
        resp = requests.post(url, headers=create_headers, json={"input": model_input}, timeout=90)
    except requests.RequestException as exc:
        raise ReplicateError(f"Replicate request error: {exc}") from exc

    if resp.status_code >= 400:
        raise ReplicateError(
            f"Replicate rejected the request for '{model_ref}' "
            f"(HTTP {resp.status_code}): {resp.text[:400]}"
        )

    try:
        prediction = resp.json()
    except ValueError as exc:
        raise ReplicateError(f"Replicate returned invalid JSON: {exc}") from exc

    deadline = time.monotonic() + timeout
    get_url = (prediction.get("urls") or {}).get("get")

    while prediction.get("status") not in ("succeeded", "failed", "canceled"):
        if time.monotonic() > deadline:
            raise ReplicateError(
                f"Replicate prediction for '{model_ref}' timed out after {int(timeout)}s"
            )
        if not get_url:
            raise ReplicateError("Replicate response did not include a polling URL")
        time.sleep(poll_interval)
        try:
            r = requests.get(get_url, headers=poll_headers, timeout=30)
        except requests.RequestException as exc:
            raise ReplicateError(f"Replicate poll error: {exc}") from exc
        if r.status_code >= 400:
            raise ReplicateError(
                f"Replicate poll failed (HTTP {r.status_code}): {r.text[:300]}"
            )
        prediction = r.json()

    status = prediction.get("status")
    if status != "succeeded":
        detail = prediction.get("error") or status
        raise ReplicateError(f"Replicate prediction {status}: {detail}")

    return prediction.get("output")


def first_url(output: Any) -> Optional[str]:
    """Extract a single asset URL from Replicate's polymorphic ``output`` field.

    Replicate models return a bare URL string, a list of URL strings, or a dict
    with a ``url``/``video``/``image`` key (FileOutput). Returns the first URL
    found, or None.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output or None
    if isinstance(output, list):
        for item in output:
            found = first_url(item)
            if found:
                return found
        return None
    if isinstance(output, dict):
        for key in ("url", "video", "image", "output"):
            val = output.get(key)
            if isinstance(val, str) and val:
                return val
    return None
