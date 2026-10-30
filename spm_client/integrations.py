"""Drop-in integration wrappers (bd SPM-Polaris-814s).

``patch_openai`` / ``patch_anthropic`` point an official SDK client at the SPM
gateway so the client gains ambient capture + recall injection with zero config
-- the same seamless drop-in pattern Mem0's ``mem0.proxy`` proved in the
market. The gateway is multi-protocol (``/v1/chat/completions``,
``/v1/responses``, ``/v1/messages``), so both OpenAI and Anthropic clients stay
shape-compatible with a base_url swap; the gateway handles memory on its own.

``auto()`` reads ``SPM_API_KEY`` / ``SPM_GATEWAY_URL`` and returns a new
``openai.OpenAI`` client already pointed at SPM (one line to start).
"""
from __future__ import annotations

import os
from typing import Any


def _sync_client_transport(client: Any, *, base_url: str, api_key: str) -> None:
    """Update the official SDK's already-built httpx transport in place.

    Both official clients copy the constructor values into a private sync
    transport. Updating only the public ``base_url``/``api_key`` attributes is
    insufficient: requests would continue going to the original provider.
    """
    http_client = getattr(client, "_client", None)
    if http_client is None:
        return
    http_client.base_url = base_url
    headers = getattr(http_client, "headers", None)
    if headers is not None:
        if api_key:
            if client.__class__.__module__.startswith("anthropic"):
                headers["x-api-key"] = api_key
                headers.pop("authorization", None)
            else:
                headers["authorization"] = f"Bearer {api_key}"


def _openai_base_url(base_url: str) -> str:
    target = base_url.rstrip("/")
    return target if target.endswith("/v1") else target + "/v1"


def patch_openai(client: Any, *, api_key: str = "", base_url: str = "") -> Any:
    """Point an ``openai.OpenAI`` client at the SPM gateway (drop-in).

    OpenAI SDK posts to ``{base_url}/chat/completions``, so the gateway URL is
    ``.../v1``. Memory (ambient capture + recall injection) is handled by the
    gateway; the caller keeps the exact same request shape.
    """
    if base_url and not api_key:
        raise ValueError("api_key is required when patching an OpenAI client to SPM")
    if api_key:
        client.api_key = api_key
    if base_url:
        target = _openai_base_url(base_url)
        client.base_url = target
        _sync_client_transport(client, base_url=target, api_key=api_key)
    elif api_key:
        _sync_client_transport(client, base_url=getattr(client, "base_url", ""), api_key=api_key)
    return client


def patch_anthropic(client: Any, *, api_key: str = "", base_url: str = "") -> Any:
    """Point an ``anthropic.Anthropic`` client at the SPM gateway (drop-in).

    Anthropic SDK posts to ``{base_url}/v1/messages``, so the gateway URL is
    the host root. The client stays shape-compatible; the gateway serves the
    messages protocol and handles memory.
    """
    if base_url and not api_key:
        raise ValueError("api_key is required when patching an Anthropic client to SPM")
    if api_key:
        client.api_key = api_key
    if base_url:
        target = base_url.rstrip("/")
        client.base_url = target
        _sync_client_transport(client, base_url=target, api_key=api_key)
    elif api_key:
        _sync_client_transport(client, base_url=getattr(client, "base_url", ""), api_key=api_key)
    return client


def auto(*, api_key: str = "", base_url: str = "") -> Any:
    """One-line drop-in: return an ``openai.OpenAI`` client on the SPM gateway.

    Falls back to ``SPM_API_KEY`` / ``SPM_GATEWAY_URL`` env vars.
    """
    key = api_key or os.environ.get("SPM_API_KEY", "").strip()
    url = base_url or os.environ.get("SPM_GATEWAY_URL", "").strip()
    if not key or not url:
        raise ValueError("SPM_API_KEY and SPM_GATEWAY_URL are required (or pass api_key/base_url)")
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The OpenAI SDK is required for auto(); install it with 'pip install openai'."
        ) from exc
    client = openai.OpenAI(api_key=key, base_url=_openai_base_url(url))
    return client
