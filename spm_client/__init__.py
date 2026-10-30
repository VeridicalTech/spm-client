"""SPM client: the hosted memory plane, the gateway, and offline verification.

Two surfaces, one distribution:

* the **memory plane** (:mod:`spm_client.memory`) - remember / recall / read /
  delete / status against `https://api.spmos.ai/mcp`, plus request receipts from
  the gateway. Its base error is `SPMError`.
* the **gateway** (:class:`SpmClient`, :class:`ChatSession`) - an
  OpenAI-compatible chat client with transparent session state. Its base error is
  `SpmError` (lower-case "pm").

`SPMError` and `SpmError` are unrelated classes; they are kept under their
original names so neither surface changes shape, and they are named here so the
distinction is hard to miss rather than a typo waiting to happen.

:mod:`spm_client.evidence` verifies an evidence handle offline, from the
published public key, with no call back to the server that minted it.

Install (local dev):
    pip install -e sdk/python

PyPI publish blocked by LEGAL_RELEASE_BLOCKER until legal sign-off.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Iterator, Mapping

import httpx

logger = logging.getLogger(__name__)


__version__ = "0.2.1"
__all__ = [
    "SpmClient",
    "ChatSession",
    "SpmError",
    "AuthError",
    "QuotaError",
    "NotFoundError",
    "ServerError",
    # memory plane
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MCP_URL",
    "AsyncSPMMemoryClient",
    "EvidenceRef",
    "RecallResult",
    "RequestReceipt",
    "SPMConfig",
    "SPMError",
    "SPMMemoryClient",
    # offline evidence verification
    "EvidenceVerificationError",
    "VerifiedEvidence",
    "canonical_payload_bytes",
    "verify_evidence_token",
    # drop-in integration wrappers (bd SPM-Polaris-814s)
    "patch_openai",
    "patch_anthropic",
    "auto",
]

from .integrations import auto, patch_anthropic, patch_openai

from .evidence import (
    EvidenceVerificationError,
    VerifiedEvidence,
    canonical_payload_bytes,
    verify_evidence_token,
)
from .memory import (
    DEFAULT_GATEWAY_URL,
    DEFAULT_MCP_URL,
    AsyncSPMMemoryClient,
    EvidenceRef,
    RecallResult,
    RequestReceipt,
    SPMConfig,
    SPMError,
    SPMMemoryClient,
)


# ---------------------------------------------------------------------------
# Typed error hierarchy
# ---------------------------------------------------------------------------


class SpmError(Exception):
    """Base exception for all SPM client errors."""

    def __init__(self, message: str, *, status_code: int | None = None, raw: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw


class AuthError(SpmError):
    """Raised on 401/403 responses (invalid or missing API key)."""


class QuotaError(SpmError):
    """Raised on 429 responses (rate limit or quota exceeded)."""


class NotFoundError(SpmError):
    """Raised on 404 responses."""


class ServerError(SpmError):
    """Raised on 5xx responses."""


def _format_error_message(resp: httpx.Response, base_msg: str) -> str:
    """Augment a raw HTTP error with cause + fix guidance."""
    status = resp.status_code
    if status in {401, 403}:
        return (
            f"{base_msg}\n"
            "Invalid or expired API key. Check SPM_API_KEY. "
            "Rotate at [dashboard]/api-keys."
        )
    if status == 429:
        retry_after_raw = resp.headers.get("Retry-After", "")
        try:
            retry_after = int(retry_after_raw)
            retry_msg = f"{retry_after}s"
        except (ValueError, TypeError):
            retry_msg = "unknown"
        return (
            f"{base_msg}\n"
            f"Rate limit exceeded. Retry after {retry_msg}. "
            "See docs/rate-limiting."
        )
    if status >= 500:
        corr_id = (
            resp.headers.get("X-Correlation-Id")
            or resp.headers.get("X-Request-Id", "unknown")
        )
        return (
            f"{base_msg}\n"
            f"Server error ({status}). Check service health at /health. "
            f"Correlation ID: {corr_id}"
        )
    return base_msg


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        body = resp.json()
        msg = body.get("error", {}).get("message", resp.text) if isinstance(body, dict) else resp.text
    except Exception:
        msg = resp.text or f"HTTP {resp.status_code}"
    msg = _format_error_message(resp, msg)
    if resp.status_code in {401, 403}:
        raise AuthError(msg, status_code=resp.status_code, raw=resp.text)
    if resp.status_code == 429:
        raise QuotaError(msg, status_code=resp.status_code, raw=resp.text)
    if resp.status_code == 404:
        raise NotFoundError(msg, status_code=resp.status_code, raw=resp.text)
    if resp.status_code >= 500:
        raise ServerError(msg, status_code=resp.status_code, raw=resp.text)
    raise SpmError(msg, status_code=resp.status_code, raw=resp.text)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SpmClient:
    """Low-level HTTP client for the SPM-Polaris Gateway.

    Args:
        api_key: Bearer key issued by the SPM-Polaris console.
        base_url: Gateway base URL, e.g. ``https://api.example.com``.
        timeout: Per-request timeout in seconds (default 120s for streaming).
    """

    @classmethod
    def from_env(cls) -> "SpmClient":
        """Create a client from environment variables.

        Reads ``SPM_API_KEY`` and ``SPM_GATEWAY_URL``.  Raises :class:`ValueError`
        with an actionable message if any required variable is missing or empty.
        """
        api_key = os.environ.get("SPM_API_KEY", "").strip()
        base_url = os.environ.get("SPM_GATEWAY_URL", "").strip()

        missing: list[str] = []
        if not api_key:
            missing.append("  SPM_API_KEY — your API key from the SPM dashboard")
        if not base_url:
            missing.append("  SPM_GATEWAY_URL — e.g. http://localhost:18081 for local dev")

        if missing:
            raise ValueError(
                "Cannot create SpmClient from environment. Missing variables:\n"
                + "\n".join(missing)
                + "\n\nSet these variables or pass api_key and base_url directly to SpmClient()."
            )

        return cls(api_key=api_key, base_url=base_url)

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "http://localhost:18081",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SpmClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        """Return the gateway health status dict."""
        resp = self._http.get("/health")
        _raise_for_status(resp)
        return resp.json()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "gpt-4o-mini",
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[str]:
        """Send a chat-completion request.

        When ``stream=True`` returns an iterator of raw SSE data lines.
        When ``stream=False`` returns the decoded response dict.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream, **kwargs}
        if stream:
            return self._stream_chat(payload)
        resp = self._http.post("/v1/chat/completions", json=payload)
        _raise_for_status(resp)
        return resp.json()

    def _stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        with self._http.stream("POST", "/v1/chat/completions", json=payload) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield data

    def list_receipts(self, *, limit: int = 20, cursor: str | None = None) -> dict[str, Any]:
        """List recent request receipts for audit and debugging.

        Args:
            limit: Maximum number of receipts to return.
            cursor: Opaque pagination cursor from a previous response (cursor-based
                pagination — the gateway does not accept an ``offset`` parameter).
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        resp = self._http.get("/v1/spm/requests", params=params)
        _raise_for_status(resp)
        return resp.json()

    def get_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Fetch a single request receipt by ID."""
        resp = self._http.get(f"/v1/spm/requests/{receipt_id}")
        _raise_for_status(resp)
        return resp.json()

    def session(
        self,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
    ) -> "ChatSession":
        """Create a stateful chat session backed by this client.

        session_id is forwarded as X-Session-Id; system_prompt is prepended to
        every request. ChatSession already supported both - the factory did not
        expose them, so a documented capability was unreachable from the one
        entry point users actually call.
        """
        return ChatSession(client=self, session_id=session_id, system_prompt=system_prompt)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class ChatSession:
    """Stateful chat session with transparent message history management.

    Maintains the full conversation history so callers only need to pass
    the latest user message. Memory is managed server-side; the session
    only tracks the local message list for context formatting.

    Args:
        client: An :class:`SpmClient` instance.
        session_id: Opaque identifier forwarded as ``X-Session-Id``. A UUID
            is generated if not provided.
        system_prompt: Optional system message prepended to every request.
    """

    def __init__(
        self,
        client: SpmClient,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self.session_id = session_id or str(uuid.uuid4())
        self._history: list[dict[str, str]] = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    def send(self, content: str, *, model: str = "gpt-4o-mini", stream: bool = False, **kwargs: Any) -> str | Iterator[str]:
        """Send a user message and return the assistant reply.

        The history is updated automatically.  When ``stream=True``, returns
        an iterator of text chunks and the history is updated lazily on
        exhaustion.
        """
        if stream:
            # Append user message INSIDE the generator so a discarded iterator
            # (caller breaks early) leaves history in a consistent state.
            return self._stream_reply(content, model=model, **kwargs)
        pending_user = {"role": "user", "content": content}
        self._history.append(pending_user)
        try:
            result = self._client.chat(self._history, model=model, stream=False, **kwargs)
            # Guard for tool-call responses where content may be None.
            message = result.get("choices", [{}])[0].get("message", {})
            reply: str = message.get("content") or ""
            self._history.append({"role": "assistant", "content": reply})
            return reply
        except BaseException:
            if self._history and self._history[-1] is pending_user:
                self._history.pop()
            raise

    def _stream_reply(self, content: str, **kwargs: Any) -> Iterator[str]:
        # Append user message only after the generator starts executing so that
        # abandoning the iterator before first iteration is harmless.
        self._history.append({"role": "user", "content": content})
        chunks: list[str] = []
        try:
            for raw in self._client.chat(self._history, stream=True, **kwargs):
                try:
                    data = json.loads(raw)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        chunks.append(delta)
                        yield delta
                except Exception:
                    logger.debug("stream parse error", exc_info=True)
                    continue
        finally:
            # Runs on normal exhaustion, early break (GeneratorExit), or
            # any exception. If chunks is empty the assistant never replied
            # (or was interrupted) — pop the user message so history stays
            # in an alternating user/assistant pattern.
            if chunks:
                self._history.append({"role": "assistant", "content": "".join(chunks)})
            elif self._history and self._history[-1] == {"role": "user", "content": content}:
                self._history.pop()

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    def reset(self, *, keep_system: bool = True) -> None:
        """Clear the conversation history.

        Args:
            keep_system: If True, retain the initial system message (default).
        """
        if keep_system and self._history and self._history[0]["role"] == "system":
            self._history = [self._history[0]]
        else:
            self._history = []
