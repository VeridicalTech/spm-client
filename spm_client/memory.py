"""Client for the hosted SPM memory plane (MCP tools over Streamable HTTP).

The gateway path needs no client: point an OpenAI- or Anthropic-compatible SDK at
/v1 and chat works. What it does not give you is the memory tools themselves, and
that is what this covers - remember / recall / read / delete / status against
https://api.spmos.ai/mcp, with the evidence handles returned by recall resolvable
through read.

The transport shape follows the adapter that already talks to this server
(integrations/aml/adapter.py): one JSON-RPC tools/call per request, structured
output preferred over the text payload, and failures mapped to categories a
caller can branch on rather than opaque exceptions. It is written fresh instead
of imported because the adapter is coupled to evaluation configuration and lives
in a frozen submission lineage.

Failure policy: nothing is swallowed. Every transport, HTTP or tool failure
raises SPMError with a stable category, and retryable ones carry retry_after.
Retries are bounded and only apply to backpressure and transport faults; an auth
rejection is never retried.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import httpx

DEFAULT_MCP_URL = "https://api.spmos.ai/mcp"
DEFAULT_GATEWAY_URL = "https://api.spmos.ai"
RETRYABLE_CATEGORIES = frozenset({"backpressure", "unavailable", "timeout"})


class SPMError(RuntimeError):
    """One client-visible failure, with a category a caller can branch on.

    Categories: auth, backpressure, unavailable, timeout, invalid_input,
    not_found, failed.
    """

    def __init__(
        self,
        category: str,
        detail: str,
        *,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail
        self.retry_after = retry_after
        self.status_code = status_code


@dataclass(frozen=True)
class SPMConfig:
    api_key: str
    mcp_url: str = DEFAULT_MCP_URL
    gateway_url: str = DEFAULT_GATEWAY_URL
    timeout_seconds: float = 150.0
    user_agent: str = "SPM-Python-SDK/0.1"
    max_attempts: int = 3
    backoff_seconds: float = 0.5
    backoff_cap_seconds: float = 8.0
    user_partition: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("SPMConfig.api_key is required")
        if not self.mcp_url.startswith(("http://", "https://")):
            raise ValueError("SPMConfig.mcp_url must be an http(s) URL")
        if not self.gateway_url.startswith(("http://", "https://")):
            raise ValueError("SPMConfig.gateway_url must be an http(s) URL")
        if self.max_attempts < 1:
            raise ValueError("SPMConfig.max_attempts must be at least 1")


@dataclass(frozen=True)
class EvidenceRef:
    """One recall evidence item, with the handle that resolves it to source text."""

    source_id: str
    read_token: str
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallResult:
    answer: str
    status: str
    evidence: tuple[EvidenceRef, ...]
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.status == "supported"


@dataclass(frozen=True)
class RequestReceipt:
    """One gateway request receipt: what each stage did for a single request.

    Mirrors RequestReceiptView (integrations/gateway/runtime.py). Only the fields
    a caller branches on are typed; everything else stays on raw.
    """

    receipt_id: str
    request_id: str
    protocol: str
    model: str
    compression_mode: str
    memory_mode: str
    request_state: str
    billing_state: str
    memory_state: str
    bytes_sent: int
    provider_usage_state: str
    error_code: str
    created_at: str
    updated_at: str
    terminal_at: str
    raw: Mapping[str, Any] = field(default_factory=dict)


def _to_receipt(payload: Mapping[str, Any]) -> RequestReceipt:
    def value(name: str) -> str:
        raw = payload.get(name)
        return "" if raw is None else str(raw)

    return RequestReceipt(
        receipt_id=value("receipt_id"),
        request_id=value("request_id"),
        protocol=value("protocol"),
        model=value("model"),
        compression_mode=value("compression_mode"),
        memory_mode=value("memory_mode"),
        request_state=value("request_state"),
        billing_state=value("billing_state"),
        memory_state=value("memory_state"),
        bytes_sent=int(payload.get("bytes_sent") or 0),
        provider_usage_state=value("provider_usage_state"),
        error_code=value("error_code"),
        created_at=value("created_at"),
        updated_at=value("updated_at"),
        terminal_at=value("terminal_at"),
        raw=dict(payload),
    )


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _tool_error_category(text: str) -> str:
    lowered = (text or "").lower()
    for needle, category in (
        ("auth", "auth"),
        ("permission", "auth"),
        ("scope", "auth"),
        ("rate", "backpressure"),
        ("validation", "invalid_input"),
        ("invalid", "invalid_input"),
    ):
        if needle in lowered:
            return category
    return "failed"


def decode_response(response: Any, tool: str) -> dict[str, Any]:
    """Turn one HTTP response into a tool payload, or raise a categorised SPMError.

    Shared by both clients so the error taxonomy exists in exactly one place.
    """
    status = response.status_code
    if status == 429:
        raise SPMError(
            "backpressure",
            f"{tool}: rate limited",
            retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            status_code=status,
        )
    if status == 404:
        raise SPMError("not_found", f"{tool}: not found", status_code=status)
    if status in (401, 403):
        raise SPMError("auth", f"{tool}: credential rejected (HTTP {status})", status_code=status)
    if status >= 500:
        raise SPMError("unavailable", f"{tool}: upstream HTTP {status}", status_code=status)
    if status != 200:
        raise SPMError("failed", f"{tool}: unexpected HTTP {status}", status_code=status)

    try:
        envelope = response.json()
        result = envelope["result"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SPMError("failed", f"{tool}: malformed JSON-RPC envelope") from exc

    text_payload = ""
    if isinstance(result, dict):
        for item in result.get("content", []) or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text_payload = str(item.get("text", ""))
                break
        if result.get("isError"):
            raise SPMError(_tool_error_category(text_payload), f"{tool}: {text_payload[:200]}")
        structured = result.get("structuredContent")
        if isinstance(structured, dict) and structured:
            return dict(structured)
    if text_payload:
        try:
            parsed = json.loads(text_payload)
        except ValueError as exc:
            raise SPMError("failed", f"{tool}: non-JSON tool payload") from exc
        if isinstance(parsed, dict):
            return parsed
    return {}


def _to_recall(payload: Mapping[str, Any]) -> RecallResult:
    refs: list[EvidenceRef] = []
    for item in payload.get("evidence_refs", []) or []:
        if isinstance(item, Mapping):
            refs.append(
                EvidenceRef(
                    source_id=str(item.get("source_id", "")),
                    read_token=str(item.get("read_token", "")),
                    raw=dict(item),
                )
            )
    decision = payload.get("evidence_decision") or {}
    status = str(decision.get("status") or payload.get("status") or "unknown")
    return RecallResult(
        answer=str(payload.get("answer", "")),
        status=status,
        evidence=tuple(refs),
        raw=dict(payload),
    )


def decode_json_response(response: Any, what: str) -> dict[str, Any]:
    """Same taxonomy as decode_response, for the gateway's plain-JSON endpoints.

    The gateway answers with JSON bodies rather than JSON-RPC envelopes, so the
    status mapping is shared but the unwrapping is not.
    """
    status = response.status_code
    if status == 429:
        raise SPMError(
            "backpressure",
            f"{what}: rate limited",
            retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            status_code=status,
        )
    if status == 404:
        raise SPMError("not_found", f"{what}: not found", status_code=status)
    if status in (401, 403):
        raise SPMError("auth", f"{what}: credential rejected (HTTP {status})", status_code=status)
    if status >= 500:
        raise SPMError("unavailable", f"{what}: upstream HTTP {status}", status_code=status)
    if status != 200:
        raise SPMError("failed", f"{what}: unexpected HTTP {status}", status_code=status)
    try:
        payload = response.json()
    except ValueError as exc:
        raise SPMError("failed", f"{what}: non-JSON response body") from exc
    if not isinstance(payload, dict):
        raise SPMError("failed", f"{what}: unexpected response shape")
    return payload


def _sleep_for(attempt: int, config: SPMConfig, retry_after: float | None) -> None:
    if retry_after is not None:
        delay = min(retry_after, config.backoff_cap_seconds)
    else:
        delay = min(config.backoff_seconds * (2**attempt), config.backoff_cap_seconds)
    time.sleep(delay * (0.5 + random.random() / 2))


class AsyncSPMMemoryClient:
    """Async client. Use as a context manager so the connection pool is closed."""

    def __init__(
        self,
        config: SPMConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Any = None,
    ) -> None:
        self._config = config
        self._sleep = sleep or _sleep_for
        self._counter = 0
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Accept": "application/json, text/event-stream",
                "User-Agent": config.user_agent,
            },
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncSPMMemoryClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def call(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        last: SPMError | None = None
        for attempt in range(self._config.max_attempts):
            self._counter += 1
            try:
                response = await self._client.post(
                    self._config.mcp_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": self._counter,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": dict(arguments)},
                    },
                )
            except httpx.TimeoutException as exc:
                last = SPMError("timeout", f"{tool}: request timed out")
                last.__cause__ = exc
            except httpx.HTTPError as exc:
                last = SPMError("unavailable", f"{tool}: transport failure {type(exc).__name__}")
                last.__cause__ = exc
            else:
                try:
                    return decode_response(response, tool)
                except SPMError as error:
                    if error.category not in RETRYABLE_CATEGORIES:
                        raise
                    last = error
            if attempt + 1 < self._config.max_attempts:
                await self._sleep(attempt, self._config, last.retry_after if last else None)
        assert last is not None
        raise last

    async def remember(
        self,
        *,
        text: str,
        idempotency_key: str,
        source_id: str = "",
        topic: str = "general",
        source_timestamp: str = "",
        user_partition: str | None = None,
    ) -> dict[str, Any]:
        return await self.call(
            "remember",
            {
                "text": text,
                "idempotency_key": idempotency_key,
                "source_id": source_id,
                "topic": topic,
                "source_timestamp": source_timestamp,
                "user_partition": self._partition(user_partition),
            },
        )

    async def recall(
        self,
        question: str,
        *,
        top_k: int = 20,
        depth: str | None = None,
        lane_policy: str | None = None,
        user_partition: str | None = None,
    ) -> RecallResult:
        arguments: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "user_partition": self._partition(user_partition),
        }
        if depth:
            arguments["depth"] = depth
        if lane_policy:
            arguments["lane_policy"] = lane_policy
        return _to_recall(await self.call("recall", arguments))

    async def read(self, read_tokens: Sequence[str], *, user_partition: str | None = None) -> dict[str, Any]:
        """Resolve recall evidence handles back to source text.

        The server parameter is an array even for one token; expired or altered
        tokens come back under not_found rather than raising.
        """
        return await self.call(
            "read",
            {"read_tokens": list(read_tokens), "user_partition": self._partition(user_partition)},
        )

    async def delete(self, *, source_id: str, idempotency_key: str) -> dict[str, Any]:
        return await self.call("delete", {"source_id": source_id, "idempotency_key": idempotency_key})

    async def status(self, *, source_ids: Sequence[str] = ()) -> dict[str, Any]:
        return await self.call("status", {"source_ids": list(source_ids)})

    async def get_receipt(self, receipt_id: str) -> RequestReceipt:
        """Fetch the gateway's receipt for one request (needs the receipt:read scope)."""
        return _to_receipt(
            await self._gateway_get(f"/v1/spm/requests/{receipt_id}", f"receipt {receipt_id}")
        )

    async def list_receipts(self, *, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        query = f"?limit={int(limit)}" + (f"&cursor={cursor}" if cursor else "")
        return await self._gateway_get(f"/v1/spm/requests{query}", "receipt list")

    async def _gateway_get(self, path: str, what: str) -> dict[str, Any]:
        last: SPMError | None = None
        for attempt in range(self._config.max_attempts):
            try:
                response = await self._client.get(f"{self._config.gateway_url.rstrip('/')}{path}")
            except httpx.TimeoutException as exc:
                last = SPMError("timeout", f"{what}: request timed out")
                last.__cause__ = exc
            except httpx.HTTPError as exc:
                last = SPMError("unavailable", f"{what}: transport failure {type(exc).__name__}")
                last.__cause__ = exc
            else:
                try:
                    return decode_json_response(response, what)
                except SPMError as error:
                    if error.category not in RETRYABLE_CATEGORIES:
                        raise
                    last = error
            if attempt + 1 < self._config.max_attempts:
                await self._sleep(attempt, self._config, last.retry_after if last else None)
        assert last is not None
        raise last

    def _partition(self, override: str | None) -> str:
        return self._config.user_partition if override is None else override


class SPMMemoryClient:
    """Blocking client with the same surface and the same error taxonomy."""

    def __init__(
        self,
        config: SPMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = None,
    ) -> None:
        self._config = config
        self._sleep = sleep or _sleep_for
        self._counter = 0
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Accept": "application/json, text/event-stream",
                "User-Agent": config.user_agent,
            },
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SPMMemoryClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def call(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        last: SPMError | None = None
        for attempt in range(self._config.max_attempts):
            self._counter += 1
            try:
                response = self._client.post(
                    self._config.mcp_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": self._counter,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": dict(arguments)},
                    },
                )
            except httpx.TimeoutException as exc:
                last = SPMError("timeout", f"{tool}: request timed out")
                last.__cause__ = exc
            except httpx.HTTPError as exc:
                last = SPMError("unavailable", f"{tool}: transport failure {type(exc).__name__}")
                last.__cause__ = exc
            else:
                try:
                    return decode_response(response, tool)
                except SPMError as error:
                    if error.category not in RETRYABLE_CATEGORIES:
                        raise
                    last = error
            if attempt + 1 < self._config.max_attempts:
                self._sleep(attempt, self._config, last.retry_after if last else None)
        assert last is not None
        raise last

    # The blocking surface mirrors the async one signature for signature, so a
    # caller can switch between them without re-reading the arguments. Passing
    # **kwargs straight through let the two drift (status() sent {} here and
    # {"source_ids": []} there), which a test caught.
    def remember(
        self,
        *,
        text: str,
        idempotency_key: str,
        source_id: str = "",
        topic: str = "general",
        source_timestamp: str = "",
        user_partition: str | None = None,
    ) -> dict[str, Any]:
        return self.call(
            "remember",
            {
                "text": text,
                "idempotency_key": idempotency_key,
                "source_id": source_id,
                "topic": topic,
                "source_timestamp": source_timestamp,
                "user_partition": self._partition(user_partition),
            },
        )

    def recall(
        self,
        question: str,
        *,
        top_k: int = 20,
        depth: str | None = None,
        lane_policy: str | None = None,
        user_partition: str | None = None,
    ) -> RecallResult:
        arguments: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "user_partition": self._partition(user_partition),
        }
        if depth:
            arguments["depth"] = depth
        if lane_policy:
            arguments["lane_policy"] = lane_policy
        return _to_recall(self.call("recall", arguments))

    def read(self, read_tokens: Sequence[str], *, user_partition: str | None = None) -> dict[str, Any]:
        return self.call(
            "read",
            {"read_tokens": list(read_tokens), "user_partition": self._partition(user_partition)},
        )

    def delete(self, *, source_id: str, idempotency_key: str) -> dict[str, Any]:
        return self.call("delete", {"source_id": source_id, "idempotency_key": idempotency_key})

    def status(self, *, source_ids: Sequence[str] = ()) -> dict[str, Any]:
        return self.call("status", {"source_ids": list(source_ids)})

    def get_receipt(self, receipt_id: str) -> RequestReceipt:
        """Fetch the gateway's receipt for one request (needs the receipt:read scope)."""
        return _to_receipt(self._gateway_get(f"/v1/spm/requests/{receipt_id}", f"receipt {receipt_id}"))

    def list_receipts(self, *, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        query = f"?limit={int(limit)}" + (f"&cursor={cursor}" if cursor else "")
        return self._gateway_get(f"/v1/spm/requests{query}", "receipt list")

    def _gateway_get(self, path: str, what: str) -> dict[str, Any]:
        last: SPMError | None = None
        for attempt in range(self._config.max_attempts):
            try:
                response = self._client.get(f"{self._config.gateway_url.rstrip('/')}{path}")
            except httpx.TimeoutException as exc:
                last = SPMError("timeout", f"{what}: request timed out")
                last.__cause__ = exc
            except httpx.HTTPError as exc:
                last = SPMError("unavailable", f"{what}: transport failure {type(exc).__name__}")
                last.__cause__ = exc
            else:
                try:
                    return decode_json_response(response, what)
                except SPMError as error:
                    if error.category not in RETRYABLE_CATEGORIES:
                        raise
                    last = error
            if attempt + 1 < self._config.max_attempts:
                self._sleep(attempt, self._config, last.retry_after if last else None)
        assert last is not None
        raise last

    def _partition(self, override: str | None) -> str:
        return self._config.user_partition if override is None else override
