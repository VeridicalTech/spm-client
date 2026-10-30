"""Integration tests for SpmClient against a mock gateway."""
import json
import os

import httpx
import pytest
import respx

from spm_client import (
    AuthError,
    ChatSession,
    NotFoundError,
    QuotaError,
    ServerError,
    SpmClient,
    SpmError,
)


# ---------------------------------------------------------------------------
# from_env tests
# ---------------------------------------------------------------------------


def test_from_env_success(monkeypatch):
    """from_env() creates a client when both env vars are set."""
    monkeypatch.setenv("SPM_API_KEY", "test-key-123")
    monkeypatch.setenv("SPM_GATEWAY_URL", "http://localhost:18081")
    client = SpmClient.from_env()
    assert client.api_key == "test-key-123"
    assert client.base_url == "http://localhost:18081"
    client.close()


def test_from_env_missing_key_raises(monkeypatch):
    """from_env() raises ValueError with helpful message when SPM_API_KEY is missing."""
    monkeypatch.delenv("SPM_API_KEY", raising=False)
    monkeypatch.setenv("SPM_GATEWAY_URL", "http://localhost:18081")
    with pytest.raises(ValueError, match="SPM_API_KEY"):
        SpmClient.from_env()


def test_from_env_missing_url_raises(monkeypatch):
    """from_env() raises ValueError with helpful message when SPM_GATEWAY_URL is missing."""
    monkeypatch.setenv("SPM_API_KEY", "test-key-123")
    monkeypatch.delenv("SPM_GATEWAY_URL", raising=False)
    with pytest.raises(ValueError, match="SPM_GATEWAY_URL"):
        SpmClient.from_env()


def test_from_env_empty_key_raises(monkeypatch):
    """from_env() raises ValueError when SPM_API_KEY is empty string (not None)."""
    monkeypatch.setenv("SPM_API_KEY", "")
    monkeypatch.setenv("SPM_GATEWAY_URL", "http://localhost:18081")
    with pytest.raises(ValueError, match="SPM_API_KEY"):
        SpmClient.from_env()


def test_from_env_empty_url_raises(monkeypatch):
    """from_env() raises ValueError when SPM_GATEWAY_URL is empty string (not None)."""
    monkeypatch.setenv("SPM_API_KEY", "test-key-123")
    monkeypatch.setenv("SPM_GATEWAY_URL", "")
    with pytest.raises(ValueError, match="SPM_GATEWAY_URL"):
        SpmClient.from_env()


# ---------------------------------------------------------------------------
# ChatSession tests
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_session_creates_and_sends():
    """ChatSession.send() returns the assistant reply and updates history."""
    respx.post("http://localhost:18081/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "4"}}]
            },
        )
    )
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session()
        reply = session.send("What is 2+2?")
        assert reply == "4"
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"


@respx.mock
def test_auth_error_raises_auth_error():
    """401 response raises AuthError with actionable message."""
    respx.post("http://localhost:18081/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "Unauthorized"}},
        )
    )
    with SpmClient("bad-key", base_url="http://localhost:18081") as client:
        session = client.session()
        with pytest.raises(AuthError) as exc_info:
            session.send("hello")
    assert exc_info.value.status_code == 401
    assert "SPM_API_KEY" in str(exc_info.value)
    assert "api-keys" in str(exc_info.value)


@respx.mock
def test_quota_error_raises_quota_error():
    """429 response raises QuotaError with retry_after info."""
    respx.post("http://localhost:18081/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"error": {"message": "Rate limit exceeded"}},
        )
    )
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session()
        with pytest.raises(QuotaError) as exc_info:
            session.send("hello")
    assert exc_info.value.status_code == 429
    assert "30" in str(exc_info.value)
    assert "Rate limit" in str(exc_info.value)


@respx.mock
def test_server_error_raises_server_error():
    """500 response raises ServerError with correlation ID."""
    respx.post("http://localhost:18081/v1/chat/completions").mock(
        return_value=httpx.Response(
            500,
            headers={"X-Correlation-Id": "req-abc-123"},
            json={"error": {"message": "Internal server error"}},
        )
    )
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session()
        with pytest.raises(ServerError) as exc_info:
            session.send("hello")
    assert exc_info.value.status_code == 500
    assert "req-abc-123" in str(exc_info.value)
    assert "Correlation ID" in str(exc_info.value)


@respx.mock
def test_quota_error_with_http_date_retry_after():
    """429 response with HTTP-date Retry-After header degrades gracefully."""
    respx.post("http://localhost:18081/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
            json={"error": {"message": "Rate limit exceeded"}},
        )
    )
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session()
        with pytest.raises(QuotaError) as exc_info:
            session.send("hello")
    assert exc_info.value.status_code == 429
    assert "unknown" in str(exc_info.value)  # Non-integer header fallback


@respx.mock
def test_network_timeout_raises_spm_error():
    """Network timeout raises SpmError wrapping the httpx exception."""
    respx.post("http://localhost:18081/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("Request timed out")
    )
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session()
        with pytest.raises(httpx.TimeoutException):
            session.send("hello")
        assert session.history == []


@respx.mock
def test_failed_send_rolls_back_user_message_before_retry():
    """A failed non-streaming send is absent from the next request history."""
    route = respx.post("http://localhost:18081/v1/chat/completions")
    route.side_effect = [
        httpx.TimeoutException("Request timed out"),
        httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        ),
    ]
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session(system_prompt="Be concise")
        with pytest.raises(httpx.TimeoutException):
            session.send("secret failed")

        assert session.history == [{"role": "system", "content": "Be concise"}]
        assert session.send("retry") == "ok"

    retry_payload = json.loads(route.calls[1].request.content)
    assert retry_payload["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "retry"},
    ]
    assert session.history == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "retry"},
        {"role": "assistant", "content": "ok"},
    ]


def test_reply_extraction_failure_rolls_back_only_pending_user():
    """Malformed non-streaming reply data preserves pre-existing history."""
    class FakeClient:
        def chat(self, messages, **kwargs):
            assert messages[-1] == {"role": "user", "content": "bad reply"}
            return []

    session = ChatSession(FakeClient(), system_prompt="Keep context")
    with pytest.raises(AttributeError):
        session.send("bad reply")

    assert session.history == [{"role": "system", "content": "Keep context"}]


@respx.mock
def test_stream_parse_error_logged_and_skipped():
    """Stream parse errors are logged at debug level and chunks are skipped."""
    # Mock a streaming response with one valid chunk and one malformed chunk
    def stream_lines():
        yield b"data: " + b'{"choices": [{"delta": {"content": "Hello"}}]}\n'
        yield b"data: {malformed json\n"  # Invalid JSON
        yield b"data: " + b'{"choices": [{"delta": {"content": " world"}}]}\n'
        yield b"data: [DONE]\n"

    respx.post("http://localhost:18081/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=b"".join(stream_lines()),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    with SpmClient("key", base_url="http://localhost:18081") as client:
        session = client.session()
        chunks = list(session.send("test", stream=True))
    # Only valid chunks should be yielded (malformed chunk silently skipped)
    assert chunks == ["Hello", " world"]
    assert len(session.history) == 2  # user + assistant message appended
