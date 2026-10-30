"""Tests for the drop-in integration wrappers (bd SPM-Polaris-814s)."""
from __future__ import annotations

import pytest
import httpx
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from spm_client import auto, patch_anthropic, patch_openai


class _FakeOpenAI:
    __module__ = "openai"

    def __init__(self):
        self.base_url = "https://api.openai.com/v1"
        self.api_key = "sk-default"
        self._client = type("Transport", (), {"base_url": httpx.URL(self.base_url), "headers": httpx.Headers()})()


class _FakeAnthropic:
    __module__ = "anthropic"

    def __init__(self):
        self.base_url = "https://api.anthropic.com"
        self.api_key = "ant-default"
        self._client = type("Transport", (), {"base_url": httpx.URL(self.base_url), "headers": httpx.Headers()})()


def test_patch_openai_points_at_gateway_v1():
    client = _FakeOpenAI()
    patch_openai(client, api_key="spm-live-key", base_url="https://api.spmos.ai")
    assert client.api_key == "spm-live-key"
    assert client.base_url == "https://api.spmos.ai/v1"
    assert str(client._client.base_url) == "https://api.spmos.ai/v1"
    assert client._client.headers["authorization"] == "Bearer spm-live-key"


def test_patch_anthropic_points_at_gateway_root():
    client = _FakeAnthropic()
    patch_anthropic(client, api_key="spm-live-key", base_url="https://api.spmos.ai")
    assert client.api_key == "spm-live-key"
    assert client.base_url == "https://api.spmos.ai"
    assert str(client._client.base_url) == "https://api.spmos.ai"
    assert client._client.headers["x-api-key"] == "spm-live-key"


def test_patch_keeps_empty_unchanged():
    client = _FakeOpenAI()
    patch_openai(client)
    assert client.base_url == "https://api.openai.com/v1"
    assert client.api_key == "sk-default"


def test_patch_requires_explicit_spm_key_when_changing_base_url():
    with pytest.raises(ValueError, match="api_key is required"):
        patch_openai(_FakeOpenAI(), base_url="https://api.spmos.ai")
    with pytest.raises(ValueError, match="api_key is required"):
        patch_anthropic(_FakeAnthropic(), base_url="https://api.spmos.ai")


def test_openai_base_url_suffix_is_idempotent():
    client = _FakeOpenAI()
    patch_openai(client, api_key="spm-live-key", base_url="https://api.spmos.ai/v1/")
    assert client.base_url == "https://api.spmos.ai/v1"


def test_auto_requires_env(monkeypatch):
    monkeypatch.delenv("SPM_API_KEY", raising=False)
    monkeypatch.delenv("SPM_GATEWAY_URL", raising=False)
    with pytest.raises(ValueError, match="SPM_API_KEY"):
        auto()


def test_auto_reports_missing_openai_sdk_after_env_validation(monkeypatch):
    monkeypatch.setenv("SPM_API_KEY", "spm-test")
    monkeypatch.setenv("SPM_GATEWAY_URL", "https://api.spmos.ai")
    # Simulate the optional dependency being unavailable after env validation.
    import builtins
    original_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="OpenAI SDK"):
        auto()


class _CaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.requests.append((self.path, dict(self.headers), self.rfile.read(length)))
        body = json.dumps({"id": "resp_test", "choices": [{"message": {"content": "ok"}}]}).encode()
        if self.path.endswith("/messages"):
            body = json.dumps({"id": "msg_test", "content": [{"type": "text", "text": "ok"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def test_real_openai_and_anthropic_clients_send_to_patched_transport():
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    OpenAI = openai.OpenAI
    Anthropic = anthropic.Anthropic

    _CaptureHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        openai = patch_openai(OpenAI(api_key="initial", base_url="http://127.0.0.1:1"), api_key="spm_openai", base_url=root)
        openai.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        anthropic = patch_anthropic(Anthropic(api_key="initial", base_url="http://127.0.0.1:1"), api_key="spm_anthropic", base_url=root)
        anthropic.messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert [item[0] for item in _CaptureHandler.requests] == ["/v1/chat/completions", "/v1/messages"]
    assert _CaptureHandler.requests[0][1].get("Authorization") == "Bearer spm_openai"
    assert _CaptureHandler.requests[1][1].get("X-Api-Key") == "spm_anthropic"
