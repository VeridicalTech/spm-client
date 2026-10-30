# Changelog

## 0.2.1 — pre-release

- Added tested `patch_openai`, `patch_anthropic`, and `auto` helpers.
- Synchronised the official SDK client's internal HTTP transport, URL, and
  authentication headers; changing `base_url` requires an explicit SPM key.
- Made OpenAI `/v1` URL handling idempotent.
- Added optional `[integrations]` dependencies for official OpenAI and
  Anthropic clients.
- PyPI publication is pending legal release approval.

## 0.2.0

- Hosted memory, gateway chat/session, receipts, and offline evidence
  verification surfaces.
