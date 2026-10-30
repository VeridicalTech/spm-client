# Security policy

## Reporting a vulnerability

Do not open a public issue for credentials, tenant isolation, authentication,
provider-key handling, evidence verification, or other security-sensitive
problems. Report privately to [security@spmos.ai](mailto:security@spmos.ai)
with a description, affected version/commit, reproduction steps, and the
minimum evidence needed to reproduce the issue. Redact API keys and customer
content.

We will acknowledge a report as soon as practical, investigate the affected
release boundary, and coordinate disclosure and a fix with the reporter.

## Scope

This repository contains the Python client and integration wrappers. The hosted
SPM service, production gateway, and provider infrastructure are separate
systems; do not test them without written authorization.

## Supported releases

The latest tagged release is the only release receiving routine security fixes.
Checkout snapshots and pre-release builds are not security support guarantees.

## Credential hygiene

Never commit SPM keys, provider keys, tokens, private evidence material, or
customer prompts. If a credential is exposed, revoke it immediately and report
the incident privately.
