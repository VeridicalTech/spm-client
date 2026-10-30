# spm-client

Python client for SPM: the hosted memory plane, the OpenAI-compatible gateway,
and offline verification of the evidence handles a recall returns.

**Status**: pre-release. PyPI publishing is blocked by `LEGAL_RELEASE_BLOCKER`
until legal sign-off; install from a checkout.

## Install

```bash
pip install .                         # client only: remember/recall/read/delete/status,
                                      # request receipts, gateway chat
pip install ".[evidence]"             # + Ed25519 evidence verification (cryptography)
pip install -e .                       # editable, for development
```

The only base runtime dependency is `httpx`. Install `spm-client[integrations]`
when using the official OpenAI or Anthropic drop-in helpers; those SDKs are
optional. `cryptography` is optional and needed only to verify an EdDSA handle.

## Memory plane

```python
from spm_client import SPMMemoryClient, SPMConfig

with SPMMemoryClient(SPMConfig(api_key="spm_live_...")) as spm:
    spm.remember(text="Ryan switched to lattes on 2026-09-15.",
                 idempotency_key="note-2026-09-20-a")
    result = spm.recall("what does Ryan drink?")
    print(result.answer, result.status)
    if result.supported:
        for span in spm.read([ref.read_token for ref in result.evidence]):
            print(span)
```

`AsyncSPMMemoryClient` is the same surface with `await`. Both are context
managers. Failures raise `SPMError` with a stable `category`; retryable ones
carry `retry_after`.

## Gateway (chat)

```python
from spm_client import SpmClient

with SpmClient("your-api-key", base_url="https://api.spmos.ai") as client:
    session = client.session()
    print(session.send("Hello, remember my name is Alice."))
    print(session.send("What's my name?"))   # session history is automatic
```

Streaming, receipts (`client.list_receipts()`, `client.get_receipt(id)`) and the
typed error hierarchy (`AuthError`, `QuotaError`, `NotFoundError`,
`ServerError`, base `SpmError`) are covered in `examples/`.

## Drop-in OpenAI / Anthropic clients

Install the optional integration dependencies:

```bash
pip install ".[integrations]"
```

OpenAI:

```python
from openai import OpenAI
from spm_client import patch_openai

client = patch_openai(
    OpenAI(api_key="sk-upstream"),
    api_key="spm_live_...",
    base_url="https://api.spmos.ai",
)
reply = client.chat.completions.create(
    model="your-configured-model",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Anthropic:

```python
from anthropic import Anthropic
from spm_client import patch_anthropic

client = patch_anthropic(
    Anthropic(api_key="sk-ant-upstream"),
    api_key="spm_live_...",
    base_url="https://api.spmos.ai",
)
reply = client.messages.create(
    model="your-configured-model",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
```

The helpers update the official SDK's actual HTTP transport, not only public
attributes: requests go to SPM's `/v1/chat/completions` or `/v1/messages` and
use the SPM key. The upstream provider key is replaced for that client object;
keep it only when you need to construct the client before patching, and never
commit either credential.

An explicit SPM key is required whenever a helper changes `base_url`; the
helper will not silently forward an upstream credential to SPM. OpenAI URLs
are normalized idempotently, so both `https://api.spmos.ai` and
`https://api.spmos.ai/v1` are accepted.

For a one-line OpenAI client, set `SPM_API_KEY` and `SPM_GATEWAY_URL`, then call
`spm_client.auto()`. The helper validates environment variables before importing
the optional OpenAI dependency and gives an actionable install error.

## Verify an evidence handle offline

```python
import requests
from spm_client import verify_evidence_token

keys = {
    key["kid"]: key["x"]
    for key in requests.get(
        "https://api.spmos.ai/.well-known/spm-evidence-keys.json", timeout=10
    ).json()["keys"]
}
verified = verify_evidence_token(ref.read_token, public_keys=keys)
assert verified.source_id == ref.source_id
```

Verification needs no credential and no call back to the server that minted the
handle. See `docs/sdk/EVIDENCE_VERIFICATION.md` for what it does and does not
prove.

## Two error bases

`SPMError` (memory plane) and `SpmError` (gateway client) are unrelated
classes; import the one you mean.

## Tests

```bash
pip install ".[dev]"
python -m pytest client_tests -q
```

The repository suite (`tests/`) runs the client's tests too, against the source
tree in this directory. The client's own directory is named `client_tests` rather
than `tests` on purpose: the repository already has a `tests` namespace, and a
second `tests` package inside a directory that goes on `sys.path` shadows it.
