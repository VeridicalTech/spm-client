"""Offline verification of an SPM evidence handle.

A recall answer carries evidence_refs, each with a read_token. That token is a
signed envelope: the payload says which source, which span, which decision, and
the signature binds it. read() resolves it server-side, but a caller that wants
to check the handle itself should not have to ask the server that minted it -
that is what this module is for.

It deliberately re-implements canonicalisation instead of importing the server's
(integrations/saas_api/canonical.py) so a third party can copy this file without
the rest of the product. The copy is held byte-identical to the server's by a
conformance test over a corpus that includes the awkward cases: non-BMP keys
(keys sort by UTF-16 code units, not code points), floats, and nested documents.

Algorithm support: HS256 needs only the stdlib but a shared secret, so it is for
our own operational checks. Ed25519 needs the shared secret's replacement - a
published public key - and therefore needs the cryptography package, imported
lazily so the stdlib-only path keeps working without it.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

EVIDENCE_WIRE_TYPE = "gated-evidence+v1"
# Source spans are a second signed wire type under the same scheme (the payload
# fields differ, the signature does not). Production returns both in one recall,
# so a verifier that only knows the first one reports a false failure on spans.
EVIDENCE_WIRE_TYPES = (EVIDENCE_WIRE_TYPE, "temporary-source-span+v1")
EVIDENCE_WIRE_VERSION = 1
SUPPORTED_ALGORITHMS = ("HS256", "EdDSA", "Ed25519")


class EvidenceVerificationError(RuntimeError):
    """The handle could not be verified; reason names which check failed."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass(frozen=True)
class VerifiedEvidence:
    payload: Mapping[str, Any]
    algorithm: str
    kid: str
    expiry: int
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return str(self.payload.get("source_id", ""))

    @property
    def evidence_text(self) -> str:
        return str(self.payload.get("evidence_text", ""))


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be")


def _serialize_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite numbers are not valid canonical JSON")
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e", 1)
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        return f"{mantissa}e{sign}{exponent}"
    return text


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _serialize_float(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_serialize(k)}:{_serialize(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid canonical JSON")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            if key in normalized:
                raise ValueError("duplicate canonical JSON object key")
            normalized[key] = _normalize(item)
        return {key: normalized[key] for key in sorted(normalized, key=_utf16_sort_key)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Byte-for-byte the server's canonicalisation of an evidence payload."""
    return _serialize(_normalize(payload)).encode("utf-8")


def _b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


# The ways a base64url field can fail to decode: a character the ASCII encode
# refuses, a length or padding that cannot form whole groups, and anything else
# the decoder raises as a ValueError. One list, because both fields that reach a
# decoder - the wire signature and a caller-supplied public key - have to keep
# those failures inside this module's documented error type.
_B64URL_DECODE_ERRORS = (binascii.Error, UnicodeEncodeError, ValueError)


def _decoded_signature(signature: str) -> bytes:
    """Signature bytes, with every malformed spelling kept inside our error type.

    The signature is caller-supplied text, and there are two ways for it to be
    unusable as base64url: characters the ASCII encode refuses
    (UnicodeEncodeError) and a length or padding that cannot form whole groups
    (binascii.Error). Both mean the same thing here - this signature cannot be
    checked - and both used to escape as built-ins even though this module
    documents that verification failures arrive as EvidenceVerificationError.
    The gate reaches the same verdict by catching the decode and reporting a
    mismatch rather than a separate kind of failure, so this mirrors it.
    """
    try:
        return _b64url_decode(signature)
    except _B64URL_DECODE_ERRORS as exc:
        raise EvidenceVerificationError("signature", "evidence signature is invalid") from exc


def _verify_ed25519(public_key: Any, message: bytes, signature: bytes, kid: str) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise EvidenceVerificationError(
            "unsupported_algorithm",
            "Ed25519 verification needs the cryptography package",
        ) from exc

    if isinstance(public_key, (bytes, bytearray)):
        raw = bytes(public_key)
    elif isinstance(public_key, str):
        try:
            raw = _b64url_decode(public_key)
        except _B64URL_DECODE_ERRORS as exc:
            # A key this verifier cannot decode is a key it cannot use, and the
            # branch below already answers `signature` for a key it cannot use:
            # from_public_bytes() raises ValueError for a wrong length and the
            # curve check does too. `unknown_key` stays what it was - the caller
            # supplied no usable key at all - so this is the same verdict a
            # caller gets for any other unusable key, never a leaked built-in.
            raise EvidenceVerificationError(
                "signature", "evidence signature is invalid"
            ) from exc
    else:
        raise EvidenceVerificationError("unknown_key", f"no usable public key for kid {kid}")
    try:
        Ed25519PublicKey.from_public_bytes(raw).verify(signature, message)
    except (InvalidSignature, ValueError) as exc:
        raise EvidenceVerificationError("signature", "evidence signature is invalid") from exc


def verify_evidence_token(
    token: Mapping[str, Any] | str,
    *,
    public_keys: Mapping[str, Any] | None = None,
    shared_secret: str | None = None,
    now: int | None = None,
    check_expiry: bool = True,
) -> VerifiedEvidence:
    """Verify one evidence handle.

    public_keys maps kid to an Ed25519 public key (raw 32 bytes or base64url).
    shared_secret is the HS256 secret, for our own operational checks only.
    Raises EvidenceVerificationError; never returns a token it could not verify.
    """
    if isinstance(token, str):
        try:
            token = json.loads(token)
        except ValueError as exc:
            raise EvidenceVerificationError("malformed", "evidence token is not JSON") from exc
    if not isinstance(token, Mapping):
        raise EvidenceVerificationError("malformed", "evidence token must be an object")
    payload = token.get("payload")
    signature = token.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, str) or not signature:
        raise EvidenceVerificationError("malformed", "evidence token needs payload and signature")
    # Type-exact on purpose. int() would coerce "1", 1.0 and True into an
    # accepted version, and leak TypeError/ValueError out of the verifier for
    # null or junk; bool is an int subclass, so only `type(...) is int` both
    # accepts the integer 1 and rejects the rest as a named failure.
    wire_version = payload.get("wire_version")
    if (
        payload.get("typ") not in EVIDENCE_WIRE_TYPES
        or type(wire_version) is not int
        or wire_version != EVIDENCE_WIRE_VERSION
    ):
        raise EvidenceVerificationError("wire_version", "unsupported evidence wire version")

    algorithm = str(payload.get("alg", ""))
    kid = str(payload.get("kid", ""))
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise EvidenceVerificationError("unsupported_algorithm", f"unsupported alg {algorithm!r}")
    if not kid:
        raise EvidenceVerificationError("unknown_key", "evidence payload carries no kid")

    try:
        message = canonical_payload_bytes(payload)
    except (TypeError, ValueError) as exc:
        # The payload is caller-owned until its signature verifies. Keep
        # malformed canonical values inside the SDK's documented error type so
        # consumers never need to guess whether a built-in came from the token
        # or from their own verification plumbing.
        raise EvidenceVerificationError(
            "canonicalization", "evidence payload canonicalization is invalid"
        ) from exc
    raw_signature = _decoded_signature(signature)
    if algorithm == "HS256":
        if not shared_secret:
            raise EvidenceVerificationError(
                "shared_secret_required",
                "this handle is HMAC-signed: only the signing service or a holder of the secret can verify it",
            )
        expected = hmac.new(shared_secret.encode("utf-8"), message, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, raw_signature):
            raise EvidenceVerificationError("signature", "evidence signature is invalid")
    else:
        if not public_keys or kid not in public_keys:
            raise EvidenceVerificationError("unknown_key", f"no public key supplied for kid {kid}")
        _verify_ed25519(public_keys[kid], message, raw_signature, kid)

    expiry = int(payload.get("expiry", 0) or 0)
    if check_expiry and expiry and expiry < (now if now is not None else int(time.time())):
        raise EvidenceVerificationError("expired", "evidence handle has expired")
    return VerifiedEvidence(
        payload=dict(payload),
        algorithm=algorithm,
        kid=kid,
        expiry=expiry,
        raw=dict(token),
    )
