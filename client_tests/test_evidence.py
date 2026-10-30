"""Conformance and verification tests for the evidence module.

The module's docstring claims its canonicalisation is held byte-identical to the
server's by a corpus test. That claim lives here, together with the signature
paths an operator would actually run: HS256 (stdlib, shared secret) and Ed25519
(public key).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from spm_client import evidence


CORPUS = [
    {},
    {"b": 2, "a": 1},
    {"nested": {"z": [1, 2, 3], "a": {"deep": True}}},
    {"nil": None, "flag": False, "count": 7},
    {"float": 1.5, "tiny": 1e-7, "big": 1e21},
    {"\U0001f600": "non-bmp key", "a": "\u00e9"},
    {"\uff5e": 1, "\ue000": 2},
    {"list": [], "obj": {}, "text": "spaces and \"quotes\""},
]


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _hs256_token(payload: dict, secret: str) -> dict:
    message = evidence.canonical_payload_bytes(payload)
    signature = hmac.new(secret.encode(), message, hashlib.sha256).digest()
    return {"payload": payload, "signature": _b64(signature)}


def _payload(**overrides):
    payload = {
        "typ": evidence.EVIDENCE_WIRE_TYPE,
        "wire_version": evidence.EVIDENCE_WIRE_VERSION,
        "alg": "HS256",
        "kid": "test-kid",
        "exp": 4102444800,
        "source_id": "src_01H",
    }
    payload.update(overrides)
    return payload


def test_canonical_bytes_match_the_server_module():
    server = pytest.importorskip("integrations.saas_api.canonical")
    for value in CORPUS:
        assert evidence.canonical_payload_bytes(value) == server.canonical_bytes(value), value


def test_hs256_round_trip_verifies():
    token = _hs256_token(_payload(), "secret")
    verified = evidence.verify_evidence_token(token, shared_secret="secret")
    assert verified.payload["source_id"] == "src_01H"


def test_tampered_payload_is_rejected():
    token = _hs256_token(_payload(), "secret")
    token["payload"]["source_id"] = "src_EVIL"
    with pytest.raises(evidence.EvidenceVerificationError):
        evidence.verify_evidence_token(token, shared_secret="secret")


def test_wrong_secret_is_rejected():
    token = _hs256_token(_payload(), "secret")
    with pytest.raises(evidence.EvidenceVerificationError):
        evidence.verify_evidence_token(token, shared_secret="other")


def test_wire_version_and_type_are_type_exact():
    for bad in ({"wire_version": "1"}, {"wire_version": 1.0}, {"wire_version": True}, {"wire_version": 2}):
        token = _hs256_token(_payload(**bad), "secret")
        with pytest.raises(evidence.EvidenceVerificationError):
            evidence.verify_evidence_token(token, shared_secret="secret")
    token = _hs256_token(_payload(typ="something-else+v1"), "secret")
    with pytest.raises(evidence.EvidenceVerificationError):
        evidence.verify_evidence_token(token, shared_secret="secret")


def test_unknown_key_and_algorithm_are_named_failures():
    token = _hs256_token(_payload(kid=""), "secret")
    with pytest.raises(evidence.EvidenceVerificationError) as missing:
        evidence.verify_evidence_token(token, shared_secret="secret")
    # The reason is the machine-readable contract; the message is prose for a
    # human reading a log line.
    assert missing.value.reason == "unknown_key"
    token = _hs256_token(_payload(alg="none"), "secret")
    with pytest.raises(evidence.EvidenceVerificationError):
        evidence.verify_evidence_token(token, shared_secret="secret")


def test_expired_token_is_rejected_unless_checking_is_off():
    # The wire field is expiry, and the server mints the same name - a client
    # reading exp would silently never enforce it.
    token = _hs256_token(_payload(expiry=1), "secret")
    with pytest.raises(evidence.EvidenceVerificationError):
        evidence.verify_evidence_token(token, shared_secret="secret")
    assert evidence.verify_evidence_token(token, shared_secret="secret", check_expiry=False)


def test_ed25519_round_trip_and_tamper():
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    raw_public = key.public_key().public_bytes_raw()
    payload = _payload(alg="Ed25519")
    message = evidence.canonical_payload_bytes(payload)
    token = {"payload": payload, "signature": _b64(key.sign(message))}
    verified = evidence.verify_evidence_token(token, public_keys={"test-kid": raw_public})
    assert verified.payload["kid"] == "test-kid"

    tampered = {"payload": {**payload, "source_id": "src_EVIL"}, "signature": token["signature"]}
    with pytest.raises(evidence.EvidenceVerificationError):
        evidence.verify_evidence_token(tampered, public_keys={"test-kid": raw_public})
    _ = cryptography
