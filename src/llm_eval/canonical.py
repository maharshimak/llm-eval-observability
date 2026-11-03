"""Canonical JSON serialisation and SHA-256 fingerprints.

Every fingerprint in this package is computed over the same canonical form:
``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)`` encoded as UTF-8. Non-finite floats are rejected so a
fingerprint always describes valid JSON.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()
