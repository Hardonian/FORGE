"""Cryptographic hashing and deterministic fingerprinting utilities."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Dict


def canonical_json(data: Dict[str, Any]) -> str:
    """Format dictionary as canonical JSON with sorted keys and minimal separators."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hash(data: Dict[str, Any] | str | bytes) -> str:
    """Compute standard SHA-256 hex digest for inputs."""
    if isinstance(data, dict):
        raw = canonical_json(data).encode("utf-8")
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, bytes):
        raw = data
    else:
        raise TypeError(f"Unsupported type for hashing: {type(data)}")
    return hashlib.sha256(raw).hexdigest()

