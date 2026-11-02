"""Stable response content hashing helpers."""

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def content_hash(text: str, effective_move: dict) -> str:
    payload = text + "\n" + canonical_json(effective_move)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
