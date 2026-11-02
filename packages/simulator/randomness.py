from __future__ import annotations

import hashlib
import json

import numpy as np


def _stable_entropy(parts: tuple[object, ...]) -> list[int]:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return [int.from_bytes(digest[offset : offset + 4], "little") for offset in range(0, 16, 4)]


class EventRandomBank:
    def rng(
        self, scenario_id: str, episode_seed: int, turn: int, event_type: str
    ) -> np.random.Generator:
        seed_sequence = np.random.SeedSequence(
            _stable_entropy((scenario_id, int(episode_seed), int(turn), event_type))
        )
        return np.random.default_rng(seed_sequence)

    @staticmethod
    def action_rng(episode_seed: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([int(episode_seed), 0xA37E]))
