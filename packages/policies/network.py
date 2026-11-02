from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from packages.learning.actions import STRATEGIST_ACTIONS
from packages.learning.features import FEATURE_NAMES

OBSERVATION_SIZE = len(FEATURE_NAMES)
FEATURE_SPEC_VERSION = "features_v1"
ACTION_SPEC_VERSION = "actions_v1"


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    allowed = mask.to(dtype=torch.bool, device=logits.device)
    if logits.shape != allowed.shape:
        raise ValueError("logits and action mask must have the same shape")
    if not torch.all(allowed.any(dim=-1)):
        raise ValueError("action mask contains a row with no legal actions")
    return logits.masked_fill(~allowed, -1e9)


class StrategistActorCritic(nn.Module):
    def __init__(self, *, seed: int | None = None, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.hidden = tuple(int(width) for width in hidden)
        layers: list[nn.Module] = []
        width_in = OBSERVATION_SIZE
        for width in self.hidden:
            layers.extend([nn.Linear(width_in, width), nn.Tanh()])
            width_in = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(width_in, STRATEGIST_ACTIONS)
        self.value_head = nn.Linear(width_in, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        hidden = self.trunk(obs.to(dtype=torch.float32))
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    @staticmethod
    def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_logits(logits, mask)

    def act(
        self, obs, mask, *, greedy: bool = False
    ) -> tuple[int | list[int], float | list[float], float | list[float]]:
        observation = torch.as_tensor(obs, dtype=torch.float32)
        action_mask = torch.as_tensor(mask, dtype=torch.bool)
        single = observation.ndim == 1
        logits, values = self(observation)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        distribution = Categorical(logits=masked_logits(logits, action_mask))
        actions = distribution.logits.argmax(dim=-1) if greedy else distribution.sample()
        log_prob = distribution.log_prob(actions)
        if single:
            return (
                int(actions.item()),
                float(log_prob.item()),
                float(values.item()),
            )
        return (
            actions.tolist(),
            log_prob.tolist(),
            values.tolist(),
        )

    def evaluate_actions(
        self, obs, mask, actions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = torch.as_tensor(obs, dtype=torch.float32)
        action_mask = torch.as_tensor(mask, dtype=torch.bool)
        action_values = torch.as_tensor(actions, dtype=torch.long)
        logits, values = self(observation)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        if action_values.ndim == 0:
            action_values = action_values.unsqueeze(0)
        distribution = Categorical(logits=masked_logits(logits, action_mask))
        return distribution.log_prob(action_values), distribution.entropy(), values


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_suffix(".metadata.json")


def save_checkpoint(
    model: StrategistActorCritic,
    checkpoint_path: str | Path,
    metadata: dict,
) -> Path:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    metadata_path = checkpoint_metadata_path(checkpoint)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


class NetworkStrategist:
    policy_version = "network_strategist_v1"

    def __init__(self, checkpoint_path: str | Path, *, greedy: bool = True) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = checkpoint_metadata_path(self.checkpoint_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint metadata not found: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._validate_metadata(self.metadata)
        self.model = StrategistActorCritic(hidden=tuple(self.metadata.get("hidden", (128, 128))))
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.greedy = greedy

    @staticmethod
    def _validate_metadata(metadata: dict) -> None:
        expected = {
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "action_spec_version": ACTION_SPEC_VERSION,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"checkpoint {key} must be {value!r}")
        if "seed" not in metadata or "init" not in metadata:
            raise ValueError("checkpoint metadata must include seed and init")

    @classmethod
    def from_model(
        cls, model: StrategistActorCritic, *, greedy: bool = False
    ) -> NetworkStrategist:
        instance = cls.__new__(cls)
        instance.checkpoint_path = None
        instance.metadata_path = None
        instance.metadata = {
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "action_spec_version": ACTION_SPEC_VERSION,
            "seed": 0,
            "init": "in_memory",
        }
        instance.model = model
        instance.greedy = greedy
        return instance

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        action, _, _ = self.model.act(observation, mask, greedy=self.greedy)
        return int(action)
