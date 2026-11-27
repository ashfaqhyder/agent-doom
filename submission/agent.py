"""Evaluation entry point for the recurrent spatial actor-critic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

if __package__:
    from .actions import ACTION_ARRAYS, ACTION_NAMES, NUM_BUTTONS
else:
    from actions import ACTION_ARRAYS, ACTION_NAMES, NUM_BUTTONS

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
UNTRAINED_CHECKPOINT_PATH = WEIGHTS_DIR / "untrained_actor_critic.pt"
PPO_INIT_CHECKPOINT_PATH = WEIGHTS_DIR / "ppo_smoke.pt"
_checkpoint_override = os.environ.get("DOOM_CHECKPOINT_PATH")
CHECKPOINT_PATH = Path(_checkpoint_override) if _checkpoint_override else WEIGHTS_DIR / "ppo_smoke.pt"
if not CHECKPOINT_PATH.is_absolute():
    CHECKPOINT_PATH = SUBMISSION_DIR / CHECKPOINT_PATH

NUM_RAYS = 64
NUM_RAY_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
HIDDEN_SIZE = 128


class ActorCritic(nn.Module):
    """Compact spatial encoder with recurrent policy and value heads."""

    def __init__(self, num_actions: int = len(ACTION_NAMES)) -> None:
        super().__init__()
        self.ray_encoder = nn.Sequential(
            nn.Linear(NUM_RAYS * (1 + NUM_RAY_TYPES), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.entity_type_embedding = nn.Embedding(NUM_ENTITY_TYPES, 8)
        self.entity_encoder = nn.Sequential(
            nn.Linear(5 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.vitals_encoder = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(64 + 32 + 32 + 1, HIDDEN_SIZE), nn.ReLU())
        self.gru = nn.GRUCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.actor = nn.Linear(HIDDEN_SIZE, num_actions)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    @staticmethod
    def _normalize_vitals(vitals: Tensor) -> Tensor:
        normalized = vitals.clone()
        normalized[:, 0:2] = normalized[:, 0:2] / 200.0
        normalized[:, 2] = normalized[:, 2] / 7.0
        normalized[:, 3:8] = torch.log1p(normalized[:, 3:8].clamp(min=0.0, max=200.0)) / np.log1p(200.0)
        normalized[:, 8:14] = normalized[:, 8:14].clamp(0.0, 1.0)
        normalized[:, 14:17] = torch.tanh(normalized[:, 14:17] / 20.0)
        return normalized

    def forward(
        self, observation: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        rays = observation["rays"]
        ray_distances = rays["distance"].clamp(0.0, 1.0).unsqueeze(-1)
        ray_types = F.one_hot(rays["hit_type"].long().clamp(0, NUM_RAY_TYPES - 1), NUM_RAY_TYPES)
        ray_features = torch.cat((ray_distances, ray_types.to(ray_distances.dtype)), dim=-1)
        ray_features = self.ray_encoder(ray_features.flatten(start_dim=1))

        entities = observation["visible_entities"]
        entity_mask = entities["mask"].to(torch.bool)
        entity_features = entities["features"].clone()
        entity_features[..., 3:5] = torch.tanh(entity_features[..., 3:5] / 20.0)
        entity_types = entities["entity_type"].long().clamp(0, NUM_ENTITY_TYPES - 1)
        entity_input = torch.cat(
            (entity_features, self.entity_type_embedding(entity_types)), dim=-1
        )
        encoded_entities = self.entity_encoder(entity_input)
        mask_weights = entity_mask.unsqueeze(-1).to(encoded_entities.dtype)
        entity_count = mask_weights.sum(dim=1)
        # A clamped denominator keeps the pooled encoding finite when every slot is empty.
        entity_pool = (encoded_entities * mask_weights).sum(dim=1) / entity_count.clamp_min(1.0)
        entity_fraction = entity_count / MAX_ENTITIES

        vitals = self.vitals_encoder(self._normalize_vitals(observation["vitals"]))
        fused = self.fusion(
            torch.cat((ray_features, vitals, entity_pool, entity_fraction), dim=-1)
        )
        next_hidden = self.gru(fused, hidden)
        return self.actor(next_hidden), self.critic(next_hidden).squeeze(-1), next_hidden


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Load the checkpoint once onto the evaluation device."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.inference_mode = os.environ.get("DOOM_INFERENCE_MODE", "greedy").strip().lower()
        if self.inference_mode not in ("greedy", "sampled"):
            raise ValueError(
                f"Unsupported inference mode {self.inference_mode!r}; expected 'greedy' or 'sampled'"
            )
        try:
            self.sampling_seed = int(os.environ.get("DOOM_SAMPLING_SEED", "0"))
        except ValueError as error:
            raise ValueError("DOOM_SAMPLING_SEED must be an integer") from error
        self.checkpoint_path = CHECKPOINT_PATH
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.format_version = checkpoint.get("format_version")
        if self.format_version == 1:
            action_count = NUM_BUTTONS
        elif self.format_version == 2:
            if checkpoint.get("action_names") != list(ACTION_NAMES):
                raise ValueError(f"Action mapping mismatch in {self.checkpoint_path}")
            action_count = len(ACTION_NAMES)
        else:
            raise ValueError(
                f"Unsupported checkpoint format {self.format_version!r} in {self.checkpoint_path}"
            )
        if self.inference_mode == "sampled" and self.format_version != 2:
            raise ValueError("Sampled inference requires a categorical format-version-2 checkpoint")
        self.model = ActorCritic(action_count).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.hidden = torch.zeros(1, HIDDEN_SIZE, device=self.device)
        self._rng = np.random.default_rng(self.sampling_seed)

    def reset(self, seed: int | None = None) -> None:
        """Clear GRU state and restart the local action RNG for a reproducible episode."""
        self.hidden = torch.zeros(1, HIDDEN_SIZE, device=self.device)
        self._rng = np.random.default_rng(
            self.sampling_seed if seed is None else int(seed)
        )

    def _policy_logits(self, observation: dict) -> Tensor:
        inputs: dict[str, Any] = {
            "rays": {
                "distance": torch.as_tensor(
                    observation["rays"]["distance"], dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                "hit_type": torch.as_tensor(
                    observation["rays"]["hit_type"], dtype=torch.long, device=self.device
                ).unsqueeze(0),
            },
            "vitals": torch.as_tensor(
                observation["vitals"], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "visible_entities": {
                "features": torch.as_tensor(
                    observation["visible_entities"]["features"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0),
                "entity_type": torch.as_tensor(
                    observation["visible_entities"]["entity_type"],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                "mask": torch.as_tensor(
                    observation["visible_entities"]["mask"],
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(0),
            },
        }
        logits, _, self.hidden = self.model(inputs, self.hidden)
        return logits[0]

    def act(self, observation: dict) -> np.ndarray:
        """Return a valid button vector using configured greedy or sampled inference."""
        with torch.inference_mode():
            logits = self._policy_logits(observation)
            if self.format_version == 1:
                return logits.ge(0).to(torch.int8).cpu().numpy()
            if self.inference_mode == "sampled":
                probabilities = logits.softmax(dim=-1).cpu().numpy().astype(np.float64)
                probabilities /= probabilities.sum()
                action_index = int(
                    self._rng.choice(len(ACTION_ARRAYS), p=probabilities)
                )
            else:
                action_index = int(logits.argmax().item())
            return ACTION_ARRAYS[action_index].copy()

    def act_with_probabilities(
        self, observation: dict, *, sample: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Diagnostic helper; sample=True uses Torch RNG, while sampled act() uses local RNG."""
        if self.format_version != 2:
            raise ValueError("Action probabilities are only available for categorical checkpoints")
        with torch.inference_mode():
            probabilities = self._policy_logits(observation).softmax(dim=-1)
            if sample:
                action_index = int(torch.multinomial(probabilities, num_samples=1).item())
            else:
                action_index = int(probabilities.argmax().item())
            return (
                ACTION_ARRAYS[action_index].copy(),
                probabilities.cpu().numpy().copy(),
            )
