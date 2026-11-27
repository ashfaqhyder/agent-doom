"""Train the recurrent actor-critic with episode-rotating map PPO."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.actions import ACTION_ARRAYS, ACTION_NAMES
from submission.agent import (
    ActorCritic,
    HIDDEN_SIZE,
    PPO_INIT_CHECKPOINT_PATH,
)


@dataclass
class Rollout:
    observations: list[dict[str, Any]]
    actions: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    official_rewards: Tensor
    shaping_rewards: Tensor
    door_opening_rewards: Tensor
    destination_crossing_rewards: Tensor
    health_loss_penalties: Tensor
    bootstrap_values: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_starts: Tensor
    next_observation: dict[str, Any]
    next_hidden: Tensor
    next_episode_start: bool
    initial_hidden: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--intermediate-updates", type=int, default=0)
    parser.add_argument(
        "--checkpoint-initial",
        type=Path,
        default=PPO_INIT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--checkpoint-intermediate",
        type=Path,
        default=Path("submission/weights/ppo_intermediate.pt"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("submission/weights/ppo_categorical.pt"),
    )
    parser.add_argument("--wad", type=Path, default=Path("maps/two_rooms.wad"))
    parser.add_argument("--map-names", nargs="+", default=["MAP01"])
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--health-penalty-per-point", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exploration-reward-per-cell", type=float, default=0.0)
    parser.add_argument("--exploration-reward-cap", type=float, default=0.5)
    args = parser.parse_args()
    if args.updates < 1 or args.rollout_length < 1 or args.epochs < 1:
        parser.error("--updates, --rollout-length, and --epochs must be positive")
    if args.intermediate_updates and not 1 <= args.intermediate_updates < args.updates:
        parser.error("--intermediate-updates must be between 1 and updates - 1")
    if args.episode_ticks < 1:
        parser.error("--episode-ticks must be positive")
    if args.health_penalty_per_point < 0:
        parser.error("--health-penalty-per-point must be non-negative")
    if not args.map_names or len({name.upper() for name in args.map_names}) != len(args.map_names):
        parser.error("--map-names must contain unique map names")
    if args.exploration_reward_per_cell < 0 or args.exploration_reward_cap < 0:
        parser.error("exploration reward and cap must be non-negative")
    if args.intermediate_updates:
        initial = args.checkpoint_initial.resolve()
        intermediate = args.checkpoint_intermediate.resolve()
        output = args.checkpoint_output.resolve()
        if intermediate == output or intermediate == initial or output == initial:
            parser.error("initial, intermediate, and final checkpoint paths must be distinct")
    return args


def pack_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep only contract fields used by the policy and own the rollout arrays."""
    return {
        "rays": {
            "distance": np.array(observation["rays"]["distance"], copy=True),
            "hit_type": np.array(observation["rays"]["hit_type"], copy=True),
        },
        "vitals": np.array(observation["vitals"], copy=True),
        "visible_entities": {
            "features": np.array(observation["visible_entities"]["features"], copy=True),
            "entity_type": np.array(observation["visible_entities"]["entity_type"], copy=True),
            "mask": np.array(observation["visible_entities"]["mask"], copy=True),
        },
    }


def tensor_observation(observation: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "rays": {
            "distance": torch.as_tensor(
                observation["rays"]["distance"], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "hit_type": torch.as_tensor(
                observation["rays"]["hit_type"], dtype=torch.long, device=device
            ).unsqueeze(0),
        },
        "vitals": torch.as_tensor(
            observation["vitals"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "visible_entities": {
            "features": torch.as_tensor(
                observation["visible_entities"]["features"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "entity_type": torch.as_tensor(
                observation["visible_entities"]["entity_type"],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0),
            "mask": torch.as_tensor(
                observation["visible_entities"]["mask"],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0),
        },
    }


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Bootstrap time limits, but stop the GAE trace at either episode boundary."""
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * bootstrap_values[index] - values[index]
        boundary = terminated[index] | truncated[index]
        next_advantage = (
            delta
            + gamma * gae_lambda * (~boundary).to(rewards.dtype) * next_advantage
        )
        advantages[index] = next_advantage
    return advantages, advantages + values


class FirstVisitReward:
    """Training-only capped bonus for entering a new 64-map-unit grid cell."""

    def __init__(self, reward_per_cell: float, cap: float) -> None:
        self.reward_per_cell = reward_per_cell
        self.cap = cap
        self.visited: set[tuple[int, int]] = set()
        self.total = 0.0

    @staticmethod
    def _cell(state: dict[str, Any]) -> tuple[int, int]:
        return math.floor(float(state["x"]) / 64), math.floor(float(state["y"]) / 64)

    def reset(self, initial_state: dict[str, Any]) -> None:
        self.visited = {self._cell(initial_state)}
        self.total = 0.0

    def reward(self, state: dict[str, Any]) -> float:
        cell = self._cell(state)
        if cell in self.visited:
            return 0.0
        self.visited.add(cell)
        bonus = min(self.reward_per_cell, max(0.0, self.cap - self.total))
        self.total += bonus
        return bonus


class EpisodeRotatingMapEnv:
    """Cycle maps on reset and apply a training-only health-loss penalty."""

    def __init__(
        self,
        wad: Path,
        map_names: list[str],
        *,
        episode_ticks: int,
        health_penalty_per_point: float,
    ) -> None:
        if not map_names:
            raise ValueError("at least one map name is required")
        if health_penalty_per_point < 0:
            raise ValueError("health penalty per point must be non-negative")
        self.wad = wad
        self.map_names = tuple(name.upper() for name in map_names)
        self.episode_ticks = episode_ticks
        self.health_penalty_per_point = health_penalty_per_point
        self._map_index = -1
        self._env = None
        self._previous_health = 0.0
        self.decisions: Counter[str] = Counter()
        self.completed_episodes: Counter[str] = Counter()
        self.official_returns: Counter[str] = Counter()
        self.health_penalties: Counter[str] = Counter()

    @property
    def current_map(self) -> str:
        if self._map_index < 0:
            raise RuntimeError("call reset() before using the environment")
        return self.map_names[self._map_index]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._env is not None:
            self._env.close()
        self._map_index = (self._map_index + 1) % len(self.map_names)
        self._env = make_env(
            str(self.wad),
            self.current_map,
            episode_ticks=self.episode_ticks,
            render_buffers=False,
        )
        observation, info = self._env.reset(seed=seed, options=options)
        self._previous_health = float(self._env.privileged_state()["health"])
        return observation, info

    def step(self, action):
        if self._env is None:
            raise RuntimeError("call reset() before step()")
        observation, official_reward, terminated, truncated, info = self._env.step(action)
        state = self._env.privileged_state()
        health_lost = max(0.0, self._previous_health - float(state["health"]))
        self._previous_health = float(state["health"])
        penalty = -health_lost * self.health_penalty_per_point

        info = dict(info)
        info["health_loss_penalty"] = penalty
        info["health_lost"] = health_lost
        self.decisions[self.current_map] += 1
        self.official_returns[self.current_map] += float(official_reward)
        self.health_penalties[self.current_map] += penalty
        if terminated or truncated:
            self.completed_episodes[self.current_map] += 1
        return observation, float(official_reward) + penalty, terminated, truncated, info

    def privileged_state(self) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("call reset() before reading environment state")
        return self._env.privileged_state()

    def stats(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "decisions": int(self.decisions[name]),
                "completed_episodes": int(self.completed_episodes[name]),
                "official_return": float(self.official_returns[name]),
                "health_penalty": float(self.health_penalties[name]),
            }
            for name in self.map_names
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def unroll_sequence(
    model: ActorCritic,
    observations: list[dict[str, Any]],
    actions: Tensor,
    episode_starts: Tensor,
    initial_hidden: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate the rollout in temporal order from its saved initial GRU state."""
    hidden = initial_hidden
    log_probs: list[Tensor] = []
    values: list[Tensor] = []
    entropies: list[Tensor] = []
    for index, observation in enumerate(observations):
        if bool(episode_starts[index]):
            hidden = torch.zeros_like(hidden)
        logits, value, hidden = model(tensor_observation(observation, device), hidden)
        distribution = Categorical(logits=logits)
        action = actions[index].view(1)
        log_probs.append(distribution.log_prob(action).squeeze(0))
        values.append(value.squeeze(0))
        entropies.append(distribution.entropy().squeeze(0))
    return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


def collect_rollout(
    env,
    model: ActorCritic,
    observation: dict[str, Any],
    hidden: Tensor,
    episode_start: bool,
    length: int,
    seed: int,
    device: torch.device,
    exploration: FirstVisitReward | None,
) -> Rollout:
    observations: list[dict[str, Any]] = []
    actions: list[Tensor] = []
    old_log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    official_rewards: list[float] = []
    shaping_rewards: list[float] = []
    door_opening_rewards: list[float] = []
    destination_crossing_rewards: list[float] = []
    health_loss_penalties: list[float] = []
    bootstrap_values: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    starts: list[bool] = []
    starting_hidden = hidden.detach().clone()
    episode_number = 0

    for _ in range(length):
        policy_observation = pack_observation(observation)
        observations.append(policy_observation)
        starts.append(episode_start)
        with torch.no_grad():
            logits, value, next_hidden = model(tensor_observation(policy_observation, device), hidden)
            distribution = Categorical(logits=logits)
            sampled_action = distribution.sample()
            old_log_prob = distribution.log_prob(sampled_action)

        action_index = int(sampled_action.item())
        next_observation, reward, terminated, truncated, info = env.step(
            ACTION_ARRAYS[action_index].copy()
        )
        door_reward = float(info.get("door_opening_reward", 0.0))
        crossing_reward = float(info.get("destination_crossing_reward", 0.0))
        health_penalty = float(info.get("health_loss_penalty", 0.0))
        exploration_reward = (
            exploration.reward(env.privileged_state()) if exploration is not None else 0.0
        )
        shaping_reward = (
            door_reward + crossing_reward + health_penalty + exploration_reward
        )
        official_reward = (
            float(reward) - door_reward - crossing_reward - health_penalty
        )
        next_policy_observation = pack_observation(next_observation)
        if terminated:
            bootstrap = 0.0
        else:
            with torch.no_grad():
                _, next_value, _ = model(
                    tensor_observation(next_policy_observation, device), next_hidden
                )
            bootstrap = float(next_value.item())

        actions.append(sampled_action[0].to(torch.long).cpu())
        old_log_probs.append(float(old_log_prob.item()))
        values.append(float(value.item()))
        official_rewards.append(official_reward)
        shaping_rewards.append(shaping_reward)
        door_opening_rewards.append(door_reward)
        destination_crossing_rewards.append(crossing_reward)
        health_loss_penalties.append(health_penalty)
        rewards.append(official_reward + shaping_reward)
        bootstrap_values.append(bootstrap)
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))

        if terminated or truncated:
            episode_number += 1
            observation, _ = env.reset(seed=seed + episode_number)
            if exploration is not None:
                exploration.reset(env.privileged_state())
            hidden = torch.zeros(1, HIDDEN_SIZE, device=device)
            episode_start = True
        else:
            observation = next_observation
            hidden = next_hidden.detach()
            episode_start = False

    return Rollout(
        observations=observations,
        actions=torch.stack(actions),
        old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        values=torch.tensor(values, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        official_rewards=torch.tensor(
            official_rewards, dtype=torch.float32, device=device
        ),
        shaping_rewards=torch.tensor(
            shaping_rewards, dtype=torch.float32, device=device
        ),
        door_opening_rewards=torch.tensor(
            door_opening_rewards, dtype=torch.float32, device=device
        ),
        destination_crossing_rewards=torch.tensor(
            destination_crossing_rewards, dtype=torch.float32, device=device
        ),
        health_loss_penalties=torch.tensor(
            health_loss_penalties, dtype=torch.float32, device=device
        ),
        bootstrap_values=torch.tensor(bootstrap_values, dtype=torch.float32, device=device),
        terminated=torch.tensor(terminations, dtype=torch.bool, device=device),
        truncated=torch.tensor(truncations, dtype=torch.bool, device=device),
        episode_starts=torch.tensor(starts, dtype=torch.bool, device=device),
        next_observation=observation,
        next_hidden=hidden,
        next_episode_start=episode_start,
        initial_hidden=starting_hidden,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    checkpoint_path = args.checkpoint_initial
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ActorCritic(len(ACTION_NAMES)).to(device)
    format_version = checkpoint.get("format_version")
    if format_version == 1:
        shared_state = {
            name: value
            for name, value in checkpoint["model_state_dict"].items()
            if not name.startswith("actor.")
        }
        incompatible = model.load_state_dict(shared_state, strict=False)
        if (
            set(incompatible.missing_keys) != {"actor.weight", "actor.bias"}
            or incompatible.unexpected_keys
        ):
            raise ValueError(f"Could not transfer shared model weights from {checkpoint_path}")
    elif format_version == 2 and checkpoint.get("action_names") == list(ACTION_NAMES):
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Unsupported or incompatible checkpoint at {checkpoint_path}")
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    wad_path = args.wad
    if not wad_path.is_absolute():
        wad_path = ROOT / wad_path
    env = EpisodeRotatingMapEnv(
        wad_path,
        args.map_names,
        episode_ticks=args.episode_ticks,
        health_penalty_per_point=args.health_penalty_per_point,
    )
    observation, _ = env.reset(seed=args.seed)
    hidden = torch.zeros(1, HIDDEN_SIZE, device=device)
    episode_start = True
    exploration = (
        FirstVisitReward(args.exploration_reward_per_cell, args.exploration_reward_cap)
        if args.exploration_reward_per_cell > 0
        else None
    )
    if exploration is not None:
        exploration.reset(env.privileged_state())
    environment_steps = 0
    official_reward_total = 0.0
    shaping_reward_total = 0.0
    nonzero_official_reward_steps = 0
    try:
        for update in range(1, args.updates + 1):
            rollout = collect_rollout(
                env,
                model,
                observation,
                hidden,
                episode_start,
                args.rollout_length,
                args.seed + (update - 1) * args.rollout_length,
                device,
                exploration,
            )
            observation = rollout.next_observation
            hidden = rollout.next_hidden
            episode_start = rollout.next_episode_start
            environment_steps += args.rollout_length
            official_reward_total += float(rollout.official_rewards.sum().item())
            shaping_reward_total += float(rollout.shaping_rewards.sum().item())
            nonzero_official_reward_steps += int(
                torch.count_nonzero(rollout.official_rewards).item()
            )
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.bootstrap_values,
                rollout.terminated,
                rollout.truncated,
            )
            normalized_advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            model.train()
            for _ in range(args.epochs):
                new_log_probs, new_values, entropies = unroll_sequence(
                    model,
                    rollout.observations,
                    rollout.actions,
                    rollout.episode_starts,
                    rollout.initial_hidden,
                    device,
                )
                ratio = torch.exp(new_log_probs - rollout.old_log_probs)
                unclipped = ratio * normalized_advantages
                clipped = ratio.clamp(1.0 - 0.2, 1.0 + 0.2) * normalized_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(new_values, returns)
                entropy = entropies.mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            print(
                f"update={update}/{args.updates} steps={environment_steps} "
                f"official_reward={rollout.official_rewards.sum().item():.3f} "
                f"health_penalty={rollout.health_loss_penalties.sum().item():.3f} "
                f"shaping_reward={rollout.shaping_rewards.sum().item():.3f} "
                f"per_map={json.dumps(env.stats(), sort_keys=True)} "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}"
            )

            if update == args.intermediate_updates:
                _save_checkpoint(
                    args.checkpoint_intermediate,
                    model,
                    args,
                    checkpoint_path,
                    update,
                    environment_steps,
                    official_reward_total,
                    shaping_reward_total,
                    nonzero_official_reward_steps,
                    env.stats(),
                )

        changed = any(
            not torch.equal(initial_parameters[name], parameter.detach().cpu())
            for name, parameter in model.state_dict().items()
        )
        if not changed:
            raise RuntimeError("PPO completed without changing any model weights")

        _save_checkpoint(
            args.checkpoint_output,
            model,
            args,
            checkpoint_path,
            args.updates,
            environment_steps,
            official_reward_total,
            shaping_reward_total,
            nonzero_official_reward_steps,
            env.stats(),
        )
        output_path = args.checkpoint_output
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        print(f"weights_changed=yes checkpoint={output_path}")
        print(
            f"environment_steps={environment_steps} ppo_updates={args.updates} "
            f"official_reward_total={official_reward_total:.3f} "
            f"shaping_reward_total={shaping_reward_total:.3f} "
            f"health_penalty_total={sum(env.health_penalties.values()):.3f} "
            f"nonzero_official_reward_steps={nonzero_official_reward_steps} "
            f"per_map={json.dumps(env.stats(), sort_keys=True)}"
        )
    finally:
        env.close()


def _save_checkpoint(
    configured_path: Path,
    model: ActorCritic,
    args: argparse.Namespace,
    initial_checkpoint: Path,
    updates: int,
    environment_steps: int,
    official_reward_total: float,
    shaping_reward_total: float,
    nonzero_official_reward_steps: int,
    map_stats: dict[str, dict[str, float | int]],
) -> Path:
    output_path = configured_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "action_names": list(ACTION_NAMES),
            "training_status": f"Categorical PPO-trained ({updates} updates)",
            "training": {
                "seed": args.seed,
                "initial_checkpoint": str(initial_checkpoint),
                "updates": updates,
                "rollout_length": args.rollout_length,
                "environment_steps": environment_steps,
                "official_reward_total": official_reward_total,
                "shaping_reward_total": shaping_reward_total,
                "health_penalty_per_point": args.health_penalty_per_point,
                "health_penalty_total": sum(
                    float(stats["health_penalty"]) for stats in map_stats.values()
                ),
                "nonzero_official_reward_steps": nonzero_official_reward_steps,
                "exploration_reward_per_cell": args.exploration_reward_per_cell,
                "exploration_reward_cap": args.exploration_reward_cap,
                "wad": str(args.wad),
                "map_names": [name.upper() for name in args.map_names],
                "episode_rotation": "sequential at episode boundaries",
                "episode_ticks": args.episode_ticks,
                "per_map": map_stats,
                "reward": "doom_env default reward; health loss penalty logged separately",
            },
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )
    print(f"checkpoint_saved={output_path} updates={updates} steps={environment_steps}")
    return output_path


if __name__ == "__main__":
    main()
