import sys
from pathlib import Path

import torch
from torch.distributions import Categorical
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from submission.agent import ActorCritic, HIDDEN_SIZE
from train_ppo import (
    EpisodeRotatingMapEnv,
    FirstVisitReward,
    collect_rollout,
    compute_gae,
    pack_observation,
    tensor_observation,
    unroll_sequence,
)


def empty_observation():
    return {
        "rays": {
            "distance": torch.ones(64).numpy(),
            "hit_type": torch.zeros(64, dtype=torch.int64).numpy(),
        },
        "vitals": torch.zeros(17).numpy(),
        "visible_entities": {
            "features": torch.zeros(32, 5).numpy(),
            "entity_type": torch.zeros(32, dtype=torch.int64).numpy(),
            "mask": torch.zeros(32, dtype=torch.int8).numpy(),
        },
    }


def test_gae_bootstraps_time_limit_and_stops_episode_traces():
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.2, 0.2])
    bootstrap_values = torch.tensor([0.0, 0.5])
    terminated = torch.tensor([True, False])
    truncated = torch.tensor([False, True])

    advantages, returns = compute_gae(
        rewards,
        values,
        bootstrap_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(advantages, torch.tensor([0.8, 1.25]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 1.45]))


def test_ppo_recomputes_sampled_action_log_prob_from_rollout_start_state():
    torch.manual_seed(4)
    model = ActorCritic()
    initial_hidden = torch.randn(1, HIDDEN_SIZE)
    observation = pack_observation(empty_observation())
    with torch.no_grad():
        logits, _, _ = model(tensor_observation(observation, torch.device("cpu")), initial_hidden)
        distribution = Categorical(logits=logits)
        sampled_action = distribution.sample()
        stored_log_prob = distribution.log_prob(sampled_action)

    recomputed_log_prob, _, _ = unroll_sequence(
        model,
        [observation],
        sampled_action,
        torch.tensor([False]),
        initial_hidden,
        torch.device("cpu"),
    )
    torch.testing.assert_close(recomputed_log_prob, stored_log_prob)


def test_first_visit_reward_is_capped_and_episode_local():
    bonus = FirstVisitReward(reward_per_cell=0.02, cap=0.03)
    bonus.reset({"x": 10, "y": 10})

    assert bonus.reward({"x": 20, "y": 20}) == 0.0
    assert bonus.reward({"x": 70, "y": 20}) == 0.02
    assert bonus.reward({"x": 130, "y": 20}) == pytest.approx(0.01)
    assert bonus.reward({"x": 200, "y": 20}) == 0.0

    bonus.reset({"x": 10, "y": 10})
    assert bonus.reward({"x": 70, "y": 20}) == 0.02


def test_rollout_separates_health_penalty_from_official_return():
    class ShapedEnv:
        def step(self, action):
            return empty_observation(), 0.73, False, False, {
                "door_opening_reward": 0.1,
                "destination_crossing_reward": 0.4,
                "health_loss_penalty": -0.02,
            }

    model = ActorCritic()
    rollout = collect_rollout(
        ShapedEnv(),
        model,
        empty_observation(),
        torch.zeros(1, HIDDEN_SIZE),
        False,
        1,
        0,
        torch.device("cpu"),
        exploration=None,
    )

    torch.testing.assert_close(rollout.official_rewards, torch.tensor([0.25]))
    torch.testing.assert_close(rollout.shaping_rewards, torch.tensor([0.48]))
    torch.testing.assert_close(rollout.health_loss_penalties, torch.tensor([-0.02]))
    torch.testing.assert_close(rollout.rewards, torch.tensor([0.73]))


def test_multimap_env_rotates_on_episode_reset_and_tracks_health_separately(monkeypatch):
    class FakeEnv:
        observation_space = None
        action_space = None

        def __init__(self, map_name):
            self.map_name = map_name
            self.closed = False
            self.health = 100.0

        def reset(self, *, seed=None, options=None):
            return empty_observation(), {}

        def step(self, action):
            self.health -= 10.0
            return empty_observation(), 0.25, True, False, {}

        def privileged_state(self):
            return {"health": self.health}

        def close(self):
            self.closed = True

    created = []

    def fake_make_env(wad, map_name, **kwargs):
        env = FakeEnv(map_name)
        created.append(env)
        return env

    monkeypatch.setattr("train_ppo.make_env", fake_make_env)
    env = EpisodeRotatingMapEnv(
        Path("freedoom2.wad"),
        ["MAP03", "MAP04"],
        episode_ticks=2100,
        health_penalty_per_point=0.002,
    )
    try:
        observation, _ = env.reset(seed=7)
        model = ActorCritic()
        rollout = collect_rollout(
            env,
            model,
            observation,
            torch.zeros(1, HIDDEN_SIZE),
            True,
            2,
            7,
            torch.device("cpu"),
            exploration=None,
        )

        assert [instance.map_name for instance in created] == ["MAP03", "MAP04", "MAP03"]
        assert created[0].closed and created[1].closed
        torch.testing.assert_close(rollout.episode_starts, torch.tensor([True, True]))
        torch.testing.assert_close(rollout.official_rewards, torch.tensor([0.25, 0.25]))
        torch.testing.assert_close(rollout.health_loss_penalties, torch.tensor([-0.02, -0.02]))
        assert env.stats() == {
            "MAP03": {
                "decisions": 1,
                "completed_episodes": 1,
                "official_return": 0.25,
                "health_penalty": -0.02,
            },
            "MAP04": {
                "decisions": 1,
                "completed_episodes": 1,
                "official_return": 0.25,
                "health_penalty": -0.02,
            },
        }
    finally:
        env.close()
