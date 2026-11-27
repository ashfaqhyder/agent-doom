import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

from submission import agent
from submission import agent as agent_module
from submission.agent import ActorCritic, DoomAgent, HIDDEN_SIZE
from submission.actions import ACTION_ARRAYS, ACTION_NAMES, BUTTONS


def empty_observation():
    return {
        "rays": {
            "distance": np.ones(64, dtype=np.float32),
            "hit_type": np.zeros(64, dtype=np.int64),
        },
        "vitals": np.zeros(17, dtype=np.float32),
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros(32, dtype=np.int64),
            "mask": np.zeros(32, dtype=np.int8),
        },
    }


def test_actor_critic_handles_all_entity_slots_empty():
    model = ActorCritic()
    obs = empty_observation()
    inputs = {
        "rays": {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in obs["rays"].items()
        },
        "vitals": torch.as_tensor(obs["vitals"]).unsqueeze(0),
        "visible_entities": {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in obs["visible_entities"].items()
        },
    }
    logits, value, hidden = model(inputs, torch.zeros(1, HIDDEN_SIZE))
    assert logits.shape == (1, len(ACTION_NAMES))
    assert value.shape == (1,)
    assert hidden.shape == (1, HIDDEN_SIZE)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_agent_action_matches_multibinary_contract(monkeypatch):
    from doom_env.spaces import action_space

    monkeypatch.delenv("DOOM_INFERENCE_MODE", raising=False)
    monkeypatch.delenv("DOOM_SAMPLING_SEED", raising=False)
    agent = DoomAgent(
        observation_space=None,
        action_space=action_space(),
        device="cpu",
    )
    assert agent.format_version == 1
    action = agent.act(empty_observation())
    assert action.shape == (9,)
    assert action.dtype == np.int8
    assert set(np.unique(action)).issubset({0, 1})
    agent.reset()
    assert torch.count_nonzero(agent.hidden) == 0
    assert agent.inference_mode == "greedy"


def test_categorical_actions_fit_official_multibinary_space():
    from doom_env.spaces import BUTTONS as CONTRACT_BUTTONS, action_space

    assert BUTTONS == CONTRACT_BUTTONS
    space = action_space()
    assert all(space.contains(action) for action in ACTION_ARRAYS)


def test_agent_loads_version_two_categorical_checkpoint(tmp_path, monkeypatch):
    from doom_env.spaces import action_space

    checkpoint_path = tmp_path / "categorical.pt"
    model = ActorCritic(len(ACTION_NAMES))
    torch.save(
        {
            "format_version": 2,
            "action_names": list(ACTION_NAMES),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setattr(agent_module, "CHECKPOINT_PATH", checkpoint_path)
    agent = DoomAgent(None, action_space(), "cpu")
    action = agent.act(empty_observation())
    agent.reset()
    greedy_action, probabilities = agent.act_with_probabilities(
        empty_observation(), sample=False
    )
    agent.reset()
    with torch.random.fork_rng():
        torch.manual_seed(23)
        sampled_action, sampled_probabilities = agent.act_with_probabilities(
            empty_observation(), sample=True
        )
        agent.reset()
        torch.manual_seed(23)
        repeated_action, _ = agent.act_with_probabilities(
            empty_observation(), sample=True
        )

    assert agent.format_version == 2
    assert action_space().contains(action)
    np.testing.assert_array_equal(action, greedy_action)
    assert action_space().contains(sampled_action)
    assert probabilities.shape == (len(ACTION_NAMES),)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(sampled_probabilities.sum(), 1.0)
    np.testing.assert_array_equal(sampled_action, repeated_action)


def test_agent_local_sampling_is_reproducible_across_reset(tmp_path, monkeypatch):
    from doom_env.spaces import action_space

    checkpoint_path = tmp_path / "sampled.pt"
    model = ActorCritic(len(ACTION_NAMES))
    torch.save(
        {
            "format_version": 2,
            "action_names": list(ACTION_NAMES),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setattr(agent_module, "CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setenv("DOOM_INFERENCE_MODE", "sampled")
    monkeypatch.setenv("DOOM_SAMPLING_SEED", "19")
    agent = DoomAgent(None, action_space(), "cpu")

    def action_sequence(seed):
        agent.reset(seed=seed)
        return [tuple(agent.act(empty_observation())) for _ in range(12)]

    first = action_sequence(2026)
    repeated = action_sequence(2026)
    different_seed = action_sequence(2027)

    assert agent.inference_mode == "sampled"
    assert first == repeated
    assert first != different_seed
    assert all(action_space().contains(np.asarray(action)) for action in first)
