import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submission.actions import ACTION_ARRAYS
from training.evaluate_submission_path import _action_name, _select_action


class RecordingAgent:
    def __init__(self):
        self.observations = []

    def act(self, observation):
        self.observations.append(observation)
        return ACTION_ARRAYS[0].copy()


def test_greedy_and_sampled_modes_use_doom_agent_act():
    observation = {"test": "observation"}
    for mode in ("greedy", "sampled"):
        agent = RecordingAgent()
        action = _select_action(agent, observation, mode, np.random.default_rng(1))

        assert agent.observations == [observation]
        np.testing.assert_array_equal(action, ACTION_ARRAYS[0])


def test_random_mode_uses_seeded_shared_action_menu():
    first_rng = np.random.default_rng(21)
    second_rng = np.random.default_rng(21)

    first = [_select_action(None, {}, "random", first_rng) for _ in range(10)]
    second = [_select_action(None, {}, "random", second_rng) for _ in range(10)]

    assert all(any(np.array_equal(action, valid) for valid in ACTION_ARRAYS) for action in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_legacy_multibinary_action_outside_categorical_menu_is_labeled():
    action = np.asarray([1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    assert _action_name(action) == "MOVE_FORWARD+MOVE_LEFT"
