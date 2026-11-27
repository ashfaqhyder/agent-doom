"""Compatible categorical actions shared by training and inference."""

from __future__ import annotations

import numpy as np

BUTTONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
    "SELECT_NEXT_WEAPON",
)
NUM_BUTTONS = len(BUTTONS)
ACTION_COMBINATIONS = (
    (),
    ("MOVE_FORWARD",),
    ("MOVE_BACKWARD",),
    ("MOVE_LEFT",),
    ("MOVE_RIGHT",),
    ("TURN_LEFT",),
    ("TURN_RIGHT",),
    ("MOVE_FORWARD", "TURN_LEFT"),
    ("MOVE_FORWARD", "TURN_RIGHT"),
    ("ATTACK",),
    ("MOVE_FORWARD", "ATTACK"),
    ("MOVE_LEFT", "ATTACK"),
    ("MOVE_RIGHT", "ATTACK"),
    ("USE",),
    ("SELECT_NEXT_WEAPON",),
)
ACTION_NAMES = tuple("+".join(buttons) if buttons else "NOOP" for buttons in ACTION_COMBINATIONS)
_BUTTON_INDEX = {button: index for index, button in enumerate(BUTTONS)}

ACTION_ARRAYS = np.zeros((len(ACTION_COMBINATIONS), NUM_BUTTONS), dtype=np.int8)
for _action_index, _buttons in enumerate(ACTION_COMBINATIONS):
    for _button in _buttons:
        ACTION_ARRAYS[_action_index, _BUTTON_INDEX[_button]] = 1
ACTION_ARRAYS.setflags(write=False)
