"""Compare DoomAgent.act() policies and a uniform-random control on paired episodes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.actions import ACTION_ARRAYS, ACTION_NAMES, BUTTONS
from training.evaluate import _sector_for_point, _top_actions

TWO_ROOMS = (ROOT / "maps" / "two_rooms.wad").resolve()
DOOR_SECTOR = 2
DESTINATION_SECTOR = 1
FULLY_OPEN_CEILING = 120.0
DEFAULT_CHECKPOINTS = (
    Path("submission/weights/ppo_health_exit_final.pt"),
    Path("submission/weights/ppo_open_door_varied_smoke.pt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wad", type=Path, default=Path("maps/two_rooms.wad"))
    parser.add_argument("--map-name", default="MAP01")
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--greedy-checkpoints", type=Path, nargs="*", default=())
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(21, 31)))
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--sampling-seed", type=int, default=71000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/submission_path_comparison.csv"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episode_ticks < 1:
        parser.error("--episode-ticks must be positive")
    if not args.seeds:
        parser.error("at least one seed is required")
    return args


def _resolve(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist: {resolved}")
    return resolved


def _action_name(action: np.ndarray) -> str:
    matches = np.flatnonzero(np.all(ACTION_ARRAYS == action, axis=1))
    if len(matches) == 1:
        return ACTION_NAMES[int(matches[0])]
    if action.shape != (len(BUTTONS),):
        raise ValueError(f"DoomAgent.act returned an invalid action shape: {action.shape}")
    pressed_buttons = [BUTTONS[index] for index, pressed in enumerate(action) if pressed]
    return "+".join(pressed_buttons) if pressed_buttons else "NOOP"


def _select_action(agent, observation, mode: str, random_generator) -> np.ndarray:
    if mode in ("greedy", "sampled"):
        if agent is None:
            raise ValueError(f"{mode} inference requires a DoomAgent")
        return agent.act(observation)
    if mode == "random":
        return ACTION_ARRAYS[int(random_generator.integers(len(ACTION_ARRAYS)))].copy()
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def _evaluate_episode(
    agent,
    env,
    *,
    checkpoint: str,
    mode: str,
    seed: int,
    sampling_seed: int,
    random_seed: int,
    map_name: str,
    track_two_rooms_crossing: bool,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    if agent is not None:
        agent.reset(seed=sampling_seed + seed)
    random_generator = np.random.default_rng(random_seed + seed)
    sectors = env.sector_polygons()
    initial_state = env.privileged_state()
    initial_position = (float(initial_state["x"]), float(initial_state["y"]))
    previous_position = initial_position
    initial_health = float(initial_state["health"])
    previous_health = initial_health
    visited_sectors: set[int] = set()
    previous_sector = _sector_for_point(sectors, *initial_position)
    if previous_sector is not None:
        visited_sectors.add(previous_sector)

    crossed = previous_sector == DESTINATION_SECTOR
    first_crossing_decision: int | None = None
    sector_transitions = 0
    peak_door_ceiling = 0.0
    if track_two_rooms_crossing:
        peak_door_ceiling = float(initial_state["sector_ceiling"][DOOR_SECTOR])

    action_counts: Counter[str] = Counter()
    official_return = 0.0
    total_health_lost = 0.0
    distance_travelled = 0.0
    movement_decisions = 0
    decisions = 0
    time_to_exit: int | None = None
    info: dict[str, Any] = {}
    terminated = truncated = False

    while not (terminated or truncated):
        action = _select_action(agent, observation, mode, random_generator)
        if not env.action_space.contains(action):
            raise ValueError(f"invalid action from {mode} policy: {action}")
        action_counts[_action_name(action)] += 1
        movement_decisions += int(np.any(action[:4]))

        observation, reward, terminated, truncated, info = env.step(action)
        official_return += float(reward)
        decisions += 1
        state = env.privileged_state()
        position = (float(state["x"]), float(state["y"]))
        distance_travelled += float(
            np.hypot(position[0] - previous_position[0], position[1] - previous_position[1])
        )
        previous_position = position
        health = float(state["health"])
        total_health_lost += max(0.0, previous_health - health)
        previous_health = health
        sector = _sector_for_point(sectors, *position)
        if sector is not None:
            visited_sectors.add(sector)
            if previous_sector is not None and sector != previous_sector:
                sector_transitions += 1
            previous_sector = sector

        if track_two_rooms_crossing:
            peak_door_ceiling = max(
                peak_door_ceiling,
                float(state["sector_ceiling"][DOOR_SECTOR]),
            )
            if not crossed and sector == DESTINATION_SECTOR:
                crossed = True
                first_crossing_decision = decisions
        if time_to_exit is None and bool(info.get("exit_reached", False)):
            time_to_exit = decisions

    final_state = env.privileged_state()
    final_position = (float(final_state["x"]), float(final_state["y"]))
    return {
        "wad": str(env.wad),
        "map_name": map_name,
        "checkpoint": checkpoint,
        "mode": mode,
        "seed": seed,
        "sampling_seed": sampling_seed + seed if mode == "sampled" else "",
        "random_seed": random_seed + seed if mode == "random" else "",
        "episode_ticks": env.episode_ticks,
        "exit_reached": bool(info.get("exit_reached", False)),
        "died": bool(final_state["dead"]),
        "official_return": official_return,
        "time_to_exit_decisions": time_to_exit,
        "decisions": decisions,
        "destination_crossed": crossed if track_two_rooms_crossing else "",
        "first_crossing_decision": first_crossing_decision,
        "door_fully_opened": (
            peak_door_ceiling >= FULLY_OPEN_CEILING if track_two_rooms_crossing else ""
        ),
        "peak_door_ceiling": peak_door_ceiling if track_two_rooms_crossing else "",
        "unique_sectors": len(visited_sectors),
        "sector_transitions": sector_transitions,
        "kills": int(final_state["kills"] - int(initial_state["kills"])),
        "initial_health": initial_health,
        "final_health": float(final_state["health"]),
        "total_health_lost": total_health_lost,
        "distance_travelled": distance_travelled,
        "displacement_from_spawn": float(
            np.hypot(
                final_position[0] - initial_position[0],
                final_position[1] - initial_position[1],
            )
        ),
        "movement_fraction": movement_decisions / max(decisions, 1),
        "most_frequent_actions": _top_actions(action_counts, 5),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys()) if rows else ()
    with resolved.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return resolved.resolve()


def main() -> None:
    args = parse_args()
    wad_path = _resolve(args.wad)
    is_original_map = wad_path == TWO_ROOMS and args.map_name.upper() == "MAP01"
    os.environ["DOOM_SAMPLING_SEED"] = str(args.sampling_seed)
    device = torch.device(args.device)
    from submission import agent as agent_module

    policy_specs = [
        *((checkpoint_path, "sampled") for checkpoint_path in args.checkpoints),
        *((checkpoint_path, "greedy") for checkpoint_path in args.greedy_checkpoints),
    ]
    if not policy_specs:
        raise ValueError("provide at least one --checkpoints or --greedy-checkpoints path")

    rows: list[dict[str, Any]] = []
    env = make_env(
        str(wad_path),
        args.map_name,
        episode_ticks=args.episode_ticks,
        render_buffers=False,
    )
    try:
        for configured_path, mode in policy_specs:
            checkpoint_path = _resolve(configured_path)
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            format_version = checkpoint_data.get("format_version")
            if format_version not in (1, 2):
                raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
            if format_version == 2 and checkpoint_data.get("action_names") != list(ACTION_NAMES):
                raise ValueError(f"Expected a compatible categorical checkpoint: {checkpoint_path}")
            if mode == "sampled" and format_version != 2:
                raise ValueError(
                    f"Sampled inference requires a format-version-2 checkpoint: {checkpoint_path}"
                )
            os.environ["DOOM_INFERENCE_MODE"] = mode
            agent_module.CHECKPOINT_PATH = checkpoint_path
            agent = agent_module.DoomAgent(
                env.observation_space, env.action_space, str(device)
            )
            if agent.inference_mode != mode:
                raise RuntimeError(f"DoomAgent entered {agent.inference_mode!r}, expected {mode!r}")
            print(
                f"checkpoint={checkpoint_path} mode={mode} "
                f"sampling_seed={args.sampling_seed if mode == 'sampled' else 'n/a'} "
                f"training_status={checkpoint_data.get('training_status', 'unspecified')}"
            )
            for seed in args.seeds:
                row = _evaluate_episode(
                    agent,
                    env,
                    checkpoint=str(checkpoint_path),
                    mode=mode,
                    seed=seed,
                    sampling_seed=args.sampling_seed,
                    random_seed=args.random_seed,
                    map_name=args.map_name,
                    track_two_rooms_crossing=is_original_map,
                )
                rows.append(row)
                print(
                    f"episode checkpoint={checkpoint_path.name} mode={mode} seed={seed} "
                    f"exit={int(row['exit_reached'])} died={int(row['died'])} "
                    f"return={row['official_return']:.3f} "
                    f"crossing={row['destination_crossed']} "
                    f"sectors={row['unique_sectors']} "
                    f"kills={row['kills']} decisions={row['decisions']} "
                    f"actions={row['most_frequent_actions']}"
                )

        for seed in args.seeds:
            row = _evaluate_episode(
                None,
                env,
                checkpoint="uniform_random",
                mode="random",
                seed=seed,
                sampling_seed=args.sampling_seed,
                random_seed=args.random_seed,
                map_name=args.map_name,
                track_two_rooms_crossing=is_original_map,
            )
            rows.append(row)
            print(
                f"episode checkpoint=uniform_random mode=random seed={seed} "
                f"exit={int(row['exit_reached'])} died={int(row['died'])} "
                f"return={row['official_return']:.3f} "
                f"crossing={row['destination_crossed']} "
                f"sectors={row['unique_sectors']} "
                f"kills={row['kills']} decisions={row['decisions']} "
                f"actions={row['most_frequent_actions']}"
            )
    finally:
        env.close()

    for checkpoint, mode in sorted({(row["checkpoint"], row["mode"]) for row in rows}):
        group = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint and row["mode"] == mode
        ]
        crossings = sum(bool(row["destination_crossed"]) for row in group)
        crossing_summary = f"{crossings}/{len(group)}" if is_original_map else "n/a"
        print(
            f"summary checkpoint={Path(checkpoint).name} mode={mode} "
            f"exits={sum(row['exit_reached'] for row in group)}/{len(group)} "
            f"deaths={sum(row['died'] for row in group)}/{len(group)} "
            f"mean_return={mean(row['official_return'] for row in group):.3f} "
            f"crossings={crossing_summary} "
            f"mean_unique_sectors={mean(row['unique_sectors'] for row in group):.2f} "
            f"mean_sector_transitions={mean(row['sector_transitions'] for row in group):.2f}"
        )
    print(f"results_csv={_write_rows(args.output, rows)}")


if __name__ == "__main__":
    main()
