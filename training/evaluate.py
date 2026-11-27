"""Evaluate one checkpoint using DoomAgent's validation-time action path."""

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
from doom_env.spaces import BUTTONS
from submission.actions import ACTION_ARRAYS, ACTION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("training/evaluation_results.csv")
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def _counter_delta(start: Any, end: Any) -> int:
    return max(0, int(end) - int(start))


def _sector_for_point(
    sectors: list[tuple[int, np.ndarray]], x: float, y: float
) -> int | None:
    for sector_id, segments in sectors:
        inside = False
        for x1, y1, x2, y2 in segments:
            if (y1 > y) != (y2 > y):
                crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < crossing_x:
                    inside = not inside
        if inside:
            return sector_id
    return None


def _action_name(action: np.ndarray) -> str:
    action_tuple = tuple(int(value) for value in action)
    for index, known_action in enumerate(ACTION_ARRAYS):
        if action_tuple == tuple(int(value) for value in known_action):
            return ACTION_NAMES[index]
    pressed = [BUTTONS[index] for index, value in enumerate(action_tuple) if value]
    return "+".join(pressed) if pressed else "NOOP"


def _top_actions(counts: Counter[str], limit: int = 3) -> str:
    return ";".join(f"{name}:{count}" for name, count in counts.most_common(limit))


def evaluate_episode(agent, seed: int, episode_ticks: int) -> dict[str, Any]:
    env = make_env(
        str(ROOT / "maps" / "two_rooms.wad"),
        episode_ticks=episode_ticks,
    )
    try:
        observation, _ = env.reset(seed=seed)
        agent.reset()
        start_state = env.privileged_state()
        sectors = env.sector_polygons()
        action_counts: Counter[str] = Counter()
        visited_sectors: set[int] = set()
        total_return = 0.0
        decisions = 0
        initial_sector = _sector_for_point(
            sectors, float(start_state["x"]), float(start_state["y"])
        )
        if initial_sector is not None:
            visited_sectors.add(initial_sector)
        while True:
            action = agent.act(observation)
            action_counts[_action_name(action)] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            decisions += 1
            state = env.privileged_state()
            sector = _sector_for_point(sectors, float(state["x"]), float(state["y"]))
            if sector is not None:
                visited_sectors.add(sector)
            if terminated or truncated:
                break

        final_state = env.privileged_state()
        start_keys = np.asarray(start_state["keys"], dtype=np.int8)
        final_keys = np.asarray(final_state["keys"], dtype=np.int8)
        return {
            "seed": seed,
            "exit_reached": bool(info["exit_reached"]),
            "episode_return": total_return,
            "decisions": decisions,
            "unique_sectors": len(visited_sectors),
            "key_pickups": int(np.count_nonzero(final_keys > start_keys)),
            "item_pickups": _counter_delta(start_state["items"], final_state["items"]),
            "kills": _counter_delta(start_state["kills"], final_state["kills"]),
            "most_frequent_actions": _top_actions(action_counts),
            "_action_counts": action_counts,
        }
    finally:
        env.close()


def write_results(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    fields = (
        "checkpoint",
        "seed",
        "exit_reached",
        "episode_return",
        "decisions",
        "unique_sectors",
        "key_pickups",
        "item_pickups",
        "kills",
        "most_frequent_actions",
    )
    needs_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows
        )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    os.environ["DOOM_CHECKPOINT_PATH"] = str(checkpoint_path)
    from submission.agent import DoomAgent

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent = DoomAgent(None, None, str(device))
    print(f"checkpoint={agent.checkpoint_path}")
    print(f"training_status={checkpoint.get('training_status', 'unspecified')}")
    print(
        "seed,exit_reached,episode_return,decisions,unique_sectors,"
        "key_pickups,item_pickups,kills,most_frequent_actions"
    )

    rows = []
    all_action_counts: Counter[str] = Counter()
    for seed in args.seeds:
        row = evaluate_episode(agent, seed, args.episode_ticks)
        row["checkpoint"] = str(checkpoint_path)
        rows.append(row)
        all_action_counts.update(row["_action_counts"])
        print(
            f"{row['seed']},{int(row['exit_reached'])},{row['episode_return']:.3f},"
            f"{row['decisions']},{row['unique_sectors']},{row['key_pickups']},"
            f"{row['item_pickups']},{row['kills']},{row['most_frequent_actions']}"
        )

    write_results(args.output, rows, args.append)
    exits = sum(bool(row["exit_reached"]) for row in rows)
    print(
        f"summary: exits={exits}/{len(rows)} "
        f"mean_return={mean(row['episode_return'] for row in rows):.3f} "
        f"mean_decisions={mean(row['decisions'] for row in rows):.1f} "
        f"mean_unique_sectors={mean(row['unique_sectors'] for row in rows):.2f} "
        f"key_pickups={sum(row['key_pickups'] for row in rows)} "
        f"item_pickups={sum(row['item_pickups'] for row in rows)} "
        f"kills={sum(row['kills'] for row in rows)} "
        f"top_actions={_top_actions(all_action_counts, 5)}"
    )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    print(f"results_csv={output_path.resolve()}")


if __name__ == "__main__":
    main()
