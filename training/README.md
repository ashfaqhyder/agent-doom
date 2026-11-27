# Training

The training pipeline uses the official `doom_env` wrapper and observations. Training-only
diagnostics and reward shaping may read `privileged_state()`; that state is never passed to
`DoomAgent` or added to its observations.

## Agent and checkpoints

`submission/agent.py` encodes rays, vitals, and visible entities (using the official entity mask),
then passes the fused representation through a GRU and actor-critic heads. The current categorical
policy uses the compatible button combinations in `submission/actions.py`; the shared mapping
returns nine-bit vectors in the official `MultiBinary(9)` order.

Checkpoint format 1 remains supported for the original nine-logit Bernoulli actor. Format 2 stores
the categorical action names and maps its selected class through the shared action table.
`ppo_smoke.pt` is retained as the default format-1 checkpoint. New categorical training initializes
the shared encoder, GRU, and critic from it while creating a fresh categorical actor head.

`train_ppo.py` samples categorical actions, saves their indices and log probabilities, and replays
ordered rollouts from their starting GRU state during PPO updates. It clears recurrent state at
episode boundaries and bootstraps time-limit truncations without carrying GAE into another
episode. The optional first-visit reward is training-only: it grants a capped bonus per newly
visited 64-map-unit position cell.

## Commands

From the repository root in PowerShell, run the focused CPU smoke training without overwriting the
legacy comparison checkpoint:

```powershell
uv run python training\train_ppo.py --seed 0 --updates 2 --rollout-length 16 --epochs 2 --checkpoint-output submission\weights\ppo_categorical_smoke.pt --device cpu
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py
```

The trainer reports 32 environment decisions and two PPO updates. The validator uses
`submission/weights/ppo_smoke.pt` by default. To validate a different checkpoint, set
`DOOM_CHECKPOINT_PATH` to a path relative to `submission/` (or an absolute path).

## Evaluation

`evaluate.py` calls `DoomAgent.act()` for every decision, matching validation-time action selection.
It reports checkpoint identity, exits, official episode return, decisions, key/item pickups, kills,
unique sectors visited (from the wrapper's scorer geometry), and most frequent actions. Privileged
state is used only by the evaluator to measure those outcomes. The default map is `two_rooms.wad`,
the default tick cap is 2,100, and the default seeds are 0–9.

## Isolated map diagnostic

Run the privileged-state route diagnostic separately from the submitted agent:

```powershell
uv run python training\diagnose_two_rooms.py
```

On seed 0 it verified that the manual door opens after pressing `USE` while facing east near
`(421, 341)`. The route crosses that door, goes east of `x=704` to bypass the horizontal wall at
`y=256`, then crosses the map's exit trigger. The diagnostic uses fixed waypoints and privileged
coordinates; none of that control logic is part of `DoomAgent`.

## Categorical PPO comparison

Both experiments used the same initialization (`ppo_smoke.pt` shared weights and seed 0), map,
default environment reward, CPU, and 5,120 decisions (40 updates × 128 rollout decisions). The
exploration run adds `0.02` for each first-visited 64-unit cell, capped at `0.5` per episode.

```powershell
uv run python training\train_ppo.py --seed 0 --updates 40 --rollout-length 128 --epochs 4 --checkpoint-output submission\weights\ppo_cat_default_40x128.pt --device cpu --exploration-reward-per-cell 0
uv run python training\train_ppo.py --seed 0 --updates 40 --rollout-length 128 --epochs 4 --checkpoint-output submission\weights\ppo_cat_explore_40x128.pt --device cpu --exploration-reward-per-cell 0.02 --exploration-reward-cap 0.5
```

Both runs changed model weights and saved format-2 checkpoints. The default-reward run collected
official return `0.0` and shaping reward `0.0`. The exploration run collected official return
`-0.9` (a `-1.0` death penalty and `+0.1` kill reward) and shaping reward `+4.48`. Shaping is not
included in official episode returns.

Evaluate the retained baseline and both candidates on identical seeds and tick caps:

```powershell
uv run python training\evaluate.py --checkpoint submission\weights\ppo_smoke.pt --seeds 0 1 2 3 4 5 6 7 8 9 --episode-ticks 2100 --output training\categorical_comparison.csv
uv run python training\evaluate.py --checkpoint submission\weights\ppo_cat_default_40x128.pt --seeds 0 1 2 3 4 5 6 7 8 9 --episode-ticks 2100 --output training\categorical_comparison.csv --append
uv run python training\evaluate.py --checkpoint submission\weights\ppo_cat_explore_40x128.pt --seeds 0 1 2 3 4 5 6 7 8 9 --episode-ticks 2100 --output training\categorical_comparison.csv --append
```

Results (all 10 episodes per checkpoint used 525 decisions):

| Checkpoint | Exits | Mean official return | Mean unique sectors | Most frequent action |
| --- | ---: | ---: | ---: | --- |
| `ppo_smoke.pt` (legacy) | 0/10 | 0.000 | 1 | `MOVE_LEFT` (523/525); `MOVE_FORWARD+MOVE_LEFT` (2/525) |
| `ppo_cat_default_40x128.pt` | 0/10 | 0.000 | 1 | `MOVE_BACKWARD` (525/525) |
| `ppo_cat_explore_40x128.pt` | 0/10 | 0.000 | 1 | `MOVE_LEFT+ATTACK` (525/525) |

Seed-0 movement traces show where both candidates stop. The default-reward candidate repeatedly
chooses `MOVE_BACKWARD`, reaches approximately `(80, 416)` by decision 50, then remains against the
west wall. The exploration candidate repeatedly chooses `MOVE_LEFT+ATTACK`, moves from `(160,416)`
to approximately `(160,496)` by decision 50, then remains against the south wall. Neither reaches
the door or leaves the starting sector. The baseline also stays in one sector.

The exploration candidate was explicitly validated with:

```powershell
$env:DOOM_CHECKPOINT_PATH = "weights\ppo_cat_explore_40x128.pt"
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py
```

The official validator passed: 525-step episode, 0.6 seconds agent time, 1.2 ms p50 step latency.
It reported four environment/worktree warnings (untracked submission files, uncommitted agent
changes, CPU pinning unavailable, and peak RAM unavailable). Neither candidate improved measured
gameplay, so `ppo_smoke.pt` remains the default. The original comparison from the previous
Bernoulli milestone remains in `evaluation_results.csv`; the categorical results are in
`categorical_comparison.csv`.

## Greedy and sampled policy comparison

`evaluate_policy_modes.py` compares deterministic greedy selection with categorical sampling for
the same checkpoints, map, 2,100-tic cap, and seeds 0–9. Sampling explicitly resets Torch's RNG to
`sampling_seed + episode_seed` (default sampling seed 1337). It uses the same encoder, GRU, and
action mapping as `DoomAgent`; `DoomAgent.act()` remains greedy by default. The evaluator alone
uses privileged state to measure door heights, sectors, and displacement. Seed 0 is the
representative probability trace; five action distributions around the first 20-decision
low-displacement window are printed and saved.

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_cat_default_40x128.pt submission\weights\ppo_cat_explore_40x128.pt --modes greedy sampled --seeds 0 1 2 3 4 5 6 7 8 9 --episode-ticks 2100 --sampling-seed 1337 --trace-seed 0 --output training\policy_mode_comparison.csv --trace-output training\stall_action_probabilities.csv --device cpu
```

“Door opened” means the manual-door sector ceiling rose above its closed height; “fully open”
means it reached 120 map units. All conditions used 525 decisions per episode.

| Checkpoint | Mode | Exits | Mean official return | Opened / fully open | Mean sectors | Mean displacement | Dominant actions (counts across 10 episodes) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo_cat_default_40x128.pt` | greedy | 0/10 | 0.000 | 0/0 | 1.0 | 80.0 | `MOVE_BACKWARD` 5,250 |
| `ppo_cat_default_40x128.pt` | sampled | 0/10 | 0.000 | 0/0 | 1.0 | 158.2 | `MOVE_BACKWARD` 1,562; `TURN_RIGHT` 778; `TURN_LEFT` 641; `USE` 517; `ATTACK` 477 |
| `ppo_cat_explore_40x128.pt` | greedy | 0/10 | 0.000 | 0/0 | 1.0 | 80.0 | `MOVE_LEFT+ATTACK` 5,250 |
| `ppo_cat_explore_40x128.pt` | sampled | 0/10 | 0.000 | 0/0 | 1.0 | 189.1 | `MOVE_LEFT+ATTACK` 2,005; `TURN_LEFT` 1,203; `MOVE_BACKWARD` 920 |

Neither greedy nor sampled evaluation opened the door. For seed 0, the default-reward policy stalled
near `(80, 416)`; its greedy probabilities were `MOVE_BACKWARD=0.300`, `TURN_RIGHT=0.148`,
`TURN_LEFT=0.119`, and `USE=0.098`. The exploration policy stalled near `(160, 496)` with
`MOVE_LEFT+ATTACK=0.384`, `TURN_LEFT=0.241`, and `MOVE_BACKWARD=0.170`. Full traces are in
`stall_action_probabilities.csv`.

## Bounded door-opening curriculum

Since both action modes failed to open the door, `train_door_curriculum.py` creates temporary
training-only WAD variants. Geometry and game rules are unchanged; only the player start varies
among six poses around the manual door:

```text
(400,336,350), (416,336,0), (424,352,10),
(400,368,0), (416,368,350), (424,336,0)
```

The curriculum uses 256-tic episodes and grants `+1` once when door sector 2 reaches its fully open
ceiling of 120, then ends that training subtask episode. This reward and the privileged sector
height are training-only.

```powershell
uv run python training\train_door_curriculum.py --seed 13 --updates 24 --rollout-length 64 --epochs 4 --episode-ticks 256 --checkpoint-initial submission\weights\ppo_cat_default_40x128.pt --checkpoint-output submission\weights\ppo_door_curriculum.pt --device cpu
```

The bounded run collected 1,536 decisions, 24 PPO updates, 8 door-opening bonuses, and `0.0`
official training reward. Each of the six start poses was separately verified to open the door
after `USE`; the source map remains unchanged.

Evaluate the resulting checkpoint from the original map spawn, using the same seeds and tick cap:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_door_curriculum.pt --modes greedy sampled --seeds 0 1 2 3 4 5 6 7 8 9 --episode-ticks 2100 --sampling-seed 1337 --trace-seed 0 --output training\door_curriculum_eval.csv --trace-output training\door_curriculum_stall_probabilities.csv --device cpu
```

The curriculum checkpoint achieved 0/10 exits and mean official return `0.000` in both modes.
Greedy opened no doors and stayed in one sector with mean displacement 80.0. Sampling opened the
door fully in 1/10 episodes, but still reached no exits or additional sectors (mean displacement
204.1). Its most frequent sampled actions were `MOVE_BACKWARD` (954), `USE` (738), and
`MOVE_LEFT+ATTACK` (628). The sampled opening is measurable but does not transfer to the default
greedy path, which remains stuck at the west wall.

The candidate was explicitly validated using:

```powershell
$env:DOOM_CHECKPOINT_PATH = "weights\ppo_door_curriculum.pt"
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py
```

Validation passed: 525 steps, 0.9 seconds agent time, and 1.6 ms p50 latency, with four
environment/worktree warnings. The candidate was not promoted: its default greedy behavior still
has 0/10 door openings and 0/10 exits. `ppo_smoke.pt` remains the default until a checkpoint shows
progress under the default inference path.

## Crossing-focused curriculum

The crossing detector uses the wrapper's sector polygons: sector 2 at `(480, 352)` is the
doorway and does not count; sector 1 at `(560, 352)` is inside the destination room and does count.
The test in `tests/test_crossing_curriculum.py` verifies both points and steps through opening and
crossing, confirming that opening no longer ends the training episode. Training adds a one-time
`+0.1` full-door-opening bonus and a larger `+1.0` first-entry crossing bonus. The wrapper's
privileged position and sector geometry are confined to this training wrapper; policy observations
remain unchanged. The policy chooses USE through the shared categorical action space.

Run the bounded two-stage training from the prior door checkpoint:

```powershell
uv run python training\train_crossing_curriculum.py --seed 29 --stage1-updates 16 --stage2-updates 24 --rollout-length 128 --epochs 4 --episode-ticks 1400 --checkpoint-initial submission\weights\ppo_door_curriculum.pt --stage1-checkpoint submission\weights\ppo_crossing_stage1.pt --checkpoint-output submission\weights\ppo_crossing_curriculum.pt --device cpu
```

Stage 1 used six near-door starts (2,048 decisions); stage 2 used those starts plus four farther
starts, including the original `(160, 416, 0)` spawn (3,072 decisions). The total was 5,120
decisions and 40 PPO updates. Training recorded 9 full door openings, 0 destination crossings,
`+0.9` total shaping reward, and `+0.4` official reward. The final candidate is saved separately
as `ppo_crossing_curriculum.pt`; the stage-1 checkpoint is also preserved.

Compare both curriculum checkpoints from the original map spawn with the same seeds, 2,100-tic
cap, and greedy and seeded-sampled action selection:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_door_curriculum.pt submission\weights\ppo_crossing_curriculum.pt --modes greedy sampled --seeds 0 1 2 3 4 5 6 7 8 9 --episode-ticks 2100 --sampling-seed 1337 --trace-seed 0 --output training\crossing_curriculum_eval.csv --trace-output training\crossing_curriculum_stall_probabilities.csv --device cpu
```

The evaluator uses `DoomAgent`'s normal greedy path or its evaluation-only seeded sampler. Door,
sector, and crossing metrics use privileged state only inside the evaluator. `door opened` counts
any rise in the door ceiling; `fully open` means the ceiling reached 120. Evaluation uses only the
wrapper's official reward:

| Checkpoint | Mode | Exits | Door opened | Fully open | Crossings | Mean official return | Mean sectors | Mean displacement | Most frequent actions across 10 episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo_door_curriculum.pt` | greedy | 0/10 | 0/10 | 0/10 | 0/10 | 0.000 | 1.00 | 80.0 | `MOVE_BACKWARD` 5,250 |
| `ppo_door_curriculum.pt` | sampled | 0/10 | 1/10 | 1/10 | 0/10 | 0.000 | 1.00 | 204.1 | `MOVE_BACKWARD` 954; `USE` 738; `MOVE_LEFT+ATTACK` 628 |
| `ppo_crossing_curriculum.pt` | greedy | 0/10 | 0/10 | 0/10 | 0/10 | 0.000 | 1.00 | 80.0 | `MOVE_BACKWARD` 5,250 |
| `ppo_crossing_curriculum.pt` | sampled | 0/10 | 4/10 | 4/10 | 2/10 | -0.100 | 1.40 | 253.5 | `MOVE_BACKWARD` 770; `MOVE_FORWARD+ATTACK` 556; `TURN_RIGHT` 500 |

Seeded sampling showed limited progress (2/10 crossings), but neither checkpoint crossed in greedy
mode. In the new candidate's sampled seed-0 trace, movement stalled near `(420, 418)` by decision
102; the selected action was `SELECT_NEXT_WEAPON` with probability `0.039`, and the top action
probability was `MOVE_BACKWARD=0.156`. Since intended deployment remains greedy and the candidate
had no greedy crossings, `ppo_smoke.pt` remains the default.

The candidate was explicitly loaded and validated with:

```powershell
$env:DOOM_CHECKPOINT_PATH = "weights\ppo_crossing_curriculum.pt"
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py
```

`DoomAgent` confirmed the absolute checkpoint path as
`submission/weights/ppo_crossing_curriculum.pt` and format version 2. Validation **passed**:
525-step episode, 0.6 seconds agent time, 1.2 ms p50 latency. It reported four environment
warnings: untracked submission files, uncommitted `submission/agent.py`, CPU pinning unavailable,
and peak RAM unavailable on Windows.

## Starter provenance

The official starter is
[`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template).
At verification time its `main` revision was
[`63b90c441ca92bcdf043de9ee3930ac6370a44bc`](https://github.com/Reinforce-SST/devforge-doom-template/commit/63b90c441ca92bcdf043de9ee3930ac6370a44bc).
Every tracked file under `env/` and `validate.py` has the same Git blob ID as that upstream
revision. The local repository's `c9504683e2c50154f2afcb78b4b3323367bed7a9` is a generated
single-commit snapshot, not an upstream commit; the upstream revision is recorded here explicitly.

## Crossing replay and mixed-doorway follow-up

The two sampled successes recorded in `crossing_curriculum_eval.csv` were replayed through
`CrossingCurriculumEnv` with the same checkpoint, seeds, sampling seed, original map, and tick cap:

```powershell
uv run python training\replay_crossing_episodes.py --checkpoint submission\weights\ppo_crossing_curriculum.pt --episode-ticks 2100 --sampling-seed 1337 --window 5 --output training\crossing_replay_trace.csv
```

Both replayed transitions earned exactly one configured `+1.0` training crossing reward. Seed 3
opened at decision 501 (`TURN_LEFT`, while in sector 2); it entered sector 2 at decision 498 and
the destination sector 1 at decision 513 (`MOVE_FORWARD+ATTACK`). Seed 5 opened at decision 178
(`ATTACK`, while still in sector 0); it entered sector 2 at decision 467 (`NOOP`) and sector 1 at
decision 476 (`MOVE_FORWARD+ATTACK`). Each episode had one `+0.1` door-opening reward and one
`+1.0` crossing reward. The action/position/sector windows are saved in
`crossing_replay_trace.csv`.

The follow-up trainer mixes six closed-door starts, six matching pre-opened-door starts, and the
original spawn with the door closed. Pre-opened variants change only the temporary training WAD's
door-sector ceiling and player start; the policy still receives the normal observations and must
cross into the destination sector. Reset detects the already-open door so it is not incorrectly
awarded as a newly opened door.

```powershell
uv run python training\train_crossing_mixed_curriculum.py --seed 41 --updates 12 --rollout-length 64 --epochs 4 --episode-ticks 1400 --checkpoint-initial submission\weights\ppo_crossing_curriculum.pt --checkpoint-output submission\weights\ppo_crossing_mixed_smoke.pt --device cpu
```

This short run collected 768 decisions and 12 updates, with 2 newly opened doors, 1 destination
crossing, `+1.2` shaping reward, and `+0.1` official reward. The candidate is separate from the
previous crossing checkpoint.

Evaluate the mixed candidate and its parent on the original map, with the same 2,100-tic cap,
sampled RNG scheme, and both inference modes. Seeds 0–9 reproduce the earlier comparison; 10–19
are the fresh set:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_crossing_curriculum.pt submission\weights\ppo_crossing_mixed_smoke.pt --modes greedy sampled --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-ticks 2100 --sampling-seed 1337 --trace-seed 0 --output training\crossing_mixed_comparison.csv --trace-output training\crossing_mixed_stall_probabilities.csv --device cpu
```

| Checkpoint | Mode | Seeds | Exits | Deaths | Door opened / fully open | Crossings | Mean official return |
|---|---|---|---:|---:|---:|---:|---:|
| `ppo_crossing_curriculum.pt` | greedy | 0–9 | 0/10 | 0/10 | 0/10 / 0/10 | 0/10 | 0.000 |
| `ppo_crossing_curriculum.pt` | greedy | 10–19 | 0/10 | 0/10 | 0/10 / 0/10 | 0/10 | 0.000 |
| `ppo_crossing_curriculum.pt` | sampled | 0–9 | 0/10 | 1/10 | 4/10 / 4/10 | 2/10 | -0.100 |
| `ppo_crossing_curriculum.pt` | sampled | 10–19 | 0/10 | 1/10 | 6/10 / 6/10 | 1/10 | -0.080 |
| `ppo_crossing_mixed_smoke.pt` | greedy | 0–9 | 0/10 | 0/10 | 0/10 / 0/10 | 0/10 | 0.000 |
| `ppo_crossing_mixed_smoke.pt` | greedy | 10–19 | 0/10 | 0/10 | 0/10 / 0/10 | 0/10 | 0.000 |
| `ppo_crossing_mixed_smoke.pt` | sampled | 0–9 | 0/10 | 0/10 | 1/10 / 0/10 | 0/10 | 0.000 |
| `ppo_crossing_mixed_smoke.pt` | sampled | 10–19 | 0/10 | 0/10 | 0/10 / 0/10 | 0/10 | 0.000 |

The mixed candidate did not improve sampled play on either seed group: it had no crossings and
only one partial door opening over 20 episodes, versus 3/20 crossings and 10/20 fully opened doors
for its parent. Greedy play remained stuck choosing `MOVE_BACKWARD` in all episodes. The sampled
candidate was not proposed for deployment, so the default checkpoint and inference mode were not
changed and no deployment-mode validation was warranted.

## Varied-pose open-door fine-tune

`train_open_door_curriculum.py` now interleaves near, mid, and far open-door starts, so each
difficulty is visited throughout a short rollout instead of spending the budget on one group.
The task retains the full crossing objective and awards one training-only bonus for the first
confirmed entry into the destination sector.

The bounded CPU run started from `ppo_crossing_curriculum.pt`:

```powershell
uv run python training\train_open_door_curriculum.py --seed 53 --updates 12 --rollout-length 64 --epochs 4 --episode-ticks 700 --checkpoint-initial submission\weights\ppo_crossing_curriculum.pt --checkpoint-output submission\weights\ppo_open_door_varied_smoke.pt --device cpu
```

It collected 768 decisions and 12 PPO updates, changed the weights, and saved the separate
`ppo_open_door_varied_smoke.pt` checkpoint. Four episodes completed: near 2 (0 crossings), mid 1
(1 crossing), and far 1 (0 crossings). Official reward was `0.0`; training-only crossing reward
was `+1.0`.

The held-out comparison uses the same four open-door poses, seeds 0–2, 1,400-tic cap, sampling
seed 1337, and random-policy seed 2026 as `open_door_benchmark.csv`:

```powershell
uv run python training\evaluate_open_door_poses.py --checkpoint submission\weights\ppo_open_door_varied_smoke.pt --seeds 0 1 2 --modes greedy sampled random --episode-ticks 1400 --sampling-seed 1337 --random-seed 2026 --output training\open_door_varied_smoke_eval.csv --trace-output training\open_door_varied_smoke_traces.csv --device cpu
```

| Policy | Crossings | Mean decisions to cross | Deaths | Exits | Mean official return |
|---|---:|---:|---:|---:|---:|
| Benchmark greedy | 0/12 | — | 0/12 | 0/12 | 0.000 |
| Candidate greedy | 3/12 | 9 | 2/12 | 0/12 | -0.142 |
| Benchmark sampled | 1/12 | 202 | 3/12 | 0/12 | -0.233 |
| Candidate sampled | 3/12 | 140.7 | 2/12 | 0/12 | -0.142 |
| Random control | 4/12 | 35.3 | 4/12 | 1/12 | -0.233 |

The candidate improves on its benchmark in both policy modes, but does not beat the seeded random
control on these open-door poses. Its greedy crossings all came from the nearest pose. The trace
shows it repeatedly choosing `MOVE_FORWARD+ATTACK`; on original-map seed 0 it stalls at
`(432,416)` after reaching the closed doorway and never opens it. The bounded result does not
justify another longer run by itself.

The original-map comparison uses seeds 0–19, the same 2,100-tic cap, sampling seed 1337, and both
greedy and sampled inference:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_crossing_curriculum.pt submission\weights\ppo_open_door_varied_smoke.pt --modes greedy sampled --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-ticks 2100 --sampling-seed 1337 --trace-seed 0 --output training\open_door_varied_original_comparison.csv --trace-output training\open_door_varied_original_traces.csv --device cpu
```

| Checkpoint | Mode | Exits | Deaths | Doors opened / fully open | Crossings | Mean official return | Mean sectors |
|---|---|---:|---:|---:|---:|---:|---:|
| `ppo_crossing_curriculum.pt` | greedy | 0/20 | 0/20 | 0/20 / 0/20 | 0/20 | 0.000 | 1.00 |
| `ppo_open_door_varied_smoke.pt` | greedy | 0/20 | 0/20 | 0/20 / 0/20 | 0/20 | 0.000 | 1.00 |
| `ppo_crossing_curriculum.pt` | sampled | 0/20 | 2/20 | 10/20 / 10/20 | 3/20 | -0.090 | 1.40 |
| `ppo_open_door_varied_smoke.pt` | sampled | 0/20 | 3/20 | 13/20 / 12/20 | 10/20 | -0.125 | 2.05 |

Sampled crossings improved in both seed blocks (benchmark 2/10 then 1/10; candidate 8/10 then
2/10), but no episode reached an exit, the candidate had one additional death, and its mean
official return was lower. Its most frequent sampled actions were `MOVE_FORWARD+ATTACK` (1,647),
`MOVE_BACKWARD` (1,279), and `ATTACK` (767). Greedy behavior did not improve: it chose
`MOVE_FORWARD+ATTACK` for 524 of 525 decisions per episode and remained blocked by the closed
door. The candidate is retained for comparison, not deployment; the submission default and
inference behavior remain unchanged. No separate-layout test or deployment validation was run
because this checkpoint is not being selected for submission.

## Post-crossing exit follow-up

The original-map comparison added a uniform-random control that draws each action uniformly
from the shared 15-action categorical menu. The random generator is reseeded as
`random_seed + episode_seed` (random seed 2026); policy sampling remains
`sampling_seed + episode_seed` (sampling seed 1337). All episodes used seeds 0-19 and a 2,100-tic
cap:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_crossing_curriculum.pt submission\weights\ppo_open_door_varied_smoke.pt --modes sampled random --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --trace-seed 0 --output training\post_crossing_random_comparison.csv --trace-output training\post_crossing_random_stalls.csv --device cpu
```

| Policy | Crossings | Exits | Deaths | Mean official return |
|---|---:|---:|---:|---:|
| `ppo_crossing_curriculum.pt`, sampled | 3/20 | 0/20 | 2/20 | -0.090 |
| `ppo_open_door_varied_smoke.pt`, sampled | 10/20 | 0/20 | 3/20 | -0.125 |
| Uniform random, 15 actions | 5/20 | 0/20 | 1/20 | -0.035 |

The previous candidate's sampled crossings were in seeds 1, 2, 3, 5, 6, 7, 8, 9, 14, and 17.
They were replayed with the same `torch.manual_seed(1337 + seed)` behavior:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_open_door_varied_smoke.pt --modes sampled --seeds 1 2 3 5 6 7 8 9 14 17 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --trace-seed 0 --output training\post_crossing_candidate_replays.csv --trace-output training\post_crossing_candidate_replay_traces.csv --device cpu
```

The first crossing occurred at a mean of decision 292.6 (median 330), leaving a mean of 215.6
decisions before episode end. Nine of ten successful episodes stayed in destination sector 1
for every remaining decision; seed 6 returned to sector 0. Only seed 7 crossed as late as decision
508, leaving 17 decisions. The post-crossing action totals were `MOVE_FORWARD+ATTACK` 330,
`MOVE_BACKWARD` 260, `MOVE_LEFT+ATTACK` 98, and `MOVE_RIGHT+ATTACK` 48. This indicates that
crossing time was usually available; the policy failed to find the exit while remaining in the
destination room.

The bounded follow-up therefore trains the actual official exit from four varied starts in the
destination sector (door pre-opened in temporary training WADs), while two of every six task
episodes remain full-route, closed-door crossing attempts from the original spawn:

```powershell
uv run python training\train_exit_curriculum.py --seed 61 --updates 20 --rollout-length 128 --epochs 4 --episode-ticks 2100 --checkpoint-initial submission\weights\ppo_open_door_varied_smoke.pt --checkpoint-output submission\weights\ppo_exit_curriculum_smoke.pt --device cpu
```

The run collected 2,560 decisions and 20 PPO updates, changed weights, and saved the separate
`ppo_exit_curriculum_smoke.pt`. It recorded three exits in five completed destination-start
episodes, one full-route crossing in three completed full-route episodes, official reward `+2.2`,
and training-only opening/crossing reward `+1.1`. Exit episodes use the official exit reward; no
extra exit shaping or privileged observations are added to the policy.

The candidate was then evaluated from the original spawn with the same seeds, cap, and sampled RNG
settings; greedy mode was included to check the normal `DoomAgent` path:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_exit_curriculum_smoke.pt --modes greedy sampled --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --trace-seed 0 --output training\exit_curriculum_original_eval.csv --trace-output training\exit_curriculum_original_stalls.csv --device cpu
```

| Mode | First crossing | Crossings | Exits | Deaths | Mean official return |
|---|---:|---:|---:|---:|---:|
| Greedy | none | 0/20 | 0/20 | 0/20 | 0.000 |
| Sampled | mean decision 93.8 | 5/20 | 0/20 | 6/20 | -0.300 |

Although the checkpoint learned to exit from some training starts, it did not transfer to the
original spawn and regressed against the open-door candidate there. It is not proposed for
submission; the default checkpoint and inference behavior remain unchanged. No separate-layout
test or validator run was warranted.

## Held-out destination-room transfer check

The exit checkpoint was compared against the open-door navigation benchmark and a uniform-random
15-action policy on four destination-room starts not used for training:
`(592,336,0)`, `(640,400,90)`, `(752,320,180)`, and `(752,192,270)`. Each temporary WAD starts
with the manual door open; privileged checks verify the player starts in sector 1. All policies
used seeds 0-2 and the same 2,100-tic cap. Sampled policies use `1337 + pose_index * 10000 + seed`;
the random control uses `2026 + pose_index * 10000 + seed`.

```powershell
uv run python training\evaluate_exit_poses.py --checkpoints submission\weights\ppo_open_door_varied_smoke.pt submission\weights\ppo_exit_curriculum_smoke.pt --modes greedy sampled random --seeds 0 1 2 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --output training\exit_heldout_pose_results.csv --device cpu
```

| Policy | Mode | Exits | Deaths | Mean decisions to exit | Mean official return |
|---|---|---:|---:|---:|---:|
| `ppo_open_door_varied_smoke.pt` | greedy | 0/12 | 11/12 | — | -0.917 |
| `ppo_exit_curriculum_smoke.pt` | greedy | 0/12 | 11/12 | — | -0.917 |
| `ppo_open_door_varied_smoke.pt` | sampled | 2/12 | 7/12 | 213.5 | -0.400 |
| `ppo_exit_curriculum_smoke.pt` | sampled | 2/12 | 7/12 | 142.0 | -0.417 |
| Uniform random, 15 actions | random | 3/12 | 8/12 | 106.3 | -0.400 |

The three training exits did not transfer as a measurable held-out improvement: the exit checkpoint
matched the navigation benchmark's sampled exit count and deaths, while random reached more exits.
It did not beat the controls, so no additional fine-tuning stage was launched.

## Original-spawn exit-checkpoint replay

The exit checkpoint's sampled original-spawn episodes were replayed with seeds 0-19,
`torch.manual_seed(1337 + seed)`, and the same 2,100-tic cap:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_exit_curriculum_smoke.pt --modes sampled --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --trace-seed 0 --output training\exit_checkpoint_original_replay.csv --trace-output training\exit_checkpoint_original_traces.csv --device cpu
```

Of the six deaths, five occurred after crossing (seeds 0, 1, 2, 15, and 19); one occurred before
crossing (seed 5). All 14 surviving episodes had a first low-displacement window in sector 0.
Those first stalls clustered around the doorway's west-side approach and map boundaries, especially
near `x=430, y=208` and `x=432, y=493`. One survivor (seed 11) later reached sector 2 but ended the
episode there. Surviving runs mostly failed to reach the destination room; the five deaths after
crossing show that survival in the destination room also remains unresolved.

The paired original-spawn results on seeds 0-19 remain:

| Checkpoint | Mode | Exits | Crossings | Deaths | Mean official return |
|---|---|---:|---:|---:|---:|
| `ppo_crossing_curriculum.pt` | sampled | 0/20 | 3/20 | 2/20 | -0.090 |
| `ppo_open_door_varied_smoke.pt` | sampled | 0/20 | 10/20 | 3/20 | -0.125 |
| `ppo_exit_curriculum_smoke.pt` | sampled | 0/20 | 5/20 | 6/20 | -0.300 |

Because the exit checkpoint did not beat the controls from held-out starts and did not improve
original-spawn gameplay, no fresh-seed or separate-layout promotion check and no validator run were
performed. The submission default was not changed.

## Health-aware exit follow-up

The five sampled original-spawn deaths after crossing (exit checkpoint seeds 0, 1, 2, 15, and 19)
were replayed with the same per-episode Torch sampling seed and 2,100-tic cap:

```powershell
uv run python training\replay_exit_death_traces.py --checkpoint submission\weights\ppo_exit_curriculum_smoke.pt --seeds 0 1 2 15 19 --episode-ticks 2100 --sampling-seed 1337 --trace-output training\exit_checkpoint_post_crossing_traces.csv --summary-output training\exit_checkpoint_post_crossing_summary.csv --device cpu
```

| Seed | Crossing → death | Health at crossing → death | Damage after crossing | Selected ammo at crossing → before death | Visible enemies, mean/max | Moving decisions | Path distance | Attack actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 125 decisions / 497 tics | 100 → -5 | 105 | 39 → 0 | 0.04 / 1 | 123/126 | 763.5 | 28.6% |
| 1 | 180 / 720 | 100 → -5 | 105 | 0 → 0 | 0.03 / 1 | 175/181 | 1,197.4 | 37.0% |
| 2 | 373 / 1,490 | 100 → -2 | 102 | 0 → 0 | 0.10 / 1 | 357/374 | 2,526.3 | 27.8% |
| 15 | 136 / 542 | 91 → -11 | 102 | 40 → 34 | 0.16 / 1 | 125/137 | 874.1 | 29.9% |
| 19 | 193 / 771 | 100 → -5 | 105 | 0 → 0 | 0.30 / 1 | 190/194 | 1,501.7 | 30.9% |

Each trace contains per-decision health before/after, selected-weapon and reserve ammo, visible
enemy count, action, position, sector, and damage. Health fell in 10–16 discrete damage steps per
episode while the player was moving; attack was not the dominant action. Visible enemies were
usually absent, with at most one visible at a time. This supports testing a health-loss penalty,
but does not establish the source of the incoming damage. Across all post-crossing decisions, the
most frequent actions were `MOVE_FORWARD+ATTACK` (134), `MOVE_RIGHT` (81),
`MOVE_FORWARD+TURN_LEFT` (81), `MOVE_FORWARD` (71), `USE` (69), and
`SELECT_NEXT_WEAPON` (69).

The navigation benchmark and uniform-random control were measured on the four established held-out
destination-room poses, seeds 0–2, and a 2,100-tic cap. The same protocol was then used for both
fine-tune checkpoints; training-state metrics were not exposed to the policy:

```powershell
uv run python training\evaluate_exit_poses.py --checkpoints submission\weights\ppo_open_door_varied_smoke.pt submission\weights\ppo_health_exit_intermediate.pt submission\weights\ppo_health_exit_final.pt --modes greedy sampled random --seeds 0 1 2 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --output training\health_heldout_pose_comparison.csv --device cpu
```

| Policy | Mode | Exits | Deaths | Mean official return | Mean health lost | Mean survived decisions | Mean path distance | Movement fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Navigation benchmark | sampled | 2/12 | 7/12 | -0.400 | 82.0 | 347.8 | 2,392.3 | 0.642 |
| Uniform random | random | 3/12 | 8/12 | -0.400 | 71.3 | 271.0 | 1,676.3 | 0.609 |
| Health intermediate | sampled | 3/12 | 6/12 | -0.242 | 74.0 | 284.6 | 1,933.8 | 0.601 |
| Health final | sampled | 4/12 | 6/12 | -0.158 | 65.8 | 278.7 | 1,794.6 | 0.597 |
| Health final | greedy | 0/12 | 5/12 | -0.358 | 55.2 | 394.6 | 61.2 | 0.003 |

Health loss is the sum of positive per-decision health reductions; survival time is decisions
(four game tics per decision).

The final sampled policy improved held-out exit count, deaths, official return, and health lost over
the navigation benchmark. Its greedy policy mostly idled, so that survival result is not a useful
deployment gain.

One bounded CPU fine-tune started from the navigation benchmark. It used two full-route slots and
four destination-room slots in a repeating schedule. Full-route episodes had a 4,200-tic cap and
continued after crossing; destination-room episodes had a 2,100-tic cap. Door opening and crossing
were logged/rewarded once per full-route episode, while only the underlying environment could end
an episode at death, timeout, or the official exit. Health penalty was `-0.002` per health point
lost, applied only in training:

```powershell
uv run python training\train_health_aware_exit.py --seed 73 --updates 24 --intermediate-updates 12 --rollout-length 128 --epochs 4 --full-route-ticks 4200 --exit-ticks 2100 --health-penalty-per-point 0.002 --checkpoint-initial submission\weights\ppo_open_door_varied_smoke.pt --checkpoint-intermediate submission\weights\ppo_health_exit_intermediate.pt --checkpoint-output submission\weights\ppo_health_exit_final.pt --device cpu
```

The run collected 3,072 decisions over 24 PPO updates and changed weights. It saved the intermediate
checkpoint after 1,536 decisions. Training totals were official return `0.000`, total shaping
return `-0.464`, and health penalty `-0.564` across 282 health points lost. It completed two
full-route timeouts with one door opening and no crossings; four completed destination-start
episodes produced two exits and two deaths. The training reward breakdown is stored separately
from evaluator official returns in checkpoint metadata.

Both checkpoints were evaluated through `DoomAgent` on the original spawn, sampled mode, seeds
0–19, and the same cap/RNG settings as the previous benchmark comparison:

```powershell
uv run python training\evaluate_policy_modes.py --checkpoints submission\weights\ppo_health_exit_intermediate.pt submission\weights\ppo_health_exit_final.pt --modes sampled --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-ticks 2100 --sampling-seed 1337 --random-seed 2026 --trace-seed 0 --output training\health_original_spawn_candidates.csv --trace-output training\health_original_spawn_candidate_stalls.csv --device cpu
```

| Policy | Crossings | Exits | Deaths | Mean official return |
|---|---:|---:|---:|---:|
| Navigation benchmark | 10/20 | 0/20 | 3/20 | -0.125 |
| Uniform random | 5/20 | 0/20 | 1/20 | -0.035 |
| Health intermediate | 4/20 | 0/20 | 2/20 | -0.090 |
| Health final | 4/20 | 1/20 | 1/20 | 0.015 |

The final checkpoint gained one exit and reduced deaths on the original spawn, but crossings fell
from 10/20 to 4/20. It therefore fails the crossing-preservation gate despite the held-out
destination-room improvement. Neither candidate is being promoted: the submission default and
inference behavior remain unchanged, and no separate-layout test or official validator run was
warranted.

## Sampled submission-path verification

The recorded original-spawn success was replayed on `two_rooms.wad` with
`ppo_health_exit_final.pt`, environment seed 12, 2,100-tic cap, and the original Torch sampling
protocol (`torch.manual_seed(1337 + 12)`, seed 1349). It exited in 394 decisions with official
return `+1.1`, crossed into the destination sector at decision 83, visited four sectors, and ended
with 10 health. This replay used the historical Torch-based diagnostic sampler; it is distinct
from the agent-local RNG path evaluated below.

For submission-path evaluation, `DoomAgent.act()` sampled categorical actions using its own NumPy
generator. The paired evaluator resets each policy with `sampling_seed + environment_seed`, using
sampling base 71000; the random control uses `random_seed + environment_seed`, with base 2026.
`DoomAgent.reset()` clears GRU memory and restarts the local RNG at `DOOM_SAMPLING_SEED`, while
`reset(seed=N)` uses `N`. Greedy inference remains the default unless
`DOOM_INFERENCE_MODE=sampled` is set.

The fresh original-spawn comparison used seeds 21-30, the same 2,100-tic cap, and the exact
`DoomAgent.act()` sampled path for both checkpoints:

```powershell
uv run python training\evaluate_submission_path.py --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_open_door_varied_smoke.pt --seeds 21 22 23 24 25 26 27 28 29 30 --episode-ticks 2100 --sampling-seed 71000 --random-seed 2026 --output training\submission_path_fresh_original.csv --device cpu
```

The separate-layout comparison used the bundled Freedoom `MAP02` IWAD, seeds 21-25, and the same
cap and RNG scheme:

```powershell
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP02 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_open_door_varied_smoke.pt --seeds 21 22 23 24 25 --episode-ticks 2100 --sampling-seed 71000 --random-seed 2026 --output training\submission_path_fresh_map02.csv --device cpu
```

| Map / policy | Exits | Deaths | Mean official return | Progression |
|---|---:|---:|---:|---|
| `two_rooms`, health-aware final | 1/10 | 0/10 | +0.120 | Door fully open 5/10; crossings 2/10; mean sectors 1.5; mean transitions 0.5 |
| `two_rooms`, navigation benchmark | 0/10 | 2/10 | -0.190 | Door fully open 4/10; crossings 1/10; mean sectors 1.2; mean transitions 0.2 |
| `two_rooms`, uniform random | 0/10 | 2/10 | -0.200 | Door fully open 4/10; crossings 0/10; mean sectors 1.0; mean transitions 0.0 |
| Freedoom `MAP02`, health-aware final | 0/5 | 5/5 | -0.720 | Mean sectors 8.0; mean transitions 18.6 |
| Freedoom `MAP02`, navigation benchmark | 0/5 | 3/5 | -0.460 | Mean sectors 4.0; mean transitions 9.8 |
| Freedoom `MAP02`, uniform random | 0/5 | 3/5 | -0.380 | Mean sectors 6.2; mean transitions 15.6 |

The health-aware candidate improved the fresh original-map exit count and survival, including one
exit on seed 28, but it died in every `MAP02` episode and had the worst mean official return there.
Its higher sector exploration on `MAP02` did not translate into survival or exits. Do not promote
it for submission: keep the current `ppo_smoke.pt` default unchanged, and retain
`ppo_open_door_varied_smoke.pt` as the navigation comparison checkpoint.

The local official validator was run with the candidate explicitly loaded and sampled inference
enabled:

```powershell
$env:DOOM_CHECKPOINT_PATH = "C:\Projects\Devforge\agent-doom\submission\weights\ppo_health_exit_final.pt"
$env:DOOM_INFERENCE_MODE = "sampled"
$env:DOOM_SAMPLING_SEED = "71000"
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py --device cpu --ticks 2100 --map maps\two_rooms.wad --map .venv\Lib\site-packages\vizdoom\freedoom2.wad:MAP02
```

The loader probe confirmed `ppo_health_exit_final.pt`, checkpoint format 2, sampled mode, and
sampling seed 71000. Validation **passed**: `two_rooms.wad:MAP01` ran 525 decisions to the tick
cap, and `freedoom2.wad:MAP02` ran 423 decisions before death; both produced valid actions within
latency and compute budgets. There were four environment/worktree warnings: untracked submission
files, uncommitted `submission/agent.py`, CPU affinity unavailable, and peak RAM unavailable on
Windows. In particular, the candidate and source changes are not represented in a clean pushed
checkout until committed. Docker validation was unavailable: the Docker client exists, but its
Desktop Linux engine pipe was missing, so the daemon could not be reached.

## Paired multi-layout submission comparison

The comparison evaluated the current default `ppo_smoke.pt` in its normal greedy mode, the
health-aware candidate and navigation benchmark in sampled mode through `DoomAgent.act()`, and a
uniform random policy over the shared 15-action menu. Every map used paired environment seeds
31-40 and a 2,100-tic cap. Sampled agents used local RNG seed `81000 + environment_seed`; random
used `3026 + environment_seed`. Greedy inference is deterministic and used no sampling RNG.

`MAP02` and the additional Freedoom2 `MAP01` were not used for training. All prior training used
`maps/two_rooms.wad`. The evaluator also records legacy nine-button combinations that are outside
the categorical menu by their button names; all actions are still checked against the official
`MultiBinary(9)` space.

```powershell
uv run python training\evaluate_submission_path.py --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_open_door_varied_smoke.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 31 32 33 34 35 36 37 38 39 40 --episode-ticks 2100 --sampling-seed 81000 --random-seed 3026 --output training\submission_matrix_two_rooms.csv --device cpu
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP02 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_open_door_varied_smoke.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 31 32 33 34 35 36 37 38 39 40 --episode-ticks 2100 --sampling-seed 81000 --random-seed 3026 --output training\submission_matrix_freedoom2_map02.csv --device cpu
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP01 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_open_door_varied_smoke.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 31 32 33 34 35 36 37 38 39 40 --episode-ticks 2100 --sampling-seed 81000 --random-seed 3026 --output training\submission_matrix_freedoom2_map01.csv --device cpu
```

| Map / policy | Exits | Deaths | Mean official return | Progression |
|---|---:|---:|---:|---|
| `two_rooms`, `ppo_smoke.pt` (greedy) | 0/10 | 0/10 | 0.000 | Crossings 0/10; mean sectors 1.0; transitions 0.0 |
| `two_rooms`, `ppo_health_exit_final.pt` (sampled) | 0/10 | 1/10 | -0.080 | Crossings 2/10; door fully open 4/10; sectors 1.5; transitions 0.6 |
| `two_rooms`, `ppo_open_door_varied_smoke.pt` (sampled) | 0/10 | 3/10 | -0.290 | Crossings 3/10; door fully open 8/10; sectors 1.6; transitions 3.4 |
| `two_rooms`, uniform random | 0/10 | 1/10 | -0.100 | Crossings 2/10; door fully open 6/10; sectors 1.4; transitions 0.4 |
| Freedoom2 `MAP02`, `ppo_smoke.pt` (greedy) | 0/10 | 0/10 | 0.000 | Sectors 2.0; transitions 1.0 |
| Freedoom2 `MAP02`, health-aware final (sampled) | 0/10 | 7/10 | -0.410 | Sectors 5.6; transitions 10.4 |
| Freedoom2 `MAP02`, navigation benchmark (sampled) | 0/10 | 9/10 | -0.690 | Sectors 6.2; transitions 17.9 |
| Freedoom2 `MAP02`, uniform random | 0/10 | 6/10 | -0.220 | Sectors 9.2; transitions 19.8 |
| Freedoom2 `MAP01`, `ppo_smoke.pt` (greedy) | 0/10 | 0/10 | 0.000 | Sectors 2.0; transitions 2.0 |
| Freedoom2 `MAP01`, health-aware final (sampled) | 0/10 | 6/10 | -0.430 | Sectors 13.3; transitions 33.0 |
| Freedoom2 `MAP01`, navigation benchmark (sampled) | 0/10 | 9/10 | -0.760 | Sectors 14.3; transitions 28.5 |
| Freedoom2 `MAP01`, uniform random | 0/10 | 9/10 | -0.780 | Sectors 12.3; transitions 24.9 |

No policy reached an exit in 30 episodes per policy. The default has the best survival and return
on both Freedoom maps, but it barely moves. The sampled health-aware candidate beats random and the
navigation benchmark on `MAP01`, but loses to random on `MAP02` in survival, return, and sector
coverage. It therefore does not show a consistent generalization gain; keep `ppo_smoke.pt` greedy
as the conservative default, and do not promote either sampled candidate.

Training remains paused on `two_rooms`. For the next training set, use the separate Freedoom2
`MAP03`-`MAP06` layouts (verified available at 315, 139, 185, and 109 sectors respectively), and
keep evaluation `MAP01` and `MAP02` reserved. No new training run was launched in this comparison.

## Four-map health-aware PPO run

Freedoom2 `MAP03`-`MAP06` each initialized and accepted a valid action in a 128-tic smoke check.
Training rotates through those maps sequentially when an episode resets; it does not modify the
IWAD or use the reserved evaluation maps. The recurrent hidden state is cleared by the existing
rollout collector at each termination or truncation. Health loss uses the existing training-only
penalty of `-0.002` per health point, and the trainer records it separately from official reward.

```powershell
@'
from pathlib import Path
import vizdoom
from doom_env import make_env
from submission.actions import ACTION_ARRAYS

wad = Path(vizdoom.__file__).parent / "freedoom2.wad"
for index, map_name in enumerate(("MAP03", "MAP04", "MAP05", "MAP06")):
    env = make_env(str(wad), map_name, episode_ticks=128, render_buffers=False)
    try:
        env.reset(seed=100 + index)
        env.step(ACTION_ARRAYS[1].copy())
        print(f"{map_name}: initialized; action_step=ok")
    finally:
        env.close()
'@ | uv run python -
```

```powershell
uv run python training\train_ppo.py --seed 20261127 --updates 40 --intermediate-updates 20 --rollout-length 128 --epochs 4 --device cpu --checkpoint-initial submission\weights\ppo_health_exit_final.pt --checkpoint-intermediate submission\weights\ppo_multimap_health_20.pt --checkpoint-output submission\weights\ppo_multimap_health_40.pt --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-names MAP03 MAP04 MAP05 MAP06 --episode-ticks 2100 --health-penalty-per-point 0.002
```

Paired evaluation uses the same ten environment seeds and 2,100-tic cap on each reserved map.
Format-2 policies use seeded sampled `DoomAgent.act()`; `ppo_smoke.pt` uses its default greedy
mode. Random actions use the evaluator's 15-action uniform control.

```powershell
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP01 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_multimap_health_20.pt submission\weights\ppo_multimap_health_40.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 61 62 63 64 65 66 67 68 69 70 --episode-ticks 2100 --sampling-seed 91000 --random-seed 6100 --output training\multimap_eval_freedoom2_map01.csv --device cpu
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP02 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_multimap_health_20.pt submission\weights\ppo_multimap_health_40.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 61 62 63 64 65 66 67 68 69 70 --episode-ticks 2100 --sampling-seed 91000 --random-seed 6100 --output training\multimap_eval_freedoom2_map02.csv --device cpu
```

| Reserved map / policy | Exits | Deaths | Mean official return | Mean sectors / transitions |
|---|---:|---:|---:|---:|
| MAP01 / health-aware original (sampled) | 0/10 | 10/10 | -0.890 | 14.0 / 25.9 |
| MAP01 / intermediate, 20 updates (sampled) | 0/10 | 7/10 | -0.560 | 13.5 / 29.6 |
| MAP01 / final, 40 updates (sampled) | 0/10 | 5/10 | -0.420 | 8.8 / 16.5 |
| MAP01 / `ppo_smoke.pt` (greedy) | 0/10 | 0/10 | 0.000 | 2.0 / 2.0 |
| MAP01 / uniform random | 0/10 | 9/10 | -0.720 | 15.6 / 25.0 |
| MAP02 / health-aware original (sampled) | 0/10 | 8/10 | -0.630 | 7.2 / 17.6 |
| MAP02 / intermediate, 20 updates (sampled) | 0/10 | 6/10 | -0.320 | 7.3 / 16.3 |
| MAP02 / final, 40 updates (sampled) | 0/10 | 9/10 | -0.720 | 7.2 / 15.8 |
| MAP02 / `ppo_smoke.pt` (greedy) | 0/10 | 0/10 | 0.000 | 2.0 / 1.0 |
| MAP02 / uniform random | 0/10 | 8/10 | -0.590 | 6.3 / 14.0 |

The 20-update checkpoint initially improved survival and return over the original health-aware
checkpoint and random on both maps, so it was checked on fresh seeds 71-75 with the same tick cap.
The evaluation commands were:

```powershell
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP01 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_multimap_health_20.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 71 72 73 74 75 --episode-ticks 2100 --sampling-seed 92000 --random-seed 7100 --output training\multimap_fresh_freedoom2_map01.csv --device cpu
uv run python training\evaluate_submission_path.py --wad .venv\Lib\site-packages\vizdoom\freedoom2.wad --map-name MAP02 --checkpoints submission\weights\ppo_health_exit_final.pt submission\weights\ppo_multimap_health_20.pt --greedy-checkpoints submission\weights\ppo_smoke.pt --seeds 71 72 73 74 75 --episode-ticks 2100 --sampling-seed 92000 --random-seed 7100 --output training\multimap_fresh_freedoom2_map02.csv --device cpu
```

| Fresh map / policy | Exits | Deaths | Mean official return | Mean sectors / transitions |
|---|---:|---:|---:|---:|
| MAP01 / health-aware original (sampled) | 0/5 | 3/5 | -0.460 | 12.2 / 22.8 |
| MAP01 / intermediate, 20 updates (sampled) | 0/5 | 3/5 | -0.420 | 13.6 / 28.6 |
| MAP01 / `ppo_smoke.pt` (greedy) | 0/5 | 0/5 | 0.000 | 2.0 / 2.0 |
| MAP01 / uniform random | 0/5 | 3/5 | -0.380 | 17.2 / 34.8 |
| MAP02 / health-aware original (sampled) | 0/5 | 4/5 | -0.540 | 6.6 / 16.6 |
| MAP02 / intermediate, 20 updates (sampled) | 0/5 | 5/5 | -0.680 | 10.2 / 17.2 |
| MAP02 / `ppo_smoke.pt` (greedy) | 0/5 | 0/5 | 0.000 | 2.0 / 1.0 |
| MAP02 / uniform random | 0/5 | 4/5 | -0.660 | 6.4 / 23.2 |

No candidate reached an exit. The intermediate checkpoint's initial gains did not hold across the
fresh MAP02 seeds, and neither candidate outperformed the default in survival or official return.
Keep `ppo_smoke.pt` as the fallback; neither multi-map checkpoint is a submission candidate. The
final checkpoint also regressed on MAP02 versus the original health-aware checkpoint.

The official validator was run only on `two_rooms` with all `DOOM_*` inference overrides unset.
The first attempt hit this Windows console's cp1252 encoding; the same check passed with UTF-8:

```powershell
Remove-Item Env:DOOM_CHECKPOINT_PATH -ErrorAction SilentlyContinue
Remove-Item Env:DOOM_INFERENCE_MODE -ErrorAction SilentlyContinue
Remove-Item Env:DOOM_SAMPLING_SEED -ErrorAction SilentlyContinue
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py --device cpu --ticks 2100 --map maps\two_rooms.wad
```

Validation passed for the default checkpoint (525 decisions to the cap) with four warnings:
pre-existing untracked submission files, pre-existing uncommitted submission files, CPU affinity
unavailable, and RAM measurement unavailable on Windows. The new checkpoints were evaluated only
on the reserved Freedoom2 maps and were not passed to the validator because neither is proposed
for submission.

## Default checkpoint and clean-tree validation

The default resolves to `submission/weights/ppo_smoke.pt` without `DOOM_CHECKPOINT_PATH`; the
inference mode defaults to greedy without `DOOM_INFERENCE_MODE`. `.gitattributes` routes `*.pt`
through Git LFS. To verify the retained default, `submission/agent.py`,
`submission/actions.py`, and `ppo_smoke.pt` were staged; the LFS status identifies the checkpoint
as an LFS object. No commit or push was made.

A clean archive of the Git index was exported, the LFS pointer was smudged from the local object,
and the loader was run with all three `DOOM_*` overrides unset. It loaded the archived default
`ppo_smoke.pt` (format 1, greedy). The local `validate.py` run on that clean export **passed**:
525 decisions to the `two_rooms` tick cap, with valid actions and within latency/compute budgets.
Its three warnings were the archive lacking `.git` metadata, unavailable CPU affinity, and
unavailable RAM measurement on Windows. The clean-index test demonstrates the staged snapshot,
but it is not yet a clean clone of a pushed commit because the staged files have not been committed.
Docker remains unavailable: the Docker client cannot reach the Desktop Linux engine pipe.
