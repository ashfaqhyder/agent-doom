<div align="center">

# ◢ AGENT DOOM ◣

**YUGAANTAR 2026 · DEVFORGE · TRACK 1**  
*Can It Play DOOM?*

<img src="https://readme-typing-svg.demolab.com?font=Share+Tech+Mono&size=22&duration=2300&pause=850&color=F15A4A&center=true&vCenter=true&width=540&lines=INITIALIZING+VIZDOOM...;ENCODING+LOCAL+GEOMETRY...;GRU+MEMORY+ONLINE;RECURRENT+PPO+%2F+AGENT+READY" alt="Agent DOOM · ViZDoom · GRU memory · Recurrent PPO" />

<br />

![Python 3.13](https://img.shields.io/badge/PYTHON-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ViZDoom](https://img.shields.io/badge/ENGINE-VIZDOOM-A6292A?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PYTORCH-RL-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Recurrent PPO](https://img.shields.io/badge/POLICY-RECURRENT%20PPO-633CA4?style=for-the-badge)

[![Windows validator passed](https://img.shields.io/badge/WINDOWS%20VALIDATOR-PASSED-287D46?style=flat-square&logo=checkmarx&logoColor=white)](https://github.com/ashfaqhyder/agent-doom/commit/0c0005efc78f9ea443e4eea839cae6b8036ac4b5)
![Weights tracked with Git LFS](https://img.shields.io/badge/WEIGHTS-GIT%20LFS-F05032?style=flat-square&logo=git&logoColor=white)
![Docker validation pending](https://img.shields.io/badge/DOCKER%20VALIDATION-PENDING-A66A00?style=flat-square&logo=docker&logoColor=white)

[Quickstart](#quickstart) · [Architecture](#architecture) · [Results](#results) · [Repository guide](#repository-guide)

</div>

---

Agent DOOM is a recurrent reinforcement learning agent for ViZDoom. It uses local geometry, player vitals, and visible entities to choose an action. The DevForge challenge is to survive encounters, navigate unfamiliar rooms, find keycards, unlock doors, and reach the exit on unseen layouts.

Built from the [official DevForge DOOM starter](https://github.com/Reinforce-SST/devforge-doom-template).

> [!NOTE]
> The default `ppo_smoke.pt` checkpoint received **two PPO updates over 32 game decisions**. It passed the official validator from a fresh Windows clone of [`0c0005e`](https://github.com/ashfaqhyder/agent-doom/commit/0c0005efc78f9ea443e4eea839cae6b8036ac4b5) and reached the 525-decision tick cap on the bundled verification map.

## Quickstart

Install Git LFS and [uv](https://docs.astral.sh/uv/). The project targets **Python 3.13**.

```bash
git clone https://github.com/ashfaqhyder/agent-doom.git
cd agent-doom
git lfs pull
uv sync --frozen
uv run python validate.py
```

<details>
<summary><strong>Windows console encoding fix</strong></summary>

If PowerShell reports a console encoding error, rerun the validator with UTF-8 output.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
uv run python validate.py
```

</details>

<details>
<summary><strong>Validate with Docker</strong></summary>

With the Docker Linux engine running, build the starter image or use the organizer-provided image tag.

```bash
docker build -t devforge-doom .
uv run python validate.py --docker --image devforge-doom
```

The evaluation budget is **2 GB VRAM, 8 GB RAM, and one CPU thread**. CPU affinity and peak RAM remain unverified on Windows. Docker validation is pending.

</details>

## Architecture

```mermaid
flowchart LR
    R["64 spatial rays"] --> F["Feature encoders"]
    V["Player vitals"] --> F
    E["Visible entities"] --> F
    F --> G["GRU memory"]
    G --> H["Actor and critic"]
```

| Component | Role |
|---|---|
| **Spatial encoder** | Processes player-relative ray distances and hit types to represent nearby geometry. |
| **Vitals encoder** | Processes the 17-value player state, including health, ammo, and keycard flags. |
| **Entity encoder** | Uses masked pooling to summarize visible entities and handles empty observations. |
| **GRU memory** | Carries information between decisions and resets at the start of each episode. |
| **Actor–critic** | The actor selects controls. The critic estimates expected return during PPO training. |

The default policy thresholds nine button logits to produce the official binary action vector. Experimental categorical policies choose from 15 compatible button combinations and convert the selected combination to the same nine-button format.

### Training

Experiments used recurrent PPO with ordered rollouts and tested exploration bonuses, door-opening and crossing curricula, exit practice, and a training-only health-loss penalty. The multi-map run trained on Freedoom2 `MAP03`–`MAP06` and reserved `MAP01` and `MAP02` for evaluation.

[Training commands and experiment notes](training/README.md)

## Results

| Evaluation | Result |
|---|---|
| **Fresh-clone validation** | The official Windows validator passed with the default Git LFS checkpoint and no inference overrides. |
| **Doorway crossing** | The sampled open-door checkpoint crossed in **10/20** original-spawn episodes on `two_rooms`. Uniform random crossed in **5/20**. |
| **Exit behavior** | A sampled health-aware checkpoint reached **1/10** exits on a fresh `two_rooms` seed set. |
| **Reserved maps** | All tested policies recorded **zero exits** in the paired Freedoom2 `MAP01` and `MAP02` comparison. The default survived every episode and averaged two sectors per map. |

The experimental checkpoints showed progress on `two_rooms`. Reliable exits on unseen layouts remain an open goal.

## Repository guide

| Path | Contents |
|---|---|
| [`submission/agent.py`](submission/agent.py) | `DoomAgent` inference entry point |
| [`submission/weights/`](submission/weights/) | Default checkpoint tracked through Git LFS |
| [`training/`](training/) | PPO trainer, evaluation code, and experiment notes |
| [`env/`](env/) | Official ViZDoom Gymnasium wrapper |
| [`maps/`](maps/) | Bundled `two_rooms.wad` verification map |
| [`validate.py`](validate.py) | Official local submission validator |
| [`tests/`](tests/) | Agent, training, and template tests |

### Official track documentation

[Agent contract](docs/CONTRACT.md) · [Rules](docs/RULES.md) · [Submission checklist](docs/SUBMISSION.md) · [Evaluation and scoring](docs/EVALUATION.md)

The scorer uses the frozen wrapper, vanilla engine rules, and unseen maps. Inference uses the official observations, with gameplay decisions driven by trained parameters. Submissions use a public GitHub repository, store weights through Git LFS, and stay within the **1 GB** repository limit.
