# HETT

## Introduction

The official repository for AAAI 2026 oral paper [History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation](https://arxiv.org/abs/2512.14222).

Aerial Vision-and-Language Navigation (AVLN) requires Unmanned Aerial Vehicle (UAV) agents to localize targets in large-scale urban environments based on linguistic instructions. While successful navigation demands both global environmental reasoning and local scene comprehension, existing UAV agents typically adopt mono-granularity frameworks that struggle to balance these two aspects. To address this limitation, this work proposes a History-Enhanced Two-Stage Transformer (HETT) framework, which integrates the two aspects through a coarse-to-fine navigation pipeline. Specifically, HETT first predicts coarse-grained target positions by fusing spatial landmarks and historical context, then refines actions via fine-grained visual analysis. In addition, a historical grid map is designed to dynamically aggregate visual features into a structured spatial memory, enhancing comprehensive scene awareness. Additionally, the CityNav dataset annotations are manually refined to enhance data quality. Experiments on the refined CityNav dataset show that HETT delivers significant performance gains, while extensive ablation studies further verify the effectiveness of each component.



Project Page:[HETT](https://crotonyl.github.io/HETT.git.io/)



## Setup

This code was developed with Python 3.10, PyTorch 2.2.2, and CUDA 11.3 on Ubuntu 22.04.

To set up the environment, create the conda environment and install PyTorch.

```bash
conda create -n hett python=3.10 &&
conda activate hett &&
conda install pytorch torchvision pytorch-cuda=11.3 -c pytorch -c nvidia
```

Install the dependencies for HETT.

```bash
pip install -r requirements.txt
```

## Data Preparation

Follow the instruction in [CityNav](https://github.com/water-cookie/citynav) for data download.

Download the refined dataset and corresponding checkpoints:

https://www.dropbox.com/scl/fo/ua3kn6mw3adn2hcsdikt3/AEdoRoo-OeXnbrpeOpk4BcQ?rlkey=v5rlqiqa1k9ml8isinpv7glgk&st=ahseq0r2&dl=0

## Usage

```bash
cd multiagent
# train
./train.sh
# eval
./eval.sh
```

## SBFNav paper-based reimplementation

`sbfnav/` is an independent, paper-based reimplementation of “Map the
Possibilities: Spatial Belief Fields for Language-Goal Aerial Navigation”
(arXiv:2609.05841). It is not official SBFNav code. The paper specification,
missing-detail assumptions, and live results are in
`docs/sbfnav_reproduction.md`, `docs/sbfnav_assumptions.md`, and
`docs/sbfnav_results.md`. Tracked epoch-7 training logs, complete validation
prediction JSONL files, candidate-analysis outputs, provenance, and case
visualizations are indexed in `artifacts/sbfnav/README.md`; checkpoints,
weights, caches, and datasets remain external.

Use the dedicated environment and supervised launcher for GPU work:

```bash
/home/ubuntu5/miniconda3/envs/SBFNav/bin/python scripts/supervise_experiment.py \
  --run-dir /home/ubuntu5/Workspace/hett-runs/sbfnav/<run-name> \
  --python /home/ubuntu5/miniconda3/envs/SBFNav/bin/python \
  --config configs/sbfnav/main.yaml --phase train-eval \
  --config-override data.root=/path/to/audited/data
```

Resume model, optimizer, scheduler, and RNG state by adding
`--resume-from /path/to/last.pt --resume-optimizer`. The supervisor takes the
single-GPU advisory lock, snapshots source/configuration, hashes annotations,
and records commands, PIDs, logs, telemetry, package versions and GPU details.
Per-epoch Val-Seen/Val-Unseen proxy metrics select `best_val_unseen.pt` by the
released CityNav benchmark's horizontal (`Pose.xy`) SR, SPL, then NE. A
separately labelled 3-D Euclidean diagnostic is also recorded but is never
used for checkpoint selection. Run full validation on that frozen checkpoint
with:

```bash
/home/ubuntu5/miniconda3/envs/SBFNav/bin/python scripts/evaluate_sbfnav.py \
  --config configs/sbfnav/main.yaml --checkpoint /path/to/best_val_unseen.pt \
  --split val_unseen --output-dir /path/to/new-output
```

Candidate coverage can be audited with `scripts/analyze_sbf_candidates.py`.

To diagnose the large OSR-to-SR gap or low-SPL successes, rerun a development
split with per-step diagnostics enabled and then classify the trajectories:

```bash
/home/ubuntu5/miniconda3/envs/SBFNav/bin/python scripts/evaluate_sbfnav.py \
  --config configs/sbfnav/main.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --split val_unseen \
  --output-dir /path/to/val_unseen_stepdiag \
  --record-step-diagnostics \
  --override data.root=/path/to/audited/data

/home/ubuntu5/miniconda3/envs/SBFNav/bin/python scripts/analyze_sbf_closed_loop.py \
  --predictions /path/to/val_unseen_stepdiag/predictions.jsonl \
  --output-dir /path/to/val_unseen_closed_loop_audit
```

The audit separates high/low-SPL final successes, episodes that entered the
20 m success region but later finished outside it, and episodes that never
reached the goal region. When step diagnostics are present it also measures
candidate switching and selector-score margins. This is analysis-only and does
not change navigation, stopping, or benchmark metrics.
For a full-split GPU audit, keep it under the same lock/provenance supervisor:

```bash
/home/ubuntu5/miniconda3/envs/SBFNav/bin/python scripts/supervise_experiment.py \
  --run-dir /home/ubuntu5/Workspace/hett-runs/sbfnav/<candidate-run> \
  --python /home/ubuntu5/miniconda3/envs/SBFNav/bin/python \
  --config configs/sbfnav/main.yaml --phase candidate-analysis \
  --checkpoint /path/to/checkpoint.pt \
  --evaluation-split val_seen --evaluation-split val_unseen \
  --analysis-prefix-fraction 0.5 --config-override data.root=/path/to/audited/data
```

The configurations under `configs/sbfnav/ablations/` inherit the main config
and implement the requested architecture/loss/K controls. Test-Unseen is
sealed by default. After full Val-Seen and Val-Unseen evaluation, create its
manifest with `scripts/freeze_sbfnav_test.py`; this binds the raw and effective
configuration (including overrides), checkpoint, validation metrics, seed and
single-use sentinel. Run the final evaluation through the supervisor with
`--phase eval --evaluation-split test_unseen --allow-test-unseen
--freeze-manifest <manifest>`. The evaluator rejects any hash/override mismatch
and atomically consumes the sentinel before reading Test-Unseen episodes.
That single full-split process also verifies the easy/medium/hard annotation
partition and emits all three difficulty rows; truncated or difficulty-only
Test-Unseen commands are rejected.

## Citation

```bibtex
@misc{ding2025historyenhancedtwostagetransformeraerial,
      title={History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation}, 
      author={Xichen Ding and Jianzhe Gao and Cong Pan and Wenguan Wang and Jie Qin},
      year={2025},
      eprint={2512.14222},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.14222}, 
}
```

## Acknowledgements

We would like to express our gratitude to the authors of the following codebase.

- [CityNav](https://github.com/water-cookie/citynav)
- [AVDN](https://github.com/eric-ai-lab/Aerial-Vision-and-Dialog-Navigation)
