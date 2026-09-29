# SBFNav reproduction results

This is a living results record for the independent paper-based
reimplementation. Empty evaluation cells mean “not run”, not zero. Test-Unseen
must remain sealed until a configuration/checkpoint freeze manifest exists.

## Stage A — data, coordinates, and planning state

Completed 2026-09-27 at git parent `7fe5e2d` using Revised CityNav and seed 0.

- Schema and target/CityRefer link checks passed on train, Val-Seen, and
  Val-Unseen. Development code rejects Test-Unseen by default.
- Split counts: 21,878 / 2,470 / 2,697 for train/Val-Seen/Val-Unseen. The 4
  Val-Unseen maps are disjoint from all seen maps.
- World→normalized→world maximum error across the 21 visualized targets was
  `2.85e-14 m`; real-trajectory unit tests enforce `<0.05 m` (half a 0.1 m
  source pixel).
- Generated 21 diagnostic panels (7 per development split), each showing the
  nine state channels, valid mask, referenced landmarks, exploration/trajectory,
  and normalized Gaussian field target. Field sums were
  `[0.99999994, 1.00000012]`.
- External artifacts:
  `/home/ubuntu5/Workspace/hett-runs/sbfnav/phase_a_visualizations_v2/`.
- Visual inspection confirmed north-up maps, heading-dependent footprints,
  accumulated exploration, static landmark alignment, referenced-polygon
  highlighting, and target-centered supervision on train/seen/unseen examples.
- Tests: `15 passed` for coordinates, data loading/split sealing, planning
  channels, Gaussian targets, and progress targets.

No model checkpoint or navigation metric exists at this stage.

### Input integrity incident and recovery

On 2026-09-27, after Stage A had already validated the source annotation, the
shared `processed_citynav/citynav_train_seen.json` developed three corrupted
bytes inside one floating-point trajectory record. The shared input was not
modified. A private, byte-for-byte copy was repaired by recovering the matching
record from official Original CityNav; exactly three bytes differ from the
then-corrupted shared file. The repaired file parses to 21,878 episodes and has
SHA-256 `c24a16ddaab820d32bb6ce44a373bc2e2b8f6d88e8d08fdff9ba165f190388b2`.
Subsequent runs use the explicit audited override
`data.root=/home/ubuntu5/Workspace/hett-runs/sbfnav/recovered_inputs_20260927/data`.
The supervisor records this override and all effective annotation hashes, and
its source snapshot links to that effective root. Validation and Test-Unseen
annotations remain unmodified links to the supplied Revised CityNav files.

## Stage C — one-batch and resume smoke

Completed 2026-09-27 with `configs/sbfnav/debug.yaml`, Revised CityNav, seed 0,
one train record and one sampled prefix. This is an execution check, not a
performance result.

- Offline frozen SigLIP text and vision towers, 9-channel map, SBF, NMS,
  16-candidate selector, all losses, bf16 backward, optimizer step, atomic
  checkpoint and reload ran on the RTX 5090.
- Epoch 1 total loss was 10.8078; an actual optimizer/RNG resume continued from
  `global_step=1` to epoch 2 / step 2 and produced total loss 9.4792.
- Resume testing found and fixed CUDA-map-location migration of the CPU RNG
  state. Cached fp16 text features are now promoted to fp32 at inference after
  a closed-loop smoke exposed the dtype boundary.
- Peak allocated GPU memory was 0.99 GB in this tiny debug configuration.
- One Val-Seen closed-loop episode completed with the epoch-2 smoke checkpoint.
  Its metrics are intentionally not interpreted because the checkpoint saw one
  sample for two updates.
- External artifacts:
  `/home/ubuntu5/Workspace/hett-runs/sbfnav/single_batch_smoke_20260927/` and
  `/home/ubuntu5/Workspace/hett-runs/sbfnav/single_episode_eval_20260927_r2/`.

The final integration rerun used a two-epoch planned schedule, intentionally
stopped after epoch 1 at LR `5e-4`, then restored model, optimizer, scheduler,
CPU/CUDA RNG and `global_step=2`. It completed epoch 2 at step 4, inherited the
parent best-Val-Unseen checkpoint when epoch 2 did not improve, and completed
both supervised post-resume evaluations. Artifacts are in
`debug_planned_e1_s0_20260927/` and `debug_planned_resume_e2_s0_20260927/`.

The full-width main configuration passed a 32-example forward/backward and
closed-loop validation smoke at about 1.9 GiB peak allocated GPU memory after
tensor-native frozen-SigLIP preprocessing. A 256-example, two-epoch cache
benchmark recorded 1,609 cache hits on epoch 2 and stable finite losses. A
batch-16/effective-batch-16 run used 3.22 GiB and completed the same 256-example
epoch faster than batch-4 accumulation; the frozen main configuration therefore
uses batch 16, no accumulation, vision batch 128, and a bounded CPU-fp16 cache.
Runs: `main_smoke_opt_s0_20260927/`, `main_cache_bench_s0_20260927/`, and
`main_batch16_bench_s0_20260927/` under the external SBFNav run root.
The first formal launch was intentionally stopped after 10/5,470 batches and
before any epoch/checkpoint when telemetry showed the unique training text was
still being encoded online. Its intact audit run is `main_e20_s0_20260927/`.
The corrected pipeline batch-precomputes every unique frozen instruction and
landmark-name feature before optimization; no training/evaluation result from
the stopped launch is used.

### Formal-run audit: NMS defect found before completion

The second formal launch, `main_e20_r2_s0_20260927/`, was manually stopped
after epoch 2 to investigate its 32-episode validation proxy.  Its epoch-2
proxy was Val-Seen NE/SR/OSR/SPL `47.96/25.00/28.13/21.59` and Val-Unseen
`46.42/18.75/18.75/16.61`; these are diagnostics, not final results.

The audit found that the initial local-maximum implementation filled a field
with fewer than K strict modes using adjacent, unsuppressed logits.  It thus
collapsed most Top-K proposals around one mode.  On the same evenly spaced 256
Val-Unseen episodes, the same epoch-2 checkpoint, and the same 50% teacher
prefix, replacing only this operation with greedy 3×3 suppression changed
field-only 20 m coverage from `30.47/31.25/31.25%` at K=`1/5/16` to
`30.47/62.89/89.06%`; oracle distance changed from `33.20 m` to `13.99 m`.
This matches the paper's stated purpose for NMS and explains why the earlier
selector accuracy was misleadingly high on near-duplicate candidates.

Applying corrected proposals to the old selector lowered the 32-episode
closed-loop proxy (Val-Seen SR `15.625%`, Val-Unseen SR `9.375%`) while final
candidate 20 m oracle coverage rose to `90.625%` and `75.00%`, respectively.
This is expected distribution mismatch and proves that the old selector must
not be resumed.  The two-epoch checkpoint SHA-256 is
`e7593c943a65e011056ab5bfb41fd263a5777658bbe6c6dd8ff3d88cdee08e1b` and is
retained only for audit.  Formal training will restart from random
initialization after the corrected NMS tests pass.

The corrected 32-record overfit gate subsequently completed all 30 epochs at
git `9368dcf`.  Total loss fell `10.50219→5.91827` (ratio `0.5635`, required
at most `0.8`), selector nearest-candidate accuracy reached `96.875%`, and
training Top-16 candidate coverage at 20 m reached `100%`.  All 62 tests passed
against the audited recovered data root.  The last checkpoint is
`/home/ubuntu5/Workspace/hett-runs/sbfnav/overfit32_greedynms_e30_s0_20260927/training/checkpoints/last.pt`
(SHA-256 `ba120e79b97de7cb528bd37d19fe34d9ccc3146d4afc68008e1832f4bf8147f8`).
Its 32-episode development evaluations are execution smoke checks only; a
model fit to 32 Train-Seen records is not expected to generalize.

### Formal-run metric audit and pause

The corrected-NMS formal run
`main_e20_greedynms_r3_s0_20260927` completed five epochs before being paused
for an evaluation-protocol audit. Its epoch-5 evenly spaced 128-episode proxy
reported Val-Seen `benchmark_2d` NE/SR/OSR/SPL
`35.73/42.19/57.81/36.93`, compared with the revised-paper
`32.5/43.24/53.77/40.93`. The separately computed 3-D diagnostic was
`38.10/33.59/40.63/27.94`; comparing that SR to the paper table had therefore
created a spurious 8.59-point penalty.

The audit confirmed that released CityNav/HETT leaderboard evaluation reduces
poses to `xy`, whereas the initial reproduction used 3-D SR for checkpoint
selection. The target z annotations are near ground/object surfaces while the
controller maintains a legal flight altitude, so the two success definitions
are materially different. Output keys and checkpoint selection are corrected
to use `benchmark_2d`, while retaining `diagnostic_3d`. Across epochs 1–5,
both definitions happened to choose epoch 2, so the epoch-5 model, optimizer,
scheduler, and RNG state can be resumed without invalidating the learned
weights. No Test-Unseen data was loaded during this audit.

The corrected resume path was exercised through the repository supervisor in
`resume_metricfix_smoke_e6_s0_20260927_r2`: it restored epoch 5 / global step
27,350 with optimizer, scheduler, and RNG, performed one isolated update, and
completed one-episode evaluations on both development splits. The inherited
epoch-2 best checkpoint stayed byte-identical (SHA-256
`1862afb6f74ccf80747a6a47ac30b5994e504d4d832c6868c3b3fc92ce27572c`), while
its legacy metadata was migrated to a `benchmark_2d` selection key of
`[28.125, 23.360944, -46.543645]`. All 64 tests then passed. The smoke's zero
learning rate is expected and non-evidentiary because its one-record loader
intentionally changes the scheduler's total-step denominator; the formal
resume retains the original 5,470-batch epoch geometry and therefore the
original 20-epoch schedule.

Formal training resumed from epoch 5 at commit `b1fabbe` with the original
optimizer, scheduler, RNG, 5,470 batches per epoch, and audited data override.
Epoch 6 completed at global step 32,820 with total loss `7.20189` and selector
nearest-candidate accuracy `50.27%`. On the fixed evenly spaced 128-episode
proxy, benchmark-2D NE/SR/OSR/SPL was
`34.18/43.75/55.47/36.26` on Val-Seen and
`48.95/21.875/39.84/17.74` on Val-Unseen. These are checkpoint-selection
proxies, not full-split final results. The revised paper values are
`32.5/43.24/53.77/40.93` and `49.2/20.24/35.41/19.15`, respectively. Epoch 2
remains the current best proxy checkpoint because its Val-Unseen benchmark SR
was `28.125%`; its SHA-256 remains
`1862afb6f74ccf80747a6a47ac30b5994e504d4d832c6868c3b3fc92ce27572c`.

### Epoch-7 full-validation checkpoint

At the user's requested inspection point, training was allowed to finish epoch
7 and its proxy validation, then was stopped only after the atomic `last.pt`
could be loaded with `completed_epoch=7`. The checkpoint is
`main_e20_benchmark2d_resume_e5_s0_20260927/training/checkpoints/last.pt`, at
global step 38,290, with SHA-256
`ba8b48217ed948380ee8939df89b9eaeb8352e79f1bc5f9d0acc3a493e645564`.
Training is paused at this inspection point; the original 20-epoch setting is
a reproduction assumption, not a paper-published value.

The current committed evaluator ran each complete revised validation split
once. The supervised run `e7_fullval_s0_20260927` completed successfully at
git `5be2330`; both metric files use effective-configuration SHA-256
`0f2f86399f36fbd6e6e17efbbf38d6743379c83da5836f217dccd03148c37506`,
seed 0, and the checkpoint hash above.

| Split | Episodes | NE (m) | SR (%) | OSR (%) | SPL (%) | Progress (m) |
|---|---:|---:|---:|---:|---:|---:|
| Val-Seen | 2,470 | 34.00 | 40.61 | 58.34 | 34.06 | 76.88 |
| Val-Unseen | 2,697 | 49.79 | 19.84 | 37.37 | 15.25 | 69.59 |

The separately labelled 3-D diagnostics were Val-Seen NE/SR/OSR/SPL
`36.33/32.19/37.98/26.56` and Val-Unseen
`51.18/16.54/22.28/12.74`. They are not used for checkpoint selection or the
paper comparison.

The same single-pass evaluation produced the following benchmark-2D
difficulty rows. Counts sum exactly to each complete split.

| Split / difficulty | Episodes | NE (m) | SR (%) | OSR (%) | SPL (%) |
|---|---:|---:|---:|---:|---:|
| Val-Seen easy | 821 | 33.78 | 40.68 | 62.61 | 29.79 |
| Val-Seen medium | 803 | 33.14 | 43.46 | 59.65 | 37.37 |
| Val-Seen hard | 846 | 35.02 | 37.83 | 52.96 | 35.06 |
| Val-Unseen easy | 744 | 49.63 | 22.31 | 43.95 | 14.52 |
| Val-Unseen medium | 927 | 50.02 | 17.37 | 35.92 | 13.49 |
| Val-Unseen hard | 1,026 | 49.70 | 20.27 | 33.92 | 17.37 |

The separate full teacher-prefix proposal audit ran at prefix fraction 0.5
through the repository supervisor in `e7_full_candidates_s0_20260927`, git
`c623481`. It wrote one prediction record per episode and verified the same
checkpoint/effective-configuration hashes.

| Split / proposals | Oracle distance (m) | Top-1@20 (%) | Top-5@20 (%) | Top-16@20 (%) |
|---|---:|---:|---:|---:|
| Val-Seen field peaks | 11.84 | 44.45 | 77.09 | 94.29 |
| Val-Seen + landmarks | 11.36 | 44.45 | 77.09 | 94.29 |
| Val-Unseen field peaks | 15.09 | 26.51 | 61.18 | 87.02 |
| Val-Unseen + landmarks | 14.33 | 26.51 | 61.18 | 87.02 |

Landmark centroids are appended after the K=16 field peaks, so they improve
the oracle over the complete candidate set but intentionally do not redefine
the field Top-1/5/16 coverage rows.

Twenty representative closed-loop panels (five metric-selected successes and
five failures from each validation split) were rendered and visually checked
for world/canvas alignment. Each shows accumulated RGB, trajectory, final
candidates and selector choice, referenced-landmark mask, exploration, and the
GT marker explicitly labelled as visualization-only. They are stored outside
Git at
`/home/ubuntu5/Workspace/hett-runs/sbfnav/e7_case_visualizations_v2_s0_20260927/`;
the per-split manifests bind every image to its episode and metrics.

A repository copy of the reasonably sized epoch-7 audit material is indexed at
`artifacts/sbfnav/README.md`. It includes the Val-Seen/Val-Unseen metrics, training/evaluation logs,
provenance, checksums, and the 20 panels. Five large raw per-episode prediction
JSONL files were omitted when this standalone reproduction repository was
created; they are not required to reproduce the reported aggregate metrics. Checkpoints, weights, caches,
datasets, and source snapshots remain outside Git. The resumed formal-run copy
is explicitly a point-in-time partial snapshot rather than a final result.

The diagnostic-only 256-record Val-Unseen closed-loop decomposition is reported
separately in `docs/sbfnav_closed_loop_diagnostic_256.md`. That subset contains
only `birmingham_block_5`, so its metrics must not replace the complete-split
results above.

## Stage B — 32-record overfit gate

Completed 2026-09-27 at git commit `003a7f6`, Revised CityNav repaired-copy
SHA-256 documented above, seed 0. The supervised run used 32 training records,
one deterministic teacher prefix per record, batch size 2, and 30 epochs.

- Total loss fell from 9.90370 to 4.02662 (ratio 0.4066; required ratio at most
  0.8). Field loss fell 6.38528→3.14025.
- Selector nearest-candidate training accuracy rose from 31.25% to 96.875%; the
  altitude Smooth-L1 and progress MSE also converged without NaN.
- The run completed its post-training 32-episode Val-Seen and Val-Unseen
  closed-loop smoke evaluations. Those small prefix-selected metrics are not
  model-selection or paper-comparison evidence.
- Checkpoint:
  `/home/ubuntu5/Workspace/hett-runs/sbfnav/overfit32_e30_s0_20260927_r3/training/checkpoints/last.pt`,
  SHA-256 `25be1fa883f6778a21b26b8e982f644cf308573ceff17634732a01c6ccc56820`.
- Supervisor service/PID at launch:
  `sbfnav-overfit-r3-s0-20260927.service` / `3743678`; final supervisor status
  is `complete`. Commands, logs, source snapshot, input hashes and telemetry
  are retained in the run root.

## Navigation comparison

| Method | Val-Seen NE / SR / OSR / SPL | Val-Unseen NE / SR / OSR / SPL | Test-Unseen NE / SR / OSR / SPL |
|---|---|---|---|
| Official SBFNav (revised) | 32.5 / 43.24 / 53.77 / 40.93 | 49.2 / 20.24 / 35.41 / 19.15 | 38.5 / 32.46 / 49.82 / 30.59 |
| Our reproduction (epoch-7 inspection) | 34.00 / 40.61 / 58.34 / 34.06 | 49.79 / 19.84 / 37.37 / 15.25 | sealed |
| Difference (ours − paper) | +1.50 / −2.63 / +4.57 / −6.87 | +0.59 / −0.40 / +1.96 / −3.90 | — |

Exact development commands are retained in the external runs' `commands.json`.
In compact form, both used `configs/sbfnav/main.yaml`, seed 0, the audited
`data.root=/home/ubuntu5/Workspace/hett-runs/sbfnav/recovered_inputs_20260927/data`
override, and the epoch-7 checkpoint above. No Test-Unseen annotation was read
or hashed by either run.
