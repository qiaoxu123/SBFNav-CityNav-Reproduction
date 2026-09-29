# SBFNav epoch-7 Val-Unseen closed-loop diagnostic

This report records a diagnostic-only evaluation of the independent,
paper-based SBFNav reimplementation. No model architecture, loss, controller,
metric, or checkpoint was changed. Test-Unseen was not accessed.

## Reproducibility record

- Date: 2026-09-27 (Asia/Shanghai)
- Evaluated source commit: `3e4a0a439711131d671b488dbbce0526274689c2`
- Branch: `codex/reproduce-sbfnav`
- Checkpoint completed epoch: 7
- Checkpoint SHA-256:
  `ba8b48217ed948380ee8939df89b9eaeb8352e79f1bc5f9d0acc3a493e645564`
- Configuration: `configs/sbfnav/main.yaml`
- Data: audited recovered Revised CityNav, seed 0
- Split/subset: first 256 records of Val-Unseen
- Subset caveat: all 256 records are from `birmingham_block_5`; this is a
  targeted diagnostic subset, not an evenly sampled estimate of the complete
  four-map Val-Unseen split.

Before evaluation, an already-running epoch 8–20 resume was found to conflict
with the diagnostic-only instruction. The user-level unit
`sbfnav-main-e20-resume-e7-s0-20260927.service` was stopped cleanly and became
`inactive`; its run directory and checkpoints were not deleted or modified.

Syntax checking passed:

```bash
/home/ubuntu5/miniconda3/envs/SBFNav/bin/python -m py_compile \
  scripts/evaluate_sbfnav.py \
  scripts/analyze_sbf_closed_loop.py
```

The first requested test invocation produced 19 passes and one data-read
failure because the shared `processed_citynav/citynav_train_seen.json` contains
the previously documented invalid UTF-8 byte. Re-running the same tests against
the audited recovered input passed 20/20; no code change was required:

```bash
SBFNAV_TEST_DATA_ROOT=/home/ubuntu5/Workspace/hett-runs/sbfnav/recovered_inputs_20260927/data \
  /home/ubuntu5/miniconda3/envs/SBFNav/bin/python -m pytest \
  tests/test_sbf_selector.py tests/test_sbf_navigation.py \
  tests/test_sbf_metrics.py tests/test_sbf_candidates.py -q
```

Direct script execution initially failed before argument parsing with
`ModuleNotFoundError: sbfnav`, without creating the output directory. The same
requested command was therefore run with the repository root explicitly added
through `PYTHONPATH=.`:

```bash
PYTHONPATH=. /home/ubuntu5/miniconda3/envs/SBFNav/bin/python \
  scripts/evaluate_sbfnav.py \
  --config configs/sbfnav/main.yaml \
  --checkpoint /home/ubuntu5/Workspace/hett-runs/sbfnav/main_e20_benchmark2d_resume_e5_s0_20260927/training/checkpoints/last.pt \
  --split val_unseen --max-episodes 256 --record-step-diagnostics \
  --output-dir /home/ubuntu5/Workspace/hett-runs/sbfnav/e7_valunseen_stepdiag_256 \
  --override data.root=/home/ubuntu5/Workspace/hett-runs/sbfnav/recovered_inputs_20260927/data

/home/ubuntu5/miniconda3/envs/SBFNav/bin/python \
  scripts/analyze_sbf_closed_loop.py \
  --predictions /home/ubuntu5/Workspace/hett-runs/sbfnav/e7_valunseen_stepdiag_256/predictions.jsonl \
  --output-dir /home/ubuntu5/Workspace/hett-runs/sbfnav/e7_valunseen_closed_loop_audit_256
```

The compact outputs and complete 256-record prediction/audit JSONL files are
tracked under `artifacts/sbfnav/e7_valunseen_closed_loop_diagnostic_256/`.

## Metrics and four-way decomposition

The benchmark-2D metrics are NE 62.61 m, SR 10.94%, OSR 27.34%, SPL 7.14%,
and progress 65.57 m. The OSR−SR gap is exactly 42/256 = 16.40625 percentage
points.

| Category | Episodes | Fraction | Mean path ratio | Mean switches | Mean selector margin |
|---|---:|---:|---:|---:|---:|
| A: success, SPL ≥ 0.5 | 24 | 9.38% | 1.47 | 3.54 | 0.530 |
| B: success, SPL < 0.5 | 4 | 1.56% | 2.77 | 1.75 | 0.531 |
| C: oracle success, final failure | 42 | 16.41% | 2.01 | 3.64 | 0.592 |
| D: never reached 20 m | 186 | 72.66% | 1.56 | 2.61 | 0.676 |

All 42 C episodes entered the 20 m radius and subsequently left it. Their
first-success step has mean 5.38, median 5, range 0–12, and histogram
`{0:1, 1:2, 2:4, 3:3, 4:8, 5:7, 6:5, 7:3, 8:2, 9:2, 10:2, 11:1, 12:2}`.
Final target distance averages 46.74 m (median 40.80, maximum 138.00), or an
average 26.74 m beyond the success radius. Twenty-eight terminate by stagnation
and 14 by the horizon.

C has a mean 3.64 candidate switches over 20 m, but this is almost identical
to A's 3.54. Three C episodes have zero such switches and ten have at most one.
Its per-episode mean switch-distance mean is 9.87 m, while the mean of the
per-episode maximum is 39.23 m. Mean selector margin is 0.592 (median 0.618),
although the minimum within an episode averages 0.083. Candidate switching and
momentary uncertainty contribute to individual failures, but cannot by
themselves explain the systematic OSR−SR gap.

The four B episodes have path ratios 2.50–3.08. Defining a direction reversal
as an angle greater than 90 degrees between consecutive non-zero horizontal
movement vectors, they contain 23 reversals in 37 comparisons (62.2%); every B
episode has at least two, and two have 8 and 10. This is strong evidence that
oscillation/inefficient closed-loop motion, not merely failure to localize the
goal, depresses SPL.

For D, minimum target distance averages 57.29 m (median 51.07, minimum 20.18).
At the final step, the Top-16 field oracle is within 20 m for 115/186 = 61.83%
of episodes, but the selector-selected candidate is within 20 m for 0/186.
Thus 115 final states are proposal-good/selection-bad by this diagnostic, while
71/186 = 38.17% have no field proposal within 20 m. D is mixed, but the larger
share is selector failure rather than SBF proposal failure. This final-step
decomposition does not prove that a good proposal existed at every earlier
step.

## Representative priority cases

Selector margin below is the mean top-1 minus top-2 logit margin over recorded
steps.

| Episode | First hit | Final distance | Path ratio | Switches | Margin | Observation |
|---|---:|---:|---:|---:|---:|---|
| `birmingham_block_5/44/5` | 4 | 138.00 m | 1.84 | 2 | 0.660 | Entered once, then departed 118.00 m beyond the radius; stagnation. |
| `birmingham_block_5/18/5` | 4 | 107.73 m | 2.52 | 6 | 0.256 | Stayed inside through step 5, then oscillated away to the horizon. |
| `birmingham_block_5/21/0` | 4 | 96.93 m | 1.60 | 5 | 0.188 | Reached 2.82 m, then left; very small minimum margin (0.0004). |
| `birmingham_block_5/15/2` | 2 | 89.72 m | 3.51 | 6 | 0.864 | High average confidence did not prevent departure; stagnation. |
| `birmingham_block_5/1/1` | 0 | 88.91 m | 9.88 | 3 | 0.780 | Began inside 20 m but navigated away because no success stop exists. |

The first and fifth examples show why selector uncertainty is not a sufficient
explanation: both have relatively large mean margins, and the fifth is already
successful at the initial pose.

## Diagnosis and next-step priority

The primary cause of the OSR−SR gap is **missing stopping/termination
behavior**. The metric gap is exactly the C population, every C trajectory
leaves after entering the valid radius, and one severe example starts inside
the radius. This conclusion does not rely on using GT distance for inference;
GT distance is used only retrospectively for diagnosis.

Secondary findings are selector/candidate temporal instability and controller
overshoot/oscillation. For D, the final-proposal decomposition specifically
identifies selector failure in 61.83% of never-reached episodes and proposal
failure in 38.17%. No direct evidence here implicates observation/map
construction.

Recommended investigation order, without applying any change in this stage:

1. **C — add non-oracle stopping:** directly targets the entire measured
   OSR−SR gap; it must use learned confidence/progress/stability, not GT range.
2. **A — temporal consistency/hysteresis:** reduces abrupt target changes and
   complements a learned stop rule, especially in high-switch C cases.
3. **E — continue epochs 8–20:** may improve the D selector failures, but
   training alone cannot guarantee stopping after arrival. Do not resume until
   the diagnostic decision is made.
4. **B — modify controller:** the 50 m horizontal step and frequent B reversals
   make overshoot plausible, but this is a broader semantic change than adding
   an explicit non-oracle termination mechanism.
5. **D — inspect observation construction:** lowest priority because final
   candidate oracle coverage is often good and no alignment anomaly was found.
