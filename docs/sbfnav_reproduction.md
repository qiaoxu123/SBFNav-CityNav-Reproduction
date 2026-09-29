# SBFNav paper-based reimplementation specification

> Status: implementation specification, 2026-09-27. This repository is an
> independent **paper-based reimplementation** of “Map the Possibilities:
> Spatial Belief Fields for Language-Goal Aerial Navigation” (arXiv:2609.05841v1).
> It is not the authors' official implementation and must never be described as
> one.

## Sources and reproducibility boundary

The specification was checked against all three arXiv v1 artifacts on
2026-09-27:

- abstract/HTML: <https://arxiv.org/abs/2609.05841> and
  <https://arxiv.org/html/2609.05841>;
- PDF SHA-256:
  `d87c7c717b574a1eed604a7c2489e257b6714496c27a88e3a817b2bd6c3604f5`;
- TeX source archive SHA-256:
  `6112ea4f4e4a40c841c35d61c73ad98d6a38cc6ca92fa034df6af2fa5cb6f1f3`.

The paper repeatedly refers to Supplement B.4 (loss parameters), B.6
(controller), B.7 (latency), and Supplements A–D. None is present in the PDF,
HTML, or v1 source archive. `00README.json` lists only
`AnonymousSubmission2027.tex`; its appendix input is commented out. Therefore,
all details listed as assumptions below are genuinely unavailable in the
released paper. Concrete choices live in `configs/sbfnav/*.yaml` and are
catalogued in `docs/sbfnav_assumptions.md`.

## Problem definition

At navigation step \(t\), the agent receives an egocentric top-down RGB-D
observation \(o_t\), world pose \(s_t=(x_t,y_t,z_t,\psi_t)\), instruction
\(I\), accumulated history \(H_t=\{(o_\tau,s_\tau)\}_{\tau=1}^t\), and the
polygon/name prior for instruction-referenced landmarks \(G_I\). It predicts a
metric 3-D waypoint and executes one bounded action before observing again.
Success is final 3-D Euclidean distance at most 20 m from the target.

The learned decision is

\[
M_t=\mathrm{UpdateMap}(H_t,G_I),\quad
C_t=G_\theta(M_t,I),\quad
c_t^*=\arg\max_{c_i\in C_t}S_\omega(c_i\mid M_t,I).
\]

The selector provides horizontal position, an auxiliary head provides
altitude, and a deterministic controller tracks the resulting waypoint. At
inference, target coordinates and target-distance signals are forbidden.

## World-aligned planning state

Each city block remains north-up and is padded symmetrically to a square in
metric extent, then resampled to 224×224. Padding is invalid map area, not
observed black imagery. A stored affine transform maps every state layer using
identical geometry.

The exact nine channels stated in the paper, in implementation order, are:

| Channel | Meaning | Availability |
|---:|---|---|
| 0–2 | accumulated observed RGB | observation history only |
| 3 | normalized world height reconstructed from depth | observation history only |
| 4 | contours of all available named landmarks | static CityRefer prior |
| 5 | mask of instruction-referenced landmarks | instruction-conditioned static prior |
| 6 | current observation footprint | current pose/altitude/heading |
| 7 | cumulatively explored area | observation history |
| 8 | trajectory history | pose history |

The RGB and height channels are zero outside channel 7. Channel 4 is the
outline/contour rather than a filled instance mask; channel 5 is filled
referenced polygons. Invalid square padding is represented by a separate
validity mask used by the field target/loss and candidate generation, not a
tenth model channel.

### Coordinate convention

Local RGBD rasters use a north-up affine transform with 0.1 m pixels. World
\(+x\) maps to increasing image column and world \(+y\) maps to decreasing
image row. For bounds \([x_{min},x_{max}]\times[y_{min},y_{max}]\), square side
\(S=\max(x_{max}-x_{min},y_{max}-y_{min})\), and symmetric padding, normalized
coordinates obey

\[
u_x=(x-x_0)/S,\quad u_y=(y_1-y)/S,
\]

where \(x_0=x_{min}-(S-(x_{max}-x_{min}))/2\) and
\(y_1=y_{max}+(S-(y_{max}-y_{min}))/2\). The inverse is
\(x=x_0+u_xS, y=y_1-u_yS\). Raster and 28×28 field cell centers use
\(u_x=(c+0.5)/W, u_y=(r+0.5)/H\); edge coordinates are used only for bounds.
Maps are never agent-rotated: heading affects the projected current footprint,
while all output coordinates remain in the common north-up world frame.

The paper does not specify asymmetric-versus-symmetric square padding, cell
center convention, line thickness, height normalization, or observation fusion
when footprints overlap. Those are reproduction assumptions and are tested for
round-trip consistency.

## Spatial Belief Field

Required paper-defined behavior:

1. A compact residual CNN maps the 9×224×224 state to a spatial feature grid.
2. The frozen `google/siglip-base-patch16-224` text tower encodes the complete
   instruction into token features \(E_I=[e_1,\ldots,e_L]\); the pooled-only
   representation is not the main model.
3. Spatial map features query instruction tokens through multi-head
   cross-attention. The residual update is
   \(\widetilde F=F+\tanh(\gamma)A(F,E_I)\), with scalar \(\gamma\) initialized
   near zero.
4. A convolutional decoder outputs logits \(\Phi\in\mathbb R^{28\times28}\).
   Masked spatial softmax gives a normalized target-plausibility distribution
   over valid field cells.
5. Every field cell maps through the retained affine transform to metric world
   coordinates.

SigLIP stays frozen in the main configuration, with an explicit opt-in switch
for fine-tuning. Exact residual blocks, channel widths, attention dimensions,
head count, decoder, text length, and initial \(\gamma\) are not published.

## Field and ranking supervision

The field target is a normalized Gaussian centered on the metric ground-truth
goal. Distances are computed between world-space cell centers and the goal, so
one Gaussian scale has the same meaning across differently sized blocks.
Invalid padded cells receive zero mass. If no valid cell remains, the sample is
rejected before the loss rather than producing an all-mask NaN.

The paper defines soft cross entropy

\[
L_{field}=-\sum_u Q(u)\log P(u).
\]

For candidates with metric goal distances \(d_i\) and selector logits \(r_i\):

\[
q_i=\operatorname{softmax}(-d_i/\tau),\quad
L_{list}=-\sum_iq_i\log\operatorname{softmax}(r)_i.
\]

Good candidates have \(d_i\le R_{good}\), bad candidates have
\(d_i\ge R_{bad}\). The pairwise term is the mean
\([m+r_j-r_i]_+\) over valid good/bad pairs and is exactly zero when either set
is empty. The hard-nearest loss is negative log probability of
\(\arg\min_i d_i\). Thus

\[
L_{rank}=L_{list}+\lambda_mL_{margin}+\lambda_hL_{hard},
\]

\[
L=L_{field}+\lambda_rL_{rank}+\lambda_zL_z+\lambda_pL_{progress}.
\]

\(L_z\) is smooth-L1. The progress loss is trajectory-progress regression used
only as representation regularization; progress is not an inference input,
candidate feature, stopping rule, or goal-selection signal. The target
definition, head topology, all thresholds, temperature, margins, weights, and
Gaussian width are absent with Supplement B.4 and are assumptions.

All masked reductions use finite log-softmax operations, explicitly handle
empty candidate/pair sets, and log `field`, `list`, `margin`, `hard`, `altitude`,
`progress`, `rank`, and `total` separately.

## Candidate proposal

NMS is applied to field logits, not probabilities, and the highest K local
maxima are retained. The paper's main configuration is K=16; a controlled
selector diagnostic uses K=10. Peak cell centers are transformed to world
coordinates and unioned with referenced-landmark polygon centroids. Invalid or
out-of-bounds peaks are masked before Top-K; landmark centroids are clipped only
for floating-point boundary noise and otherwise rejected. Near-duplicate
candidates are deterministically merged while retaining provenance
(`field_peak`, `landmark_centroid`, or both).

NMS neighborhood/radius, plateau tie-breaking, deduplication radius, and the
behavior when fewer than K valid local maxima exist are not published.

Candidate-oracle analysis reports minimum target distance and Recall@1/5/K at
10 m and 20 m (plus configured thresholds), separately for field-only and the
augmented pool. The paper reports Language-SBF field Recall@10(20 m) of 72.7%
on Val Unseen and 80.4% on Test Unseen; Recall@32 is 90.9% and 94.6%.

## Semantic-geometric selector

For each candidate, the selector receives only:

- a candidate-centered local accumulated RGB/map crop, with unobserved pixels
  masked, encoded by the frozen SigLIP-B/16 vision tower;
- complete-instruction token features;
- instruction-referenced landmark-name text features;
- explicit candidate-to-agent and candidate-to-each-landmark geometry
  \([\Delta x,\Delta y,d,\sin\beta,\cos\beta]\), expressed in world axes with
  distances/displacements normalized only by declared configuration constants;
- candidate/source-validity padding masks.

Projected visual, geometry, landmark, and instruction tokens plus a learned
`[SCR]` token pass through a lightweight Transformer. A linear head on `[SCR]`
produces one logit per candidate, and the maximum is selected.

The selector must not receive the field logit, probability, rank, or any
monotone transform thereof. Candidate order is deterministically shuffled
during training and a regression test perturbs SBF scores while holding
coordinates/features fixed to prove selector invariance. Source type is kept
for analysis but is not embedded in the main selector.

The crop's physical size, interpolation/masking details, SigLIP output choice,
landmark aggregation, transformer topology, and feature dimensions are not in
the paper.

## Altitude and receding-horizon inference

The selected candidate supplies horizontal position. A separate auxiliary head
\(f_z(M_t,I)\) predicts vertical position and is supervised with smooth-L1.
The prediction is clamped to declared per-map legal altitude bounds before
forming \(w_t=(x_t^*,y_t^*,z_t^*)\).

Inference follows Algorithm 1: append the current observation/pose, update the
accumulated map, regenerate candidates, rerank, predict altitude, execute one
bounded proportional-control step, and repeat up to \(T_{max}\). There is no
open-loop execution of a single static prediction. Ground truth and distance
to target are unavailable to all stopping logic. Controller gains, horizontal
and vertical step limits, yaw handling, horizon, altitude convention, and
non-oracle stopping behavior were deferred to missing Supplement B.6.

## Data and local schema audit

The local annotations are lists whose records contain `area`, `block`,
`object_ids`, `ann_ids`, `descriptions`, 5/6-DoF `trajectory`,
`marker_positions`, `target_positions`, scores, and split. CityRefer provides
object center/dimensions/contours/descriptions plus processed target, landmark,
and surrounding-object phrases. Raster TIFs hold georeferenced world height;
paired PNGs hold north-up RGB at 0.1 m/pixel.

Split-JSON instruction text is authoritative. It is not always byte-identical
to the same description index in CityRefer `objects.json` (826 records across
the four revised splits), whereas object IDs, description indices, target
coordinates, and processed landmark links remain aligned. The loader reports
these text differences without replacing revised text with stale CityRefer
text; this also avoids the legacy HETT `Episode.target_description` behavior.
Processed landmark strings are likewise not guaranteed to equal CityRefer's
canonical names (`Rd`/`Road` and optional `building` suffixes occur). Resolution
uses same-map canonical-name exact matching after normalization, then a
deterministic normalized Levenshtein fallback, matching the existing HETT
text-only strategy. Requested→resolved pairs remain available for audit; no
target coordinate participates in resolution.

| Data version | train_seen | val_seen | val_unseen | test_unseen |
|---|---:|---:|---:|---:|
| Revised, local | 21,878 | 2,470 | 2,697 | 5,281 |
| Original, local | 22,002 | 2,498 | 2,826 | 5,311 |

The revised data spans 24 train maps, 23 val-seen maps, 4 val-unseen maps, and
6 test-unseen maps. Unseen map sets are disjoint from train/val-seen; no exact
episode identifier crosses splits. The paper states 32,637 instructions over
5,850 target objects and says 13 Birmingham plus 33 Cambridge blocks feed the
observation simulator. The supplied CityRefer annotations actually cover 34
maps and the RGBD directory contains 40 map pairs. This discrepancy is recorded
rather than silently reconciled.

Development loaders reject `test_unseen`; final-test loading requires an
explicit frozen-evaluation flag and a freeze manifest containing configuration
and checkpoint hashes.

## Evaluation protocol and metrics

The [CityNav paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.pdf)
describes a 20 m spherical radius, and SBFNav says Euclidean distance without
stating the dimensions. The released CityNav/HETT navigation evaluators that
produce the benchmark tables, however, explicitly reduce poses and targets to
`xy` before computing NE, SR, OSR, path length, and SPL. In particular,
`multiagent/env.py` evaluates `[pose.xy ...]` against an `xy` goal. The target
z annotations are object-surface elevations (median target AGL is 0.65 m on
Val-Seen and -0.24 m on Val-Unseen), while the UAV controller is constrained
to fly at least 10 m AGL. Consequently, substituting 3-D distance silently
changes the published benchmark and the effective horizontal success radius.

The reproduction uses the released benchmark's 2-D metrics as
`benchmark_2d` for official-table comparison and checkpoint selection. It also
reports a stricter paper-literal `diagnostic_3d`, never used for selection. For
a predicted path \(p_0,\ldots,p_T\), goal \(g\), and 20 m threshold:

- NE: \(\|p_{T,xy}-g_{xy}\|_2\), averaged in meters;
- SR: fraction with final NE ≤20 m;
- OSR: fraction for which \(\min_t\|p_{t,xy}-g_{xy}\|_2\le20\) m;
- SPL: \(S\,l^*/\max(l,l^*)\), where \(l\) is executed 2-D path length and
  \(l^*=\|p_{0,xy}-g_{xy}\|_2\) is benchmark straight-line start-to-goal distance;
- progress: declared training target and diagnostic only, never a stop oracle;
- proposal coverage: fraction with a candidate within each threshold.

SBFNav implements and tests a self-contained evaluator in both explicit modes.
The official-result comparison always uses `benchmark_2d`; `diagnostic_3d` is
additional analysis only. Difficulty subsets use the supplied
`*_easy.json`, `*_medium.json`, and `*_hard.json`. They are verified to be a
disjoint, exhaustive partition by episode ID. Full-split evaluation executes
each trajectory once and aggregates all three rows from that membership map;
the final Test-Unseen process therefore produces overall and difficulty
metrics without a second access to the sealed split.

Every result record must include checkpoint path and SHA-256, split, original
or revised data version, seed, command, git commit, complete resolved config,
and package/GPU provenance.

## Official paper results

Table 1 (all numbers copied from arXiv v1; percentages are in percentage
points):

| Method | VS NE | VS SR | VS OSR | VS SPL | VU NE | VU SR | VU OSR | VU SPL | TU NE | TU SR | TU OSR | TU SPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 222.3 | 0.00 | 1.15 | 0.00 | 223.0 | 0.00 | 0.90 | 0.00 | 208.8 | 0.00 | 1.44 | 0.00 |
| Seq2Seq+GSM | 58.5 | 8.43 | 17.31 | 7.28 | 78.6 | 5.13 | 10.90 | 4.65 | 98.1 | 3.81 | 13.82 | 2.79 |
| CMA+GSM | 68.0 | 6.25 | 13.28 | 5.40 | 75.9 | 4.38 | 9.29 | 3.90 | 94.6 | 4.68 | 12.01 | 4.05 |
| MGP | 59.7 | 8.69 | 35.51 | 8.28 | 75.1 | 5.84 | 22.19 | 5.56 | 93.8 | 6.38 | 26.04 | 6.08 |
| AerialVLN+GSM | 56.6 | 10.16 | 22.20 | 7.89 | 72.7 | 6.35 | 15.24 | 5.06 | 85.1 | 6.72 | 18.21 | 5.16 |
| FlightGPT | 66.1 | 17.57 | 30.26 | 15.78 | 68.1 | 14.69 | 29.33 | 13.24 | 76.2 | 21.20 | 35.38 | 19.24 |
| GeoNav | 58.6 | 21.80 | 39.98 | 12.37 | 64.1 | 16.99 | 32.41 | 9.96 | 73.5 | 25.91 | 41.68 | 16.09 |
| HTNav | 47.2 | 28.30 | 47.92 | 25.07 | 59.8 | 15.85 | 34.43 | 13.55 | 68.5 | 22.23 | 41.02 | 19.63 |
| HETT | 45.2 | 25.16 | 48.40 | 23.01 | 62.1 | 17.48 | 25.09 | 14.46 | 72.9 | 22.97 | 39.30 | 17.01 |
| SBFNav | 33.1 | 42.95 | 53.64 | 40.63 | 56.2 | 19.57 | 34.59 | 18.52 | 43.6 | 32.29 | 46.01 | 30.43 |
| HETT (revised) | 37.2 | 31.09 | 51.86 | 25.76 | 51.3 | 19.10 | 34.78 | 16.70 | 40.4 | 28.90 | 49.56 | 23.79 |
| SBFNav (revised) | 32.5 | 43.24 | 53.77 | 40.93 | 49.2 | 20.24 | 35.41 | 19.15 | 38.5 | 32.46 | 49.82 | 30.59 |
| Human | 9.1 | 89.31 | 96.40 | 60.17 | 9.4 | 88.39 | 95.54 | 62.66 | 9.8 | 87.86 | 95.29 | 57.04 |

Official SBFNav Test-Unseen difficulty breakdown (original CityNav):

| Difficulty | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| Easy | 40.8 | 32.25 | 53.05 | 29.03 |
| Medium | 44.5 | 31.32 | 43.24 | 30.31 |
| Hard | 46.3 | 33.42 | 39.44 | 32.50 |

Representation ablations reported on validation are Point (VS/VU SR
29.80/16.20), Spatial Point (37.40/16.20), SBF without text (29.00/18.80), and
Language SBF (40.40/20.80). With K=10, the selector improves Val-Unseen SR from
18.76 to 20.36 and Test-Unseen SR from 31.26 to 33.19; oracle Top-K reaches
50.39 and 60.69 respectively.

## Training specification status

The main paper publishes neither epoch count, batch size, optimizer, learning
rate, scheduler, weight decay, warm-up, precision, augmentation, seed count,
checkpoint-selection rule, nor feature-caching policy. It only states frozen
SigLIP-B/16 towers, K=16 for the main result, the objective forms, 224/28 map
sizes, and a compact/lightweight architecture. The reproduction uses a fully
versioned YAML assumption set, bf16 on one RTX 5090, frozen-feature caching,
best-Val-Unseen checkpoint selection, and multiple seeds for final evidence.

## Required staged validation

1. **Coordinates/data:** verify split isolation and at least 20 train/VS/VU
   renderings; assert world→pixel→world errors within half a source pixel and
   world→field→world quantization within half a field-cell diagonal.
2. **Overfit:** deterministically overfit 32–128 train samples. Field mass,
   nearest-candidate recall, selector accuracy, and total/sub-loss curves must
   improve; otherwise full training is forbidden.
3. **Short run:** offline SigLIP load, forward/backward, bf16, finite gradients,
   checkpoint/save/resume, deterministic eval, logging, and GPU/memory checks.
4. **Main training:** only through `scripts/supervise_experiment.py`; evaluate
   VS and VU each epoch and save `last` plus best VU. Full training and each
   stable stage are committed and pushed.
5. **Frozen final test:** write a freeze manifest after all model/hyperparameter/
   checkpoint decisions. Run `test_unseen` exactly once from that manifest.

## Planned configurable ablations

The code exposes: no gated cross-attention, pooled instead of token text, no
geometry, no landmark text, no landmark candidates, K in {4,8,16,32}, no
ranking/margin/hard/altitude losses, SBF-only Top-1, and selector-only control.
Only smoke/subset checks are automatic; full-data ablations require an explicit
registered experiment to prevent unbounded GPU use.
