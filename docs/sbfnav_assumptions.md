# SBFNav reproduction assumptions

This file separates choices made by this paper-based reimplementation from
claims actually present in arXiv:2609.05841v1. Values are provisional until
`configs/sbfnav/main.yaml` is frozen. Changing a value after validation creates
a new experiment; no Test-Unseen observation may influence it.

| Area | Reproduction assumption | Why it is needed |
|---|---|---|
| Square canvas | Symmetric metric padding; north-up; cell centers define coordinates | Padding placement and sampling convention are absent |
| Source bounds | Use each TIF affine bounds, not hard-coded rounded `MAP_BOUNDS` | TIF is the georeferencing authority; legacy bounds differ by about 0.05 m |
| State fusion | Latest observation overwrites accumulated RGB/height in overlap; explored mask is union | Fusion rule is absent |
| Observation footprint | North-up map evidence is revealed inside a yaw-rotated square whose half-width equals altitude above ground (a 90° footprint proxy) | Camera intrinsics/FOV and raster observation simulator are absent |
| Height | Normalize above-ground world height by configured maximum and clamp to [0,1] | “normalized height” has no formula |
| Landmark contour | 1-pixel antialiased-equivalent binary outline after 224 resampling; referenced channel is filled polygons | Thickness/fill semantics are incomplete beyond “contours” and “mask” |
| Trajectory | Draw metric polyline with configured 2 m radius, including current position | Raster thickness is absent |
| Field encoder | Four residual stages with strides 2,2,2 and a final 28×28 grid; GroupNorm and GELU | No layer topology is published |
| Feature dimension | 256; cross-attention 8 heads; dropout 0.1 | Width/head/dropout are absent |
| Gate | Scalar `gamma=0`, hence initial `tanh(gamma)=0` | Paper only says “near zero” |
| Text | SigLIP max length 64; use last hidden token states and attention mask | Token length/truncation are absent |
| Decoder | Residual 3×3 block plus 1×1 field-logit convolution at 28×28 | Decoder topology is absent |
| Gaussian | Metric sigma 20 m, truncated only by the valid map mask | Sigma is in missing Supplement B.4 |
| NMS | Greedy 3×3 suppression: select the highest valid logit, suppress its Chebyshev-radius-1 neighbourhood, and repeat to K; equal logits use deterministic row-major order | NMS radius/ties/algorithm are absent; greedy suppression preserves K spatially distinct proposals even for a smooth unimodal field |
| Deduplication | Merge candidates within 2 m, preserving all source tags; field candidate wins coordinate tie | Deduplication is not described |
| Selector crop | 100 m square world crop rendered to 224×224; unobserved RGB is zero and explicit validity masking is applied before vision encoding | Crop extent and masking implementation are absent |
| Vision feature | Frozen SigLIP pooled image feature as one visual token | Paper says “feature” but not token/pooled output |
| Geometry | World-axis dx/dy/d divided by square map side; bearing is `atan2(dy,dx)`; preserve sin/cos | Normalization/frame are absent |
| Landmark tokens | One token per referenced landmark, summing projected name and candidate-to-landmark geometry; cap 8 with a mask | Aggregation and cap are absent |
| Selector | 256-d, 2-layer, 8-head pre-norm Transformer, FFN 1024, dropout 0.1 | Only “lightweight Transformer” is stated |
| Candidate ordering | Deterministically shuffle during training, stable source/coordinate order at eval | Prevents order from proxying forbidden SBF rank |
| Altitude target | Absolute target z; predict normalized offset from map ground level; clamp to ground+10…ground+100 m | Absolute/offset convention and range are absent |
| Progress | `clip(1 - current_3d_distance / initial_3d_distance, 0, 1)` with MSE | Definition and regression loss are absent |
| Ranking | `tau=20 m`, `R_good=20 m`, `R_bad=40 m`, margin `m=1` | Missing Supplement B.4 |
| Loss weights | field 1.0; rank 1.0; margin within rank 0.2; hard within rank 0.5; altitude 0.1; progress 0.1 | Missing Supplement B.4 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-2, 5% linear warm-up then cosine decay | Training details are absent |
| Training | 20 epochs, seed 0 main development run, batch 16 effective using accumulation as needed, bf16, gradient clip 1.0 | Epoch/batch/precision/seed are absent |
| Frozen feature cache | Precompute all unique instruction/landmark token features in batches of 256; use tensor-native vision normalization, vision batches of 128, and a bounded 2,000,000-entry CPU-fp16 crop LRU keyed by deterministic training state/candidate coordinate | Cache/precomputation policy is absent; frozen features are mathematically reusable |
| State sampling | Sample up to four teacher-trajectory states per episode per epoch; history contains only prefix observations | Temporal sampling is absent |
| Controller | Proportional direction to waypoint, at most 50 m horizontal and 10 m vertical per macro replan; yaw faces motion; horizon 20; no learned/progress/distance stop. This matches the released CityNav/HETT reachability budget (10×5 m inner moves for each of 20 decisions). | Missing Supplement B.6 |
| Stop | End only at fixed horizon or legal-map/controller stagnation; never use goal distance | Paper forbids progress for stopping but gives no alternative |
| Metric protocol | Use released CityNav/HETT `Pose.xy` NE/SR/OSR/SPL for official comparison; additionally report a 3-D Euclidean diagnostic | SBFNav says Euclidean distance but does not state dimensions; the executable benchmark protocol is explicitly 2-D |
| Checkpoint | Highest Val-Unseen benchmark-2D SR, tie-break benchmark-2D SPL then lower NE, without Test-Unseen access | Selection rule is absent |
| Per-epoch validation | Deterministic 128-episode split-wide sample from each validation split; final reporting reruns the selected checkpoint on every episode | Full closed-loop validation every epoch is prohibitively expensive and no protocol is published; 128 reduces SR quantization to 0.78125 points after a 32-episode audit proved too noisy |

## Known fidelity constraints

- The arXiv v1 source omits the supplements it cites, so the implementation
  cannot be a hyperparameter-exact reproduction.
- Local revised/original split counts match the supplied files, but the local
  RGBD/CityRefer map inventory does not match the paper's stated 46-block
  SensatUrban inventory.
- CityNav provides static orthographic RGB and height rasters. Online RGB-D is
  simulated from these rasters and poses; it is not a photorealistic camera
  renderer.
- Frozen SigLIP features make training tractable on one RTX 5090, but feature
  cache format and preprocessing must be recorded because neither is specified.

These choices may be revised only from train/Val-Seen/Val-Unseen evidence. Once
the freeze manifest is written, any revision invalidates it and Test-Unseen must
remain unopened until a new freeze is established.
