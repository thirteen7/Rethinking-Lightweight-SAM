# Evaluation results

All values below are ordinary legacy mIoU (%), three interaction rounds,
mask-centre first point, annotation box prompts and chunk64. No additional
quality/scoring head is present. Fitting, calibration and checkpoint selection
use SA-1B training data exclusively. The public repack preserves every tensor
from the accepted checkpoints.

| Dataset / coverage | Model | Point 1 | Point 2 | Point 3 | Box 1 | Box 2 | Box 3 |
|---|---|---:|---:|---:|---:|---:|---:|
| SA-1B: 11,186 images / 618,098 cap64 targets | TinySAM | 70.137 | 76.268 | 78.729 | 84.156 | 84.838 | 85.183 |
| SA-1B: same cap64 selection | MobileSAM | 68.205 | 74.840 | 79.202 | 83.503 | 84.079 | 84.317 |
| COCO: 4,952 annotated images / 36,335 targets | TinySAM | 55.651 | 64.430 | 69.438 | 76.515 | 77.679 | 78.364 |
| COCO: same full coverage | MobileSAM | 54.998 | 64.446 | 69.502 | 76.027 | 77.028 | 77.228 |
| LVIS: 19,626 valid images / 244,707 targets | TinySAM | 56.263 | 62.702 | 66.323 | 75.474 | 76.569 | 77.133 |
| LVIS: same full coverage | MobileSAM | 54.431 | 60.915 | 65.862 | 74.448 | 75.362 | 75.681 |

## TinySAM LVIS acceptance addendum

The original full inference exited successfully. Its original CPU acceptance
failed on one stored second-round point IoU: image 195271, annotation 94187,
value 1.2310324907302856. The original failed ledger/cache were preserved.
One bounded replay used the same frozen model and previous-image RNG state.
The other 134 point IoUs, all 135 candidate choices and final point RNG matched
exactly. GPU/CPU intersection 2158 and union 2323 agreed; the valid stored value
is 0.9289711713790894. An additive, single-value correction passed full CPU
coverage/range/order/reaggregation acceptance. The exact trigger of the original
out-of-range value was not reproduced or identified. No clamping, target
deletion or full GPU rerun was performed.

## TinySAM COCO1000 structure ablation

1,000 fixed validation images / 7,179 targets. The table uses server controls
for server module effects. Local/server numerical differences are preserved;
all gains are matched to a baseline on the same execution host. These results
do not isolate the causal effect of the encoding change alone.

| Prompt / configuration | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| Point / native, server control | 46.466 | 63.487 | 68.789 |
| Point / first-click selector | 55.798 | 61.478 | 68.082 |
| Point / first + feedback | 55.798 | 62.631 | 67.677 |
| Point / first + residual | 55.798 | 63.027 | 69.087 |
| Point / full | 55.798 | 64.473 | 69.556 |
| Box / native | 75.095 | 76.425 | 77.054 |
| Box / current selector | 75.221 | 76.497 | 77.275 |
| Box / independent decoder | 76.411 | 77.880 | 78.545 |
| Box / full | 76.601 | 77.915 | 78.549 |

The isolated feedback path has a negative third-round effect. In the full box
trajectory, the final gain over the independent decoder is only +0.003 pp.
On the full model's *same candidate pool*, the final region-selected result is
0.027 pp below independent-native selection; these are different comparisons.

ViT-H is included as the previously accepted complete model. Its earlier
SA-1B full-target evaluation has different coverage from the cap64 table and
is not used as a matched cap64 gain control here.
