# Curated point and box comparisons

Point and Box use four annotated COCO examples with fixed foreground prompts.
The live studio accepts uploads in Everything mode. A single **Run both models**
executes the original lightweight decoder and prompt-adaptive refinement for
the initial prompt and two cumulative positive corrections. Four panels show
both initial outputs and both corrected outputs; the table reports every stage.
There is no background-point selector or free-form point/box upload.

## Original model policy

Both paths use the same frozen base in the released complete checkpoint.
TinySAM chooses the highest native predicted-IoU score among its three public
candidates. MobileSAM uses three public candidates for the initial point and
the native single-mask output for boxes and later point stages. The original
path calls the native decoder without the refinement modules. Each method
retains its own selected previous logits; both receive identical fixed prompts.
Ground-truth IoU never selects the native output candidate.

## True IoU and recorded assets

The gallery carries the legacy target mask: nearest resize to a longest side
of 1024, followed by bilinear restoration and the positive threshold used by
the interaction evaluation. Live IoU is recomputed from each prediction's
integer intersection and union with that mask. It is distinct from SAM's
predicted-IoU score. Original photographs are copied without pixel edits.

[examples.json](../site/examples.json) records the image, annotation, checkpoint
fingerprints, foreground coordinates, GT file, mask fingerprints and exact
intersection/union counts. The static explorer uses recorded CPU predictions;
the studio runs the selected checkpoint on the current device. Small numeric
differences between devices are possible.

| Backbone | Prompt | COCO image | Target |
|---|---|---:|---|
| TinySAM | Point | 110972 | Black bear |
| TinySAM | Box | 414795 | Elephant |
| MobileSAM | Point | 79651 | Bottle |
| MobileSAM | Box | 329080 | Bed |

## Selection scope

These are **curated successful examples**, including curated corrective-point
coordinates. A bounded pool of existing CC BY photographs was reviewed with
frozen models to find clear initial and corrected outputs. Ground truth was
used to place foreground clicks and measure example IoU. Selection changes no
weights, fitting, calibration, inference thresholds or aggregate evaluation.
Rejected examples and candidate trajectories are preserved in the local audit.

These common-prompt qualitative trajectories differ from the paper's aggregate
evaluation, where each complete method generates its own corrective trajectory
and may use either corrective-point sign. The demo's foreground-only interface
does not redefine that evaluation protocol. Dataset averages and all reported
stages remain in [results.md](results.md).

Everything uses the separate frozen ViT-H FSD pathway. Its animation explains
the flow; paired mask views and milliseconds come from actual inference.
