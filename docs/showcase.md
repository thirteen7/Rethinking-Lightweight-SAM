# Qualitative comparison protocol

The Point and Box explorer compares the **original lightweight model** with
its prompt-adaptive counterpart. Both use the same frozen TinySAM or MobileSAM
base weights contained in the released complete checkpoint. They receive the
same image and point/box prompt. The original method calls the native decoder
directly, without the refinement modules.

## Original model output

- **TinySAM:** its official decoder forward returns three public candidates.
  The candidate with the highest native predicted-IoU score is selected for
  both point and box prompts.
- **MobileSAM:** the first point uses three public candidates, selected by the
  native predicted-IoU score. Boxes and later point rounds use the native
  single-mask output.

Candidate selection does not use ground-truth IoU. Native predicted IoU is a
model score; the percentages displayed above the photographs are measured
against the illustrated COCO annotation.

## Shared prompts and independent feedback

The first point is the annotation centroid, and the initial box is the
annotation box, following the fixed evaluation interface. Subsequent rounds
use a shared corrective click from the frozen error-point sampler applied to
the refined prediction. Each method receives its **own previous selected
logits** as mask feedback. These controlled same-prompt trajectories illustrate
the two methods; they are separate from full dataset evaluation tables.

## Displayed scores

IoU uses the existing legacy ground truth: nearest resize to a longest side of
1024, followed by bilinear restoration to the original image dimensions and
the fixed positive threshold. Integer intersection and union counts, exact
mask fingerprints, image and annotation identities, and model fingerprints
are recorded in [examples.json](../site/examples.json).

The gallery is curated for legible first-round improvements. This selection
does not fit a model, tune thresholds, choose weights, or estimate average
performance. The four highlighted targets are:

| Backbone | Prompt | COCO image | Target |
|---|---|---:|---|
| TinySAM | First point | 217400 | Train |
| TinySAM | First box | 414795 | Elephant |
| MobileSAM | First point | 79651 | Bottle |
| MobileSAM | First box | 329080 | Bed |

Everything mode uses a separate prompt-free FSD pathway and does not use the
point/box refinement modules. Its interactive flow diagram explains the
computation; the Instance masks view displays actual predictions.
