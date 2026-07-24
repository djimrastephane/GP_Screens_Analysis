# Ground-truth annotations

Used by `scripts/evaluate_detection.py` to score the detector against
human-labeled boxes (precision, recall, F1, mean IoU), and by
`scripts/evaluate_segmentation.py` to score the segmentation stage's
composite mask against human-labeled polygon masks (pooled precision,
recall, Dice, mean IoU).

Label images interactively with `scripts/label_image.py` (see its docstring
for usage) rather than hand-editing these files — it resolves the correct
preprocessed frame, prompts from the controlled failure_type vocabulary, and
keeps `--status`/`--reset`/`--preview`/`--append` all working consistently
regardless of what a file happens to be named.

## Format

One JSON file per labeled image, any filename, containing:

```json
{
  "source_filename": "Picture 1.jpg",
  "boxes": [
    {"x1": 120, "y1": 80, "x2": 340, "y2": 290, "failure_type": "corrosion_pitting"},
    {"x1": 400, "y1": 210, "x2": 460, "y2": 270, "failure_type": "erosion_hole"}
  ],
  "masks": [
    {"points": [[100, 60], [360, 60], [360, 310], [100, 310]], "failure_type": "corrosion_pitting"}
  ]
}
```

- `source_filename` must match the `source_filename` stored in
  `data/processed/images.db` (i.e. the image must already have been run
  through `scripts/batch_preprocess.py`, and through `scripts/batch_detect.py`
  / `scripts/batch_segment.py` respectively for the box / mask evaluators to
  find a matching run to score against).
- `"boxes"` and `"masks"` are both optional and independent — a file may
  have either, both, or neither. `--reset` (see `label_image.py`) clears one
  key at a time; the file is only deleted once both are empty or absent.
- `x1, y1, x2, y2` (boxes) and each polygon's `points` (masks) are pixel
  coordinates in the same preprocessed frame the detector/segmenter actually
  operate on (the 640x640 letterboxed PNG in
  `data/processed/preprocessed/png/`, not the original raw photo).
- A `"masks"` polygon needs at least 3 points. Multiple polygons for the
  same image are unioned into one ground-truth composite mask at evaluation
  time — this mirrors how the predicted composite mask itself is a union of
  all per-detection masks (see `src/segmentation/pipeline.py`). Masks are
  scored against the composite only; there's no per-instance predicted mask
  on disk to match against (`SegmentationMask.to_dict()` stores summary
  stats, not pixel arrays).
- `failure_type` should use the controlled vocabulary in `DATA_CONTRACT.md`.
  For boxes it's recorded for future classification-accuracy evaluation but
  isn't used by the current box-localisation metrics (class-agnostic). For
  masks it isn't used by the current composite-mask metrics either (also
  class-agnostic) — recorded for the same future purpose.

Add files here as analysts review and label real screens, or generate them
with `scripts/label_image.py` (append `--masks` to draw polygons instead of
boxes).
