# Ground-truth annotations

Used by `scripts/evaluate_detection.py` to score the detector against
human-labeled boxes (precision, recall, F1, mean IoU).

## Format

One JSON file per labeled image, any filename, containing:

```json
{
  "source_filename": "Picture 1.jpg",
  "boxes": [
    {"x1": 120, "y1": 80, "x2": 340, "y2": 290, "failure_type": "corrosion_pitting"},
    {"x1": 400, "y1": 210, "x2": 460, "y2": 270, "failure_type": "erosion_hole"}
  ]
}
```

- `source_filename` must match the `source_filename` stored on a detection
  run in `data/processed/images.db` (i.e. the image must already have been
  run through `scripts/batch_detect.py`).
- `x1, y1, x2, y2` are pixel coordinates in the same frame the detector
  produced its boxes in.
- `failure_type` should use the controlled vocabulary in `DATA_CONTRACT.md`.
  It's recorded for future classification-accuracy evaluation but isn't
  used by the current box-localisation metrics — the detector only
  localises defects, it doesn't assign failure type.

No labeled images exist yet. Add files here as analysts review and label
real screens.
