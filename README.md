# GP Screens Analysis

Computer vision pipeline for detecting, classifying, and quantifying failure modes on failed gravel pack (GP) screens from high-resolution inspection images.

---

## Business Objective

Failed gravel pack screens are retrieved from wells and inspected at surface. This project automates visual failure analysis — replacing manual, subjective assessment with structured computer vision outputs: defect segmentation, erosion quantification, failure type classification, severity scoring, and annotated reporting.

---

## Installation

```bash
cd GP_Screens_Analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### Streamlit Dashboard

```bash
streamlit run app/main.py
```

### Batch Inference (CLI)

```bash
python scripts/batch_inference.py --input Image/ --output outputs/
```

---

## Expected Inputs

- JPEG or PNG inspection images of failed GP screens
- Optional: CSV or Excel with well and completion context

Place source images in `Image/` or `data/raw/`. Do not modify source files.

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Binary masks | `outputs/masks/` | Defect regions per image |
| Annotated overlays | `outputs/overlays/` | Source image with overlaid detections |
| Results CSV | `data/processed/` | Tabular failure classification and metrics |
| PDF reports | `outputs/reports/` | Per-image annotated engineering report |

---

## Assumptions

- Source images are taken at surface after screen retrieval.
- Image quality varies; quality flags are applied before inference.
- Model outputs are estimates requiring human review confirmation for critical findings.
- Erosion percentages are relative to the detected screen region area.

---

## Known Limitations

- Accuracy degrades on severely occluded, blurry, or very low-resolution images.
- Scale references are not always present; absolute measurements in mm require calibration.
- Model was developed on a limited dataset; performance on unseen failure modes may be lower.
- Lighting variability from field photography affects detection confidence.
