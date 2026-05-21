# GP Screens Analysis

Computer vision pipeline for detecting, classifying, and quantifying failure modes on failed gravel pack (GP) screens from high-resolution inspection images.

---

## Business Objective

Failed gravel pack screens are retrieved from wells and inspected at surface. This project automates visual failure analysis — replacing manual, subjective assessment with structured computer vision outputs: defect segmentation, erosion quantification, failure type classification, severity scoring, and annotated reporting.

---

## Failure Modes Detected

| Failure Mode | Description |
|---|---|
| Wire-wrap erosion holes | Localised perforation from abrasive sand flow |
| Screen collapse / crushing | Structural deformation from mechanical load |
| Corrosion pitting | Material loss from chemical attack |
| Mechanical damage | Impact or abrasion during running or retrieval |
| Plugging | Partial or complete pore blockage |
| Base-pipe exposure | Loss of screen jacket exposing the base pipe |

---

## Pipeline Architecture

```
Image/                      Raw inspection images (JPEG / PNG)
  └─ src/ingestion          Quality check, metadata extraction
  └─ src/preprocessing      Resize, normalise, screen region detection
  └─ src/detection          Defect localisation (bounding boxes)
  └─ src/segmentation       Binary mask generation per defect region
  └─ src/classification     Failure type classification + severity score
  └─ src/quantification     Erosion %, defect count, diameter distribution
  └─ src/annotation         Overlay generation, 3-panel composites
  └─ src/reporting          Per-image and campaign PDF reports
  └─ app/                   Streamlit dashboard
```

---

## Installation

```bash
git clone https://github.com/djimrastephane/GP_Screens_Analysis.git
cd GP_Screens_Analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### Streamlit Dashboard

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Batch Inference (CLI)

```bash
python scripts/batch_inference.py --input Image/ --output outputs/
```

---

## Dashboard Pages

| Page | Audience | Content |
|---|---|---|
| Home | All | Campaign KPIs, erosion trends, severity distribution |
| Gallery | All | Thumbnail grid of annotated screen images |
| Analysis | Engineering | Per-image defect breakdown, 3-panel composite, failure type chart |
| Quantification | Engineering | Erosion %, defect density, scatter plots, full results table |
| Reports | All | Download per-image or campaign-level PDF reports |

Role-based views are available — Engineering users see full technical detail; management views show summary KPIs and trends.

---

## Expected Inputs

- JPEG or PNG inspection images of failed GP screens
- Optional: CSV or Excel with well and completion context (well name, depth, completion type, sand production history)

Place source images in `Image/` or `data/raw/`. Do not modify source files — the pipeline preserves originals unchanged.

---

## Outputs

| Output | Location | Description |
|---|---|---|
| Binary masks | `outputs/masks/` | Defect regions per image |
| Annotated overlays | `outputs/overlays/` | Source image with overlaid detections |
| 3-panel composites | `outputs/panels/` | Original / annotated / mask side-by-side |
| Results CSV | `data/processed/` | Tabular failure classification and metrics |
| PDF reports | `outputs/reports/` | Per-image annotated engineering report |

---

## Key Metrics

- **Erosion %** — defect area as a fraction of detected screen region
- **Defect count** — number of distinct failure locations per image
- **Severity score** — low / medium / high / critical, confidence-scored
- **Failure type distribution** — breakdown by failure mode across campaign
- **Screens requiring review** — % flagged for human confirmation

---

## Design Principles

- Source images are never modified
- All model outputs are estimates; critical findings are flagged for human review
- Confidence scores are reported on all classification outputs
- Poor image quality is flagged before inference runs
- Outputs are traceable back to the source image at every stage

---

## Assumptions & Limitations

- Images are taken at surface after screen retrieval; downhole images are not supported
- Accuracy degrades on severely occluded, blurry, or very low-resolution images
- Erosion percentages are relative to the detected screen region, not absolute screen area
- Scale references are not always present; absolute measurements in mm require image calibration
- Model performance on unseen failure modes may be lower than on training distribution

---

## Target Users

- **Engineering / Completions** — detailed failure classification, root cause evidence, export-ready reports
- **Well Integrity / HSE** — failure documentation for regulatory records and re-completion design
- **Management** — asset-level failure rate trends, severity distribution, sand control risk ranking
