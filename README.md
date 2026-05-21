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

Role-based views are available — Engineering users see full technical detail; management views show summary KPIs and trends.

### Home — Campaign Overview
![Home](docs/screenshots/01_home.png)
Campaign KPIs (images analysed, mean and max erosion %, total defects, review flags), erosion % bar chart ranked by severity, failure type distribution, and severity breakdown. Erosion % tooltip shows the exact formula.

### Gallery — Image Browser
![Gallery](docs/screenshots/02_gallery.png)
Thumbnail grid of annotated screen images, sortable by erosion % or severity, with per-card defect summary and review flags.

### Analysis — Per-Image Detail
![Analysis](docs/screenshots/03_analysis.png)
Six KPI cards per image: erosion % (with formula tooltip), defect count, severity (with threshold basis tooltip), dominant failure type, mean model confidence, and review flag count. Engineering view adds an auto-generated assessment — risk level, likely mechanism, plain-English interpretation, and recommended actions. Defect size summary (largest defect, average defect, largest diameter) sits above the per-defect table. Severity threshold basis and recommended actions are in expandable panels.

### Quantification — Metrics & Charts
![Quantification](docs/screenshots/04_quantification.png)
Erosion % bar chart with labelled severity thresholds and full metric definition in the caption. Charts include failure type distribution and erosion vs defect count scatter. Full metrics table adds largest defect %, average defect diameter, and mean confidence columns alongside defect count.

### Reports — PDF Downloads
![Reports](docs/screenshots/05_reports.png)
Download the campaign summary PDF or individual per-image annotated engineering reports as a ZIP archive.

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

| Metric | Definition |
|---|---|
| Erosion % | Total defect pixel area ÷ visible screen pixel area × 100. Model estimate of damaged area fraction — not a direct measurement of metal loss or open-flow area increase. |
| Severity | < 5 % → Low · 5–20 % → Medium · 20–50 % → High · ≥ 50 % → Critical. Screen collapse and complete plugging escalate one level regardless of area. |
| Mean confidence | Average model confidence across all detections for the image (0–100 %). Detections below 70 % are flagged for human review. |
| Largest defect % | Area of the single largest detected defect as % of visible screen area. More indicative of breach severity than defect count alone. |
| Avg defect diameter | Mean equivalent circular diameter across all detections (pixels). Converts to mm when `pixels_per_mm` is calibrated. |
| Defect count | Number of distinct failure locations detected per image. |
| Failure type distribution | Breakdown by failure mode across the full campaign. |

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
