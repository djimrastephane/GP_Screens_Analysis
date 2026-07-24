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

Five pages, organised around what an engineer actually needs to do rather than
around pipeline stages. Role-based views are available on every page —
Engineering users see full technical detail; management views show summary
KPIs and trends. The Management/Engineering choice persists as you navigate
between pages.

### Overview — Campaign Dashboard, Assessment & Full Data
![Overview](docs/screenshots/01_overview.png)
Three tabs on one page instead of three separate ones. **Dashboard**: campaign KPIs (images analysed, mean/max erosion %, total defects, review flags — with a direct link into the Review Queue when any are flagged), erosion % bar chart ranked by severity, failure type distribution, and severity breakdown. **Assessment**: colour-coded campaign risk banner, observed conditions generated from actual data, morphological classification basis per failure type, potential root causes, and prioritised recommended actions. **Full Data & Export**: erosion-vs-defect-count scatter, the complete per-image metrics table, and CSV export for both the metrics table and the full defects table.

### Review Queue — Triage Worklist
![Review Queue](docs/screenshots/02_review_queue.png)
Every detection flagged for human review, sorted least-confident-first, with the model's own reasoning string per row. Mark items reviewed inline (persisted to the database, not just session state) and the queue shrinks — toggle "show already-reviewed" to bring them back. Filter by severity or failure type, and jump straight to the full per-image view for any row.

### Gallery — Browse, Tag Wells & Compare Screens
![Gallery](docs/screenshots/03_gallery.png)
Thumbnail grid of annotated screen images, sortable and filterable by severity or dominant failure type. Tag images with a well name and completion zone, then group the grid by well. Select two or more screens for comparison:

![Gallery comparison](docs/screenshots/03b_gallery_compare.png)
Side-by-side metrics table and annotated thumbnails for just the selected screens, plus an erosion % chart scoped to the comparison.

### Analysis — Per-Image Detail
![Analysis](docs/screenshots/04_analysis.png)
Six KPI cards: erosion % (formula tooltip), defect count, severity (threshold tooltip), dominant failure type, mean model confidence, and review flag count. Engineering view adds: image quality panel (focus score, illumination, quality flag, screen coverage); auto-generated engineering assessment with risk level, likely mechanism, plain-English interpretation, and morphological classification basis explaining *why* the failure type was assigned; scale calibration UI to enter pixels/mm from a visible ruler — instantly converts all diameters to mm and areas to cm²; per-defect table with the model's own reasoning string per detection and an inline reviewed checkbox that stays in sync with the Review Queue. Reachable directly from a Review Queue row, landing on the right image.

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
| Largest diameter | Equivalent circular diameter of the largest defect (pixels, or mm when scale is calibrated from a ruler in the image). |
| Damage density | Defects per cm² of screen area (requires scale calibration). |
| Total damaged area | Cumulative defect area in cm² (requires scale calibration). |
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
