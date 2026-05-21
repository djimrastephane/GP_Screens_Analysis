# Project Context

## Industry

Oil and Gas — Sand Control and Well Completion

## Project Title

GP Screens Analysis — Gravel Pack Screen Failure Investigation using Computer Vision

## Central Method

Computer vision is the primary analytical method. All workflows are image-driven.

---

# Main Focus Areas

- Gravel pack (GP) screen failure detection and classification
- Erosion quantification from inspection images
- Defect segmentation and localisation
- Failure severity scoring
- Multi-image comparison across screens from the same well or field
- Root cause hypothesis generation from visual evidence
- Sand control performance benchmarking

---

# Failure Modes of Interest

- Wire-wrap erosion holes
- Screen collapse or crushing
- Corrosion pitting
- Mechanical damage from running or retrieval
- Plugging (partial or complete)
- Base-pipe exposure
- Unknown or ambiguous damage

---

# Typical Input Data

## Primary Inputs

- High-resolution JPEG / PNG photographs of failed screens
- Inspection photographs taken at surface after screen retrieval
- Annotated or unannotated images from field teams

## Supporting Context

- Well completion summaries
- Sand production history
- Screen specifications (type, gauge, OD, manufacturer)
- Producing interval depth and length
- Gravel pack design parameters

## Documents

- Post-job completion reports
- Sand control review reports
- Lessons learned reports

---

# Target Users

## Higher Management

Needs:

- Summary of failure rates across asset
- Cost of failure overview
- Sand control risk ranking
- Trend by field, year, or completion type

## Engineering Teams

Needs:

- Detailed failure classification per screen
- Erosion maps and segmentation overlays
- Confidence-scored defect reports
- Comparison between screens from same well
- Evidence-based root cause hypotheses
- Export-ready engineering reports

## Well Integrity and Completions Teams

Needs:

- Failure documentation for regulatory or HSE records
- Input to re-completion or workaround design
- Historical failure pattern analysis

---

# Key KPIs

## Screen Failure

- Erosion percentage per screen
- Number of failure locations per image
- Failure type distribution
- Severity score distribution
- Screens requiring human review (%)

## Asset-Level

- Failure rate by field, well, or year
- Mean erosion percentage by completion type
- Time to failure from completion date

---

# Design Philosophy

Applications should:

- Preserve every source image unchanged
- Maintain full traceability from model output back to source image
- Distinguish model estimates from confirmed findings
- Flag poor image quality before inference
- Provide confidence scores on all classification outputs
- Be usable by engineers who are not machine learning specialists
- Support export of annotated images, masks, and tabular results

---

# Deployment Preference

Initial deployment:

- Local-first
- Streamlit dashboard for image upload and review
- SQLite backend for image metadata and results

Future deployment:

- Docker containers
- Optional API layer for batch inference
- Cloud deployment if required

---

# Long-Term Vision

Build a structured failure database from historical screen images to:

- Train supervised failure classification models
- Benchmark sand control performance across fields and operators
- Provide decision support for future completion design
- Generate automated failure reports for engineering and regulatory purposes
