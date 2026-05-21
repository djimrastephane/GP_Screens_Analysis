# Data Contract

## General Rules

- All timestamps must use ISO-8601 format.
- Units must always be specified.
- Missing values must be explicit.
- Column names should use snake_case.
- No duplicate column names allowed.

---

# Screen Inspection Image Metadata

| Field | Type | Description |
|-------|------|-------------|
| image_id | text | Unique image identifier (UUID or slug) |
| source_filename | text | Original filename preserved unchanged |
| capture_date | ISO-8601 | Inspection or retrieval date |
| well_name | text | Well identifier |
| completion_zone | text | Completion interval or zone label |
| screen_type | text | Screen type (e.g. wire-wrap, premium, base-pipe) |
| screen_position | text | Position descriptor (e.g. top, middle, bottom, joint number) |
| image_resolution | text | Resolution in pixels (width x height) |
| image_quality_flag | text | ok / blurry / glare / low_resolution / occluded / bad_angle |
| scale_reference_present | boolean | Whether a physical scale bar or reference is visible |

---

# Failure Classification

| Field | Type | Description |
|-------|------|-------------|
| failure_id | text | Unique failure record identifier |
| image_id | text | Foreign key to image metadata |
| failure_type | text | Failure category (see controlled vocabulary below) |
| failure_location | text | Location descriptor on the screen |
| severity | text | low / medium / high / critical |
| erosion_percentage | float | Estimated erosion as percentage of screen area |
| hole_enlargement_mm | float | Estimated hole or breach size in mm |
| confidence_score | float | Model confidence 0.0–1.0 |
| requires_human_review | boolean | Flag for uncertain or borderline cases |
| analyst_notes | text | Free-text notes from reviewer |

### Controlled Vocabulary — failure_type

- erosion_hole
- wire_wrap_failure
- screen_collapse
- corrosion_pitting
- mechanical_damage
- plugging_partial
- plugging_complete
- unknown

---

# Segmentation and Mask Outputs

| Field | Type | Description |
|-------|------|-------------|
| mask_id | text | Unique mask identifier |
| image_id | text | Foreign key to source image |
| model_version | text | Model name and version |
| mask_path | text | Path to saved mask file |
| overlay_path | text | Path to annotated overlay image |
| defect_area_px | integer | Defect area in pixels |
| defect_area_pct | float | Defect area as percentage of screen region |
| bounding_box_xyxy | text | Bounding box as x1,y1,x2,y2 |
| run_timestamp | ISO-8601 | Inference timestamp |

---

# Well and Completion Context

| Field | Type | Description |
|-------|------|-------------|
| well_name | text | Well identifier |
| field_name | text | Field or asset name |
| completion_date | ISO-8601 | Original completion date |
| retrieval_date | ISO-8601 | Screen retrieval date |
| cumulative_sand_bbls | float | Cumulative sand production in bbls if available |
| producing_interval_ft | float | Gross interval length in feet |
| screen_count | integer | Number of screens in completion |
| gravel_pack_type | text | e.g. openhole GP, cased hole GP, frac-pack |
| failure_year | integer | Year failure was identified |

---

# File Validation Rules

## Allowed File Types

- JPG / JPEG
- PNG
- TIFF
- BMP
- PDF (inspection reports)
- CSV
- XLSX

## Rejected Files

- Executables
- Unsupported archives
- Corrupted or zero-byte images

---

# Missing Data Rules

- Null numeric values must use NaN.
- Missing categorical values must use NULL or explicit "unknown".
- Missing images must be flagged, not silently skipped.
- Missing scale references must be noted in image_quality_flag.

---

# Naming Conventions

## Preferred Naming

- image_id
- failure_type
- erosion_percentage
- confidence_score
- well_name

## Avoid

- ImageID
- FailureType
- ErosionPct
- WellName

---

# Data Quality Checks

Validate:

- duplicate image_ids
- missing well or completion context
- images with no detectable screen region
- mask files that do not correspond to a source image
- confidence scores outside 0.0–1.0
- erosion_percentage outside 0–100
- image_quality_flag not in controlled vocabulary

---

# Engineering Assumptions

- Units must remain consistent across all datasets.
- Converted units must be logged.
- Source images remain the authoritative reference and must not be mutated.
- Processed images, masks, and overlays are always stored separately from source files.
- Model outputs are estimates; final failure classification requires human review confirmation.
