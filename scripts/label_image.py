"""CLI: interactively draw ground-truth boxes on a preprocessed image.

Draws boxes with the mouse (via cv2.selectROIs) directly on the same
640x640 preprocessed frame the detector operates on, prompts for a
failure_type per box, and writes the result to data/annotations/ in the
format scripts/evaluate_detection.py expects.

Usage:
    python scripts/label_image.py "Picture 1.jpg"
    python scripts/label_image.py "Picture 1.jpg" --append
    python scripts/label_image.py "Picture 1.jpg" --preview
    python scripts/label_image.py --status
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.annotation.colours import get_colour, rgb_to_bgr
from src.annotation.draw import draw_box
from src.classification.base import FailureType

_VOCAB = [ft.value for ft in FailureType]


def _slug(filename: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", filename.lower()).strip("_")


def _find_annotation_file(annotations_dir: Path, source_filename: str) -> Path | None:
    """Find an existing annotation file for source_filename, regardless of its own
    filename — the format allows 'any filename' (see data/annotations/README.md),
    so identity is determined by the source_filename field inside, not the path."""
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("source_filename") == source_filename:
            return path
    return None


def _render_preview(img, boxes: list[dict]):
    """Return a copy of img (BGR) with ground-truth boxes + failure_type labels drawn."""
    out = img.copy()
    for b in boxes:
        colour_rgb = get_colour(b["failure_type"])
        draw_box(out, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]), colour_rgb, thickness=2)

        label = b["failure_type"].replace("_", " ")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx1, ly1 = int(b["x1"]), max(0, int(b["y1"]) - th - 8)
        cv2.rectangle(out, (lx1, ly1), (lx1 + tw + 6, ly1 + th + 6), rgb_to_bgr(colour_rgb), -1)
        cv2.putText(out, label, (lx1 + 3, ly1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _prompt_failure_type() -> str:
    print("  Failure type:")
    for i, ft in enumerate(_VOCAB, 1):
        print(f"    {i}. {ft}")
    while True:
        raw = input(f"  Enter number or name [1-{len(_VOCAB)}]: ").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(_VOCAB):
            return _VOCAB[int(raw) - 1]
        if raw in _VOCAB:
            return raw
        print(f"  Not recognised. Choose a number 1-{len(_VOCAB)} or one of: {', '.join(_VOCAB)}")


def _resolve_preprocessed_path(db_path: Path, source_filename: str) -> Path | None:
    from src.ingestion.database import init_db
    from sqlalchemy import text

    SessionFactory = init_db(db_path)
    with SessionFactory() as session:
        row = session.execute(
            text("SELECT preprocessed_png_path FROM preprocessed_images WHERE source_filename = :fn"),
            {"fn": source_filename},
        ).fetchone()
    return Path(row[0]) if row else None


def _print_status(db_path: Path, annotations_dir: Path) -> None:
    from src.ingestion.database import init_db
    from sqlalchemy import text

    SessionFactory = init_db(db_path)
    with SessionFactory() as session:
        rows = session.execute(text("SELECT source_filename FROM preprocessed_images")).fetchall()

    labeled: dict[str, int] = {}
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            labeled[data["source_filename"]] = len(data.get("boxes", []))
        except (json.JSONDecodeError, KeyError):
            continue

    header = f"{'Filename':<25} {'Status':<12} {'Boxes'}"
    print(header)
    print("-" * len(header))
    for (fn,) in rows:
        n = labeled.get(fn)
        status = "labeled" if n is not None else "unlabeled"
        print(f"{fn:<25} {status:<12} {n if n is not None else '—'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively label ground-truth boxes for one image.")
    parser.add_argument("image", nargs="?", help="source_filename as stored in images.db, e.g. 'Picture 1.jpg'")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--append", action="store_true", help="Add to existing boxes instead of replacing them")
    parser.add_argument("--status", action="store_true", help="List which images are labeled vs. not, then exit")
    parser.add_argument("--reset", action="store_true",
                        help="Delete this image's existing annotation file and exit (no drawing)")
    parser.add_argument("--preview", action="store_true",
                        help="Show this image's existing boxes drawn on it, save a PNG, and exit (no drawing)")
    parser.add_argument("--preview-dir", default="data/annotations/previews")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / args.db
    annotations_dir = root / args.annotations_dir
    annotations_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        _print_status(db_path, annotations_dir)
        return

    if not args.image:
        parser.error("image is required unless --status is given")

    if args.preview:
        ann_path = _find_annotation_file(annotations_dir, args.image)
        if ann_path is None:
            print(f"'{args.image}' has no annotation file to preview.")
            return
        png_path = _resolve_preprocessed_path(db_path, args.image)
        if png_path is None or not png_path.exists():
            print(f"No preprocessed image found for '{args.image}'.")
            return
        img = cv2.imread(str(png_path))
        boxes = json.loads(ann_path.read_text()).get("boxes", [])
        preview = _render_preview(img, boxes)

        preview_dir = root / args.preview_dir
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{_slug(args.image)}_preview.png"
        cv2.imwrite(str(preview_path), preview)
        print(f"Saved preview ({len(boxes)} box(es)) to {preview_path}")

        window = f"{args.image} — {len(boxes)} box(es) — press any key to close"
        cv2.imshow(window, preview)
        cv2.waitKey(0)
        cv2.destroyWindow(window)
        return

    if args.reset:
        reset_path = _find_annotation_file(annotations_dir, args.image)
        if reset_path is not None:
            reset_path.unlink()
            print(f"Cleared labels for '{args.image}' ({reset_path.name} deleted).")
        else:
            print(f"'{args.image}' has no annotation file to clear.")
        return

    png_path = _resolve_preprocessed_path(db_path, args.image)
    if png_path is None or not png_path.exists():
        print(
            f"No preprocessed image found for '{args.image}'. "
            "Run scripts/batch_preprocess.py first (or scripts/batch_inference.py)."
        )
        return

    existing_path = _find_annotation_file(annotations_dir, args.image)
    out_path = existing_path or annotations_dir / f"{_slug(args.image)}.json"
    existing_boxes: list[dict] = []
    if existing_path is not None:
        existing = json.loads(existing_path.read_text())
        existing_boxes = existing.get("boxes", [])
        if not args.append:
            print(f"{existing_path.name} already has {len(existing_boxes)} box(es) — they will be REPLACED.")
            print("Re-run with --append to add to them instead of overwriting.")

    img = cv2.imread(str(png_path))
    if img is None:
        print(f"Could not read image: {png_path}")
        return

    window = f"{args.image} — drag a box, ENTER/SPACE to confirm, ESC when done"
    rois = cv2.selectROIs(window, img)
    cv2.destroyWindow(window)

    new_boxes: list[dict] = []
    for i, (x, y, w, h) in enumerate(rois):
        if w == 0 or h == 0:
            continue
        crop = img[int(y):int(y + h), int(x):int(x + w)]
        crop_window = f"Box {i + 1}/{len(rois)} — press any key"
        cv2.imshow(crop_window, crop)
        cv2.waitKey(0)
        cv2.destroyWindow(crop_window)

        ft = _prompt_failure_type()
        new_boxes.append({
            "x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h),
            "failure_type": ft,
        })

    if not new_boxes:
        print("No boxes drawn — nothing saved.")
        return

    all_boxes = existing_boxes + new_boxes if args.append else new_boxes
    out_path.write_text(json.dumps({"source_filename": args.image, "boxes": all_boxes}, indent=2) + "\n")
    print(f"Saved {len(new_boxes)} new box(es) ({len(all_boxes)} total) to {out_path}")


if __name__ == "__main__":
    main()
