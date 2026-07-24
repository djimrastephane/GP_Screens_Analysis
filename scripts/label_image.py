"""CLI: interactively draw ground-truth boxes or polygon masks on a
preprocessed image.

Boxes are drawn with the mouse (via cv2.selectROIs); masks are drawn as
polygons (click points, Enter to close each one) via --masks. Both are drawn
directly on the same 640x640 preprocessed frame the detector/segmenter
operate on, prompt for a failure_type per shape, and write to
data/annotations/ in the format scripts/evaluate_detection.py (boxes) and
scripts/evaluate_segmentation.py (masks) expect.

Usage:
    python scripts/label_image.py "Picture 1.jpg"
    python scripts/label_image.py "Picture 1.jpg" --append
    python scripts/label_image.py "Picture 1.jpg" --masks
    python scripts/label_image.py "Picture 1.jpg" --preview
    python scripts/label_image.py "Picture 1.jpg" --reset
    python scripts/label_image.py "Picture 1.jpg" --reset --masks
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
import numpy as np

from src.annotation.colours import get_colour, rgb_to_bgr
from src.annotation.draw import draw_box, overlay_mask
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


def _render_preview(img, boxes: list[dict], masks: list[dict] | None = None):
    """Return a copy of img (BGR) with ground-truth masks + boxes + failure_type labels drawn."""
    out = img.copy()

    for m in masks or []:
        colour_rgb = get_colour(m["failure_type"])
        pts = np.array(m["points"], dtype=np.int32).reshape((-1, 1, 2))
        mask_canvas = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_canvas, [pts], 255)
        # overlay_mask blends arrays positionally (channel-order-agnostic) — out
        # is BGR (cv2.imread), so the colour must be pre-swapped to BGR too,
        # matching the same rgb_to_bgr() idiom draw_box uses below.
        out = overlay_mask(out, mask_canvas, rgb_to_bgr(colour_rgb), alpha=0.4)
        cv2.polylines(out, [pts], isClosed=True, color=rgb_to_bgr(colour_rgb), thickness=2)

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


def _draw_polygons(window: str, img) -> list[list[tuple[int, int]]]:
    """Interactive polygon drawing: left-click adds a point, Enter closes the
    current polygon (needs >=3 points) and starts a new one, Backspace undoes
    the last point (or reopens the last closed polygon if none is in
    progress), Esc finishes.
    """
    polygons: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    def _redraw() -> None:
        canvas = img.copy()
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if current:
            pts = np.array(current, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 200, 255), thickness=1)
            for (x, y) in current:
                cv2.circle(canvas, (x, y), 3, (0, 200, 255), -1)
        cv2.putText(canvas, "click=point  ENTER=close polygon  BKSP=undo  ESC=done",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def _on_mouse(event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            _redraw()

    # Window must exist before the callback is registered, and before the
    # first imshow — cv2.setMouseCallback silently no-ops otherwise.
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)
    _redraw()

    while True:
        # Poll (not waitKey(0)) so the HighGUI event queue keeps pumping
        # between clicks and _redraw() reflects each one immediately.
        key = cv2.waitKey(20) & 0xFF   # mask: raw waitKey() can carry extra bits on some platforms
        if key in (13, 10):            # Enter — CR=13 standard, LF=10 on some builds
            if len(current) >= 3:
                polygons.append(current)
                current = []
                _redraw()
            else:
                print("  Need at least 3 points to close a polygon.")
        elif key in (8, 127):          # Backspace=8, Delete=127 (some keyboards)
            if current:
                current.pop()
            elif polygons:
                current = polygons.pop()
            _redraw()
        elif key == 27:                # Esc
            break

    cv2.destroyWindow(window)
    if current:
        print(f"  Discarding {len(current)} unclosed point(s) — press Enter to close a polygon before Esc.")
    return polygons


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

    n_boxes: dict[str, int] = {}
    n_masks: dict[str, int] = {}
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            fn = data["source_filename"]
        except (json.JSONDecodeError, KeyError):
            continue
        if "boxes" in data:
            n_boxes[fn] = len(data["boxes"])
        if "masks" in data:
            n_masks[fn] = len(data["masks"])

    header = f"{'Filename':<25} {'Boxes':<14} {'Masks'}"
    print(header)
    print("-" * len(header))
    for (fn,) in rows:
        b = n_boxes.get(fn)
        m = n_masks.get(fn)
        b_str = f"{b} box(es)" if b is not None else "unlabeled"
        m_str = f"{m} mask(s)" if m is not None else "unlabeled"
        print(f"{fn:<25} {b_str:<14} {m_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively label ground-truth boxes or masks for one image.")
    parser.add_argument("image", nargs="?", help="source_filename as stored in images.db, e.g. 'Picture 1.jpg'")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--masks", action="store_true",
                        help="Draw ground-truth polygon masks instead of boxes")
    parser.add_argument("--append", action="store_true", help="Add to existing shapes instead of replacing them")
    parser.add_argument("--status", action="store_true", help="List which images are labeled vs. not, then exit")
    parser.add_argument("--reset", action="store_true",
                        help="Delete this image's boxes (or masks, with --masks) and exit (no drawing)")
    parser.add_argument("--preview", action="store_true",
                        help="Show this image's existing boxes+masks drawn on it, save a PNG, and exit (no drawing)")
    parser.add_argument("--preview-dir", default="data/annotations/previews")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / args.db
    annotations_dir = root / args.annotations_dir
    annotations_dir.mkdir(parents=True, exist_ok=True)

    key = "masks" if args.masks else "boxes"
    other_key = "boxes" if args.masks else "masks"
    noun = "mask" if args.masks else "box"

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
        data = json.loads(ann_path.read_text())
        boxes = data.get("boxes", [])
        masks = data.get("masks", [])
        preview = _render_preview(img, boxes, masks)

        preview_dir = root / args.preview_dir
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{_slug(args.image)}_preview.png"
        cv2.imwrite(str(preview_path), preview)
        print(f"Saved preview ({len(boxes)} box(es), {len(masks)} mask(s)) to {preview_path}")

        window = f"{args.image} — {len(boxes)} box(es), {len(masks)} mask(s) — press any key to close"
        cv2.imshow(window, preview)
        cv2.waitKey(0)
        cv2.destroyWindow(window)
        return

    if args.reset:
        reset_path = _find_annotation_file(annotations_dir, args.image)
        if reset_path is None:
            print(f"'{args.image}' has no annotation file to clear.")
            return
        data = json.loads(reset_path.read_text())
        data.pop(key, None)
        if not data.get(other_key):
            reset_path.unlink()
            print(f"Cleared {key} for '{args.image}' ({reset_path.name} deleted).")
        else:
            reset_path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Cleared {key} for '{args.image}' — kept {len(data[other_key])} {other_key}.")
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
    existing_data: dict = {"source_filename": args.image}
    existing_items: list[dict] = []
    if existing_path is not None:
        existing_data = json.loads(existing_path.read_text())
        existing_items = existing_data.get(key, [])
        if not args.append:
            print(f"{existing_path.name} already has {len(existing_items)} {noun}(s) — they will be REPLACED.")
            print("Re-run with --append to add to them instead of overwriting.")

    img = cv2.imread(str(png_path))
    if img is None:
        print(f"Could not read image: {png_path}")
        return

    new_items: list[dict] = []
    if args.masks:
        window = f"{args.image} — draw polygons (see on-screen help)"
        polygons = _draw_polygons(window, img)
        for i, poly in enumerate(polygons):
            preview = img.copy()
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            crop_window = f"Mask {i + 1}/{len(polygons)} — press any key"
            cv2.imshow(crop_window, preview)
            cv2.waitKey(0)
            cv2.destroyWindow(crop_window)

            ft = _prompt_failure_type()
            new_items.append({"points": [[int(x), int(y)] for x, y in poly], "failure_type": ft})
    else:
        window = f"{args.image} — drag a box, ENTER/SPACE to confirm, ESC when done"
        rois = cv2.selectROIs(window, img)
        cv2.destroyWindow(window)

        for i, (x, y, w, h) in enumerate(rois):
            if w == 0 or h == 0:
                continue
            crop = img[int(y):int(y + h), int(x):int(x + w)]
            crop_window = f"Box {i + 1}/{len(rois)} — press any key"
            cv2.imshow(crop_window, crop)
            cv2.waitKey(0)
            cv2.destroyWindow(crop_window)

            ft = _prompt_failure_type()
            new_items.append({
                "x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h),
                "failure_type": ft,
            })

    if not new_items:
        print(f"No {key} drawn — nothing saved.")
        return

    all_items = existing_items + new_items if args.append else new_items
    existing_data["source_filename"] = args.image
    existing_data[key] = all_items
    out_path.write_text(json.dumps(existing_data, indent=2) + "\n")
    print(f"Saved {len(new_items)} new {noun}(s) ({len(all_items)} total) to {out_path}")


if __name__ == "__main__":
    main()
