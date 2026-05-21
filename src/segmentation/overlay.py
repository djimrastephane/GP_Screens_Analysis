"""Draw segmentation overlays on preprocessed images."""

from __future__ import annotations

import cv2
import numpy as np

# BGR colours per class (OpenCV convention)
_CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "dark_blob": (0, 60, 220),      # red-orange
    "color_anomaly": (0, 200, 255), # yellow
}
_DEFAULT_COLOUR = (180, 180, 180)

_ALPHA = 0.40   # mask fill transparency
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1


def draw_overlay(
    img: np.ndarray,
    masks_and_detections: list[tuple[np.ndarray, dict]],
) -> np.ndarray:
    """Composite segmentation overlay onto an RGB uint8 image.

    Args:
        img: uint8 RGB HxWx3.
        masks_and_detections: list of (mask_HxW, detection_dict) pairs,
            where detection_dict has keys: x1, y1, x2, y2, class_name, confidence.

    Returns:
        RGB uint8 annotated image.
    """
    # Work in BGR for OpenCV drawing, convert back at end
    canvas = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

    for mask, det in masks_and_detections:
        cls = det.get("class_name", "unknown")
        colour = _CLASS_COLOURS.get(cls, _DEFAULT_COLOUR)
        conf = det.get("confidence", 0.0)

        # Semi-transparent filled mask
        fill = np.zeros_like(canvas)
        fill[mask > 0] = colour
        canvas = cv2.addWeighted(canvas, 1.0, fill, _ALPHA, 0)

        # Bounding box
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)

        # Label with class and confidence
        label = f"{cls} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(canvas, (x1, label_y - th - 2), (x1 + tw, label_y + baseline), colour, -1)
        cv2.putText(
            canvas, label, (x1, label_y),
            _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICKNESS, cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
