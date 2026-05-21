"""SAM (Segment Anything Model) segmenter wrapper.

Uses box prompts from detections. Requires:
    pip install segment-anything
    # download checkpoint to models/sam_vit_b_01ec64.pth
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

If the checkpoint is absent or segment-anything is not installed, ready=False
and the pipeline falls back to ContourMaskSegmenter automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from .base import BaseSegmenter
from .contour_mask import ContourMaskSegmenter

_FALLBACK = ContourMaskSegmenter()

_DEFAULT_CHECKPOINT = Path("models/sam_vit_b_01ec64.pth")
_DEFAULT_MODEL_TYPE = "vit_b"


class SAMSegmenter(BaseSegmenter):
    """Zero-shot segmenter using Meta's Segment Anything Model, box-prompted."""

    def __init__(
        self,
        checkpoint_path: str | Path = _DEFAULT_CHECKPOINT,
        model_type: str = _DEFAULT_MODEL_TYPE,
        device: str = "cpu",
    ):
        self._checkpoint = Path(checkpoint_path)
        self._model_type = model_type
        self._device = device
        self._predictor = None
        self._ready = False
        self._version: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        if not self._checkpoint.exists():
            logger.warning(
                f"SAMSegmenter: checkpoint not found at {self._checkpoint}. "
                "Download sam_vit_b_01ec64.pth to models/ to enable SAM segmentation."
            )
            return
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore

            sam = sam_model_registry[self._model_type](checkpoint=str(self._checkpoint))
            sam.to(device=self._device)
            self._predictor = SamPredictor(sam)
            self._ready = True
            self._version = f"SAM/{self._model_type}/{self._checkpoint.name}"
            logger.info(f"SAMSegmenter loaded: {self._checkpoint}")
        except ImportError:
            logger.warning(
                "segment-anything is not installed. "
                "Run: pip install segment-anything"
            )

    @property
    def name(self) -> str:
        return "SAMSegmenter"

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def ready(self) -> bool:
        return self._ready

    def segment_one(
        self,
        img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        class_name: str,
    ) -> np.ndarray:
        if not self._ready or self._predictor is None:
            return _FALLBACK.segment_one(img, x1, y1, x2, y2, class_name)

        import torch  # type: ignore

        self._predictor.set_image(img)
        box = np.array([[x1, y1, x2, y2]])
        transformed_box = self._predictor.transform.apply_boxes_torch(
            torch.tensor(box, dtype=torch.float32).to(self._device),
            img.shape[:2],
        )
        with torch.no_grad():
            masks_t, scores, _ = self._predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False,
            )

        # Take the single mask and convert to uint8
        mask_np = masks_t[0, 0].cpu().numpy().astype(np.uint8) * 255
        return mask_np
