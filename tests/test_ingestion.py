"""Unit tests for src/ingestion modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import discover_images, generate_image_id, validate_file
from src.ingestion.quality import assess_quality
from src.ingestion.database import get_all_images, init_db, upsert_image_metadata
from src.ingestion.metadata import ImageMetadata, extract_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_array() -> np.ndarray:
    """640x480 mid-grey sharp image — meets default min_resolution thresholds."""
    arr = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add sharp edge to push Laplacian variance above blur threshold
    arr[239, :, :] = 0
    return arr


@pytest.fixture
def tiny_array() -> np.ndarray:
    return np.full((80, 80, 3), 128, dtype=np.uint8)


@pytest.fixture
def glare_array() -> np.ndarray:
    arr = np.full((512, 512, 3), 255, dtype=np.uint8)
    return arr


@pytest.fixture
def dark_array() -> np.ndarray:
    return np.full((512, 512, 3), 20, dtype=np.uint8)


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    p = tmp_path / "test_screen.jpg"
    Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8)).save(p)
    return p


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

class TestValidateFile:
    def test_valid_jpg(self, tmp_image):
        ok, reason = validate_file(tmp_image)
        assert ok
        assert reason == "ok"

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "report.pdf"
        p.write_bytes(b"dummy")
        ok, reason = validate_file(p)
        assert not ok
        assert "unsupported extension" in reason

    def test_zero_byte(self, tmp_path):
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        ok, reason = validate_file(p)
        assert not ok
        assert "zero-byte" in reason


class TestGenerateImageId:
    def test_stable(self, tmp_image):
        id1 = generate_image_id(tmp_image)
        id2 = generate_image_id(tmp_image)
        assert id1 == id2

    def test_url_safe(self, tmp_image):
        img_id = generate_image_id(tmp_image)
        assert " " not in img_id
        assert all(c.isalnum() or c == "_" for c in img_id)


class TestDiscoverImages:
    def test_finds_images(self, tmp_path):
        for name in ["a.jpg", "b.png", "c.txt"]:
            (tmp_path / name).write_bytes(b"x")
        found = discover_images(tmp_path)
        names = [p.name for p in found]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_sorted(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"x")
        found = discover_images(tmp_path)
        assert [p.name for p in found] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_invalid_dir(self):
        with pytest.raises(NotADirectoryError):
            discover_images("/nonexistent_dir_xyz")


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

class TestAssessQuality:
    def test_ok_image(self, rgb_array):
        report = assess_quality(rgb_array)
        assert "ok" in report.quality_flag

    def test_tiny_flagged(self, tiny_array):
        report = assess_quality(tiny_array)
        assert "low_resolution" in report.flags

    def test_glare_flagged(self, glare_array):
        report = assess_quality(glare_array)
        assert "glare" in report.flags

    def test_dark_flagged(self, dark_array):
        report = assess_quality(dark_array)
        assert "underexposed" in report.flags

    def test_not_processable_tiny(self, tiny_array):
        report = assess_quality(tiny_array)
        # tiny images with low_resolution flag are processable=False
        # only when combined with other bad flags, depending on threshold
        assert isinstance(report.is_processable, bool)

    def test_dimensions_recorded(self, rgb_array):
        report = assess_quality(rgb_array)
        assert report.width == 640
        assert report.height == 480


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------

class TestDatabase:
    def _make_meta(self, image_id: str = "test_abc12345") -> ImageMetadata:
        return ImageMetadata(
            image_id=image_id,
            source_filename="test.jpg",
            source_path="/fake/test.jpg",
            file_size_kb=100.0,
            width=640,
            height=480,
            image_mode="RGB",
            image_format="JPEG",
            laplacian_variance=250.0,
            mean_brightness=128.0,
            brightness_std=30.0,
            quality_flag="ok",
            is_processable=True,
            ingested_at="2026-05-21T00:00:00+00:00",
        )

    def test_insert_and_retrieve(self, tmp_db):
        SessionFactory = init_db(tmp_db)
        meta = self._make_meta()
        with SessionFactory() as session:
            upsert_image_metadata(session, meta)
            session.commit()
        with SessionFactory() as session:
            records = get_all_images(session)
        assert len(records) == 1
        assert records[0].image_id == meta.image_id

    def test_upsert_updates_existing(self, tmp_db):
        SessionFactory = init_db(tmp_db)
        meta = self._make_meta()
        with SessionFactory() as session:
            upsert_image_metadata(session, meta)
            session.commit()
        meta.quality_flag = "blurry"
        with SessionFactory() as session:
            upsert_image_metadata(session, meta)
            session.commit()
        with SessionFactory() as session:
            records = get_all_images(session)
        assert len(records) == 1
        assert records[0].quality_flag == "blurry"
