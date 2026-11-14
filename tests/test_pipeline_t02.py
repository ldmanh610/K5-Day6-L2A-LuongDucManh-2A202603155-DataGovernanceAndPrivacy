"""Tests for T02 PoC pipeline — Luong Duc Manh 2A202603155.

Chay: pytest tests/test_pipeline_t02.py -v
"""
import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ unit tests

class TestExtractFeatures:
    """Kiem tra _extract_features tren anh gia lap."""

    def _make_image(self, h=32, w=32, fill=128):
        """Tao anh gray uniform."""
        from PIL import Image
        import tempfile, pathlib
        img = Image.fromarray(np.full((h, w, 3), fill, dtype=np.uint8))
        p = pathlib.Path(tempfile.mktemp(suffix=".jpg"))
        img.save(p)
        return p

    def test_success_case_bright_image(self):
        """Anh sang day du tra ve features hop le."""
        from src.pipeline import _extract_features
        p = self._make_image(fill=180)
        feat = _extract_features(p)
        p.unlink(missing_ok=True)
        assert "lap_var" in feat
        assert "frac_dark" in feat
        assert feat["frac_dark"] == pytest.approx(0.0, abs=1e-3), "Anh sang khong the co pixel toi"

    def test_dark_image_high_frac_dark(self):
        """Anh toi phai co frac_dark cao."""
        from src.pipeline import _extract_features
        p = self._make_image(fill=10)
        feat = _extract_features(p)
        p.unlink(missing_ok=True)
        assert feat["frac_dark"] > 0.9, "Anh toi (fill=10) phai co frac_dark > 0.9"

    def test_missing_file_returns_empty(self):
        """Neu file khong ton tai -> tra ve dict rong, khong raise."""
        from src.pipeline import _extract_features
        import pathlib
        feat = _extract_features(pathlib.Path("/nonexistent/image.jpg"))
        assert feat == {}, "File khong ton tai phai tra ve {}"


class TestComputeScores:
    """Kiem tra _compute_scores tren DataFrame gia lap."""

    def _make_df(self, n=10):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "image_id": [f"img_{i}" for i in range(n)],
            "path": [f"images/img_{i}.jpg" for i in range(n)],
            "camera_id": rng.choice(["cam_A", "cam_B"], n).tolist(),
            "light": rng.choice(["day", "night"], n).tolist(),
            "split": rng.choice(["dev", "eval"], n).tolist(),
        })

    def _make_features(self, n=10, seed=42):
        rng = np.random.default_rng(seed)
        return [{"lap_var": float(rng.uniform(50, 500)),
                 "mean_brightness": float(rng.uniform(50, 200)),
                 "std_brightness": float(rng.uniform(10, 80)),
                 "frac_dark": float(rng.uniform(0, 0.5)),
                 "frac_saturated": float(rng.uniform(0, 0.3)),
                 "blockiness": float(rng.uniform(1, 20))} for _ in range(n)]

    def test_score_range_zero_to_one(self):
        """Score phai nam trong [0, 1]."""
        from src.pipeline import _compute_scores
        df = _make_df = self._make_df()
        feats = self._make_features()
        result = _compute_scores(df, feats)
        assert result["score"].between(0, 1).all(), "Score phai trong [0,1]"

    def test_dark_image_gets_high_score(self):
        """Anh toi nhat phai co score cao hon anh sang nhat."""
        from src.pipeline import _compute_scores
        df = pd.DataFrame({
            "image_id": ["dark_img", "bright_img"],
            "path": ["images/dark.jpg", "images/bright.jpg"],
            "camera_id": ["cam_A", "cam_A"],
            "light": ["night", "day"],
            "split": ["eval", "eval"],
        })
        feats = [
            {"lap_var": 300.0, "mean_brightness": 15.0, "std_brightness": 5.0,
             "frac_dark": 0.9, "frac_saturated": 0.0, "blockiness": 2.0},   # anh toi
            {"lap_var": 300.0, "mean_brightness": 180.0, "std_brightness": 40.0,
             "frac_dark": 0.0, "frac_saturated": 0.0, "blockiness": 2.0},   # anh sang
        ]
        result = _compute_scores(df, feats)
        dark_score = result.loc[result["image_id"] == "dark_img", "score"].values[0]
        bright_score = result.loc[result["image_id"] == "bright_img", "score"].values[0]
        assert dark_score > bright_score, "Anh toi phai co score cao hon anh sang"

    def test_pred_defect_type_dark(self):
        """frac_dark > 0.4 -> pred_defect_type = dark."""
        from src.pipeline import _compute_scores
        df = pd.DataFrame({
            "image_id": ["x"],
            "path": ["p"],
            "camera_id": ["cam_A"],
            "light": ["night"],
            "split": ["eval"],
        })
        feats = [{"lap_var": 200.0, "mean_brightness": 20.0, "std_brightness": 10.0,
                  "frac_dark": 0.8, "frac_saturated": 0.0, "blockiness": 2.0}]
        result = _compute_scores(df, feats)
        assert result["pred_defect_type"].values[0] == "dark"

    def test_no_score_column_causes_error(self):
        """Neu features rong, _compute_scores van chay (fallback = 0.0)."""
        from src.pipeline import _compute_scores
        df = pd.DataFrame({
            "image_id": ["x"],
            "path": ["p"],
            "camera_id": ["cam_A"],
            "light": ["day"],
            "split": ["dev"],
        })
        # Edge case: feature tat ca = 0
        feats = [{"lap_var": 0.0, "mean_brightness": 0.0, "std_brightness": 0.0,
                  "frac_dark": 0.0, "frac_saturated": 0.0, "blockiness": 0.0}]
        result = _compute_scores(df, feats)
        assert "score" in result.columns
