"""T02 PoC PIPELINE — Luong Duc Manh · 2A202603155 · G01"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track

logger = logging.getLogger(__name__)

_LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _extract_features(img_path: Path) -> dict:
    try:
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.warning("Cannot read %s: %s", img_path, exc)
        return {}
    gray = _to_gray(img)
    lap_var = float(convolve(gray, _LAPLACE, mode="reflect").var())
    mean_b = float(gray.mean())
    std_b = float(gray.std())
    frac_dark = float((gray < 30).mean())
    frac_saturated = float((gray > 245).mean())
    return {
        "lap_var": lap_var,
        "mean_brightness": mean_b,
        "std_brightness": std_b,
        "frac_dark": frac_dark,
        "frac_saturated": frac_saturated,
    }


def _compute_scores(df: pd.DataFrame, features: list) -> pd.DataFrame:
    feat_df = pd.DataFrame(features)
    eps = 1e-6
    df = df.copy()
    for col in feat_df.columns:
        df[col] = feat_df[col].values
    cam_lap_median = df.groupby("camera_id")["lap_var"].transform("median")
    df["lap_var_norm"] = df["lap_var"] / (cam_lap_median + eps)
    blur_score = 1.0 / (df["lap_var_norm"] + eps)
    dark_score = df["frac_dark"]
    glare_score = df["frac_saturated"]
    contrast_score = 1.0 / (df["std_brightness"] + eps)

    def norm01(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + eps)

    df["score"] = (0.45 * norm01(blur_score) + 0.25 * norm01(dark_score)
                   + 0.20 * norm01(glare_score) + 0.10 * norm01(contrast_score)).round(6)
    conds = [df["frac_dark"] > 0.4, df["frac_saturated"] > 0.15, df["lap_var_norm"] < 0.5]
    df["pred_defect_type"] = np.select(conds, ["dark", "glare", "blur"], default="none")
    df["pred_action"] = np.where(df["score"] > df["score"].quantile(0.6), "RECAPTURE", "KEEP")
    df["reason"] = ("blur=" + df["lap_var"].round(1).astype(str)
                    + " dark=" + df["frac_dark"].round(3).astype(str)
                    + " glare=" + df["frac_saturated"].round(3).astype(str))
    return df


def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    return {"manifest": pd.read_csv(manifest) if manifest.exists() else None,
            "images": pd.read_csv(ctx.input_dir / "images.csv")}


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    images_df = data["images"].copy()
    logger.info("T02 PoC: extracting features from %d images...", len(images_df))
    features = []
    for _, row in images_df.iterrows():
        feat = _extract_features(ctx.input_dir / row["path"])
        if not feat:
            feat = {k: 0.0 for k in ["lap_var", "mean_brightness", "std_brightness",
                                      "frac_dark", "frac_saturated"]}
        features.append(feat)
    scored = _compute_scores(images_df, features)
    return scored


def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    budget = int(ctx.param("budget", 20))
    q = make_review_queue(scored, id_col="image_id", score_col="score", budget=budget,
                          reason_col="reason", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    try:
        render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                             title="T02 Raw QC Triage - Top-20 Review Queue")
    except Exception as exc:
        logger.warning("contact sheet failed: %s", exc)
    return q


def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        reviewed = load_review_decisions(reviewed_path)
        ctx.stats["n_reviewed"] = len(reviewed)
        scored = apply_overrides(scored, reviewed, id_col="image_id",
                                 value_col="pred_action", audit=ctx.audit)
    return scored


def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    budget = int(ctx.param("budget", 20))
    top_b = final.nlargest(budget, "score")
    recapture_rate = float((top_b["pred_action"] == "RECAPTURE").mean())
    threshold = 0.30
    if recapture_rate >= threshold:
        return Decision(status="REWORK", reason_code="HIGH_DEFECT_RATE",
                        reason=f"RECAPTURE rate in top-{budget} = {recapture_rate:.1%} >= {threshold:.0%}. Recapture needed.",
                        owner=ctx.config.get("owner", "LuongDucManh"), scope=ctx.track,
                        evidence=[str(ctx.out("ranked_review.csv"))])
    return Decision(status="GO", reason_code="ACCEPTABLE_DEFECT_RATE",
                    reason=f"RECAPTURE rate in top-{budget} = {recapture_rate:.1%} < {threshold:.0%}. Batch accepted.",
                    owner=ctx.config.get("owner", "LuongDucManh"), scope=ctx.track,
                    evidence=[str(ctx.out("ranked_review.csv"))])


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    out_cols = spec.pred_columns + [c for c in ["pred_defect_type", "pred_action", "reason"]
                                    if c in final.columns and c not in spec.pred_columns]
    final[out_cols].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
