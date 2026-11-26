# calibration.py
"""Machine mapping calibration: vision_mm -> machine_mm.

Keeps Homography (pixel -> vision_mm) and machine correction matrix corr_M
(vision_mm -> machine_mm) distinct, while recording validation errors and
calibration metadata in machine_calibration.json.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    CAMERA_WORKSPACE_HEIGHT,
    CAMERA_WORKSPACE_WIDTH,
    CONFIG_VERSION,
    MACHINE_TRAVEL_X,
    MACHINE_TRAVEL_Y,
)

CAL_FILE = Path("machine_calibration.json")


def make_grid(
    mach_w: float,
    mach_h: float,
    cols: int = 4,
    rows: int = 3,
    margin: float = 15.0,
) -> list[tuple[float, float]]:
    """Grid of machine points in serpentine order (short travel between points)."""
    cols, rows = max(2, int(cols)), max(2, int(rows))
    xs = np.linspace(margin, mach_w - margin, cols)
    ys = np.linspace(margin, mach_h - margin, rows)
    pts: list[tuple[float, float]] = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        pts += [(float(x), float(y)) for x in row]
    return pts


@dataclass
class ValidationErrorSummary:
    val_rms_mm: float = float("nan")
    val_max_mm: float = float("nan")
    n_val_points: int = 0
    per_point_err_mm: list[float] = field(default_factory=list)


@dataclass
class CalibrationResult:
    matrix: np.ndarray              # 3x3, vision_mm -> machine_mm (corr_M)
    rms_mm: float                   # fit error on training points
    max_mm: float
    loo_rms_mm: float               # leave-one-out error (honest estimate), nan if n < 5
    n_points: int
    n_inliers: int
    residuals_mm: np.ndarray
    calibration_id: str = ""
    validation: Optional[ValidationErrorSummary] = None


def apply_correction(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def _fit(src: np.ndarray, dst: np.ndarray, ransac: bool):
    if ransac:
        return cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    return cv2.findHomography(src, dst, 0)


def evaluate_independent_points(
    H: np.ndarray,
    val_vision_pts,
    val_machine_pts,
) -> ValidationErrorSummary:
    """Evaluate mapping accuracy on independent check points not used in fitting."""
    v_src = np.asarray(val_vision_pts, dtype=np.float32).reshape(-1, 2)
    m_dst = np.asarray(val_machine_pts, dtype=np.float32).reshape(-1, 2)
    if len(v_src) == 0 or len(v_src) != len(m_dst):
        return ValidationErrorSummary()
    pred = apply_correction(H, v_src)
    errs = np.linalg.norm(pred - m_dst, axis=1)
    return ValidationErrorSummary(
        val_rms_mm=float(np.sqrt(np.mean(errs ** 2))),
        val_max_mm=float(errs.max()),
        n_val_points=int(len(errs)),
        per_point_err_mm=[float(e) for e in errs],
    )


def fit_correction(
    vision_pts,
    machine_pts,
    val_vision_pts=None,
    val_machine_pts=None,
    calibration_id: Optional[str] = None,
) -> CalibrationResult:
    src = np.asarray(vision_pts, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(machine_pts, dtype=np.float32).reshape(-1, 2)
    n = len(src)
    if n < 4 or len(dst) != n:
        raise ValueError("Need at least 4 point pairs.")

    H, inl = _fit(src, dst, ransac=(n >= 8))
    if H is None:
        raise ValueError("Homography fit failed (points may be collinear).")

    err = np.linalg.norm(apply_correction(H, src) - dst, axis=1)
    inl_mask = inl.ravel().astype(bool) if inl is not None else np.ones(n, bool)

    loo = float("nan")
    if n >= 5:
        errs = []
        for i in range(n):
            keep = np.arange(n) != i
            Hi, _ = _fit(src[keep], dst[keep], ransac=False)
            if Hi is None:
                continue
            errs.append(np.linalg.norm(apply_correction(Hi, src[i:i + 1]) - dst[i]))
        if errs:
            loo = float(np.sqrt(np.mean(np.square(errs))))

    val_summary: Optional[ValidationErrorSummary] = None
    if val_vision_pts is not None and val_machine_pts is not None and len(val_vision_pts) > 0:
        val_summary = evaluate_independent_points(H, val_vision_pts, val_machine_pts)

    cal_id = calibration_id or f"cal_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    return CalibrationResult(
        matrix=H.astype(np.float64),
        rms_mm=float(np.sqrt(np.mean(err ** 2))),
        max_mm=float(err.max()),
        loo_rms_mm=loo,
        n_points=n,
        n_inliers=int(inl_mask.sum()),
        residuals_mm=err,
        calibration_id=cal_id,
        validation=val_summary,
    )


def save_correction(
    res: CalibrationResult,
    corners_px,
    path: Path = CAL_FILE,
    camera_source: str = "",
    frame_size: Optional[tuple[int, int]] = None,
    homography_px_to_vision: Optional[np.ndarray] = None,
) -> None:
    """Save corr_M, workspace corners, independent validation metrics, and metadata."""
    cal_id = res.calibration_id or f"cal_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    val_dict = None
    if res.validation is not None:
        val_dict = {
            "val_rms_mm": None if np.isnan(res.validation.val_rms_mm) else res.validation.val_rms_mm,
            "val_max_mm": None if np.isnan(res.validation.val_max_mm) else res.validation.val_max_mm,
            "n_val_points": res.validation.n_val_points,
            "per_point_err_mm": res.validation.per_point_err_mm,
        }

    data = {
        "calibration_id": cal_id,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_version": CONFIG_VERSION,
        "camera_source": str(camera_source),
        "frame_size": list(frame_size) if frame_size is not None else None,
        "workspace_mm": [CAMERA_WORKSPACE_WIDTH, CAMERA_WORKSPACE_HEIGHT],
        "machine_travel_mm": [MACHINE_TRAVEL_X, MACHINE_TRAVEL_Y],
        "corners_px": np.asarray(corners_px, dtype=float).tolist(),
        "homography_px_to_vision": (
            np.asarray(homography_px_to_vision, dtype=float).tolist()
            if homography_px_to_vision is not None
            else None
        ),
        "corr_matrix": res.matrix.tolist(),
        "matrix": res.matrix.tolist(),
        "rms_mm": res.rms_mm,
        "max_mm": res.max_mm,
        "loo_rms_mm": None if np.isnan(res.loo_rms_mm) else res.loo_rms_mm,
        "n_points": res.n_points,
        "n_inliers": res.n_inliers,
        "residuals_mm": res.residuals_mm.tolist(),
        "independent_validation": val_dict,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_correction(path: Path = CAL_FILE) -> Optional[tuple[CalibrationResult, list, dict]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        val_summary: Optional[ValidationErrorSummary] = None
        v_raw = d.get("independent_validation")
        if isinstance(v_raw, dict) and v_raw.get("n_val_points", 0) > 0:
            val_summary = ValidationErrorSummary(
                val_rms_mm=float("nan") if v_raw.get("val_rms_mm") is None else float(v_raw["val_rms_mm"]),
                val_max_mm=float("nan") if v_raw.get("val_max_mm") is None else float(v_raw["val_max_mm"]),
                n_val_points=int(v_raw.get("n_val_points", 0)),
                per_point_err_mm=[float(x) for x in v_raw.get("per_point_err_mm", [])],
            )

        mat_raw = d.get("corr_matrix", d.get("matrix"))
        res = CalibrationResult(
            matrix=np.array(mat_raw, dtype=np.float64),
            rms_mm=float(d["rms_mm"]),
            max_mm=float(d["max_mm"]),
            loo_rms_mm=float("nan") if d.get("loo_rms_mm") is None else float(d["loo_rms_mm"]),
            n_points=int(d["n_points"]),
            n_inliers=int(d["n_inliers"]),
            residuals_mm=np.array(d.get("residuals_mm", []), dtype=np.float64),
            calibration_id=str(d.get("calibration_id", "")),
            validation=val_summary,
        )
        meta = {
            "calibration_id": d.get("calibration_id", ""),
            "calibrated_at": d.get("calibrated_at", ""),
            "config_version": d.get("config_version", ""),
            "camera_source": d.get("camera_source", ""),
            "frame_size": d.get("frame_size"),
            "workspace_mm": d.get("workspace_mm"),
            "machine_travel_mm": d.get("machine_travel_mm"),
        }
        return res, d.get("corners_px", []), meta
    except Exception as e:
        print(f"[Calibration] Could not load {p}: {e}")
        return None


def delete_correction(path: Path = CAL_FILE) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        print(f"[Calibration] Could not delete {path}: {e}")