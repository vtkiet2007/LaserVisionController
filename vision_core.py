# vision_core.py
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from config import (
    CAMERA_WORKSPACE_HEIGHT,
    CAMERA_WORKSPACE_WIDTH,
    MACHINE_TRAVEL_X,
    MACHINE_TRAVEL_Y,
    STABLE_ANG_TOL_DEG,
    STABLE_POS_TOL_MM,
    STALE_TIMEOUT_S,
)


class DetectionStatus(str, Enum):
    LOST = "LOST"
    STALE = "STALE"
    DETECTED = "DETECTED"
    STABLE = "STABLE"


@dataclass
class WorkpieceInfo:
    center_mm: tuple[float, float]
    size_mm: tuple[float, float]       # (long side, short side)
    angle: float                       # degrees in [-90, 90)
    box_mm: np.ndarray
    contour_mm: np.ndarray
    status: DetectionStatus = DetectionStatus.DETECTED
    age_s: float = 0.0
    in_bounds: bool = True

    @property
    def stable(self) -> bool:
        return self.status == DetectionStatus.STABLE


def _normalize_angle(angle_deg: float) -> float:
    """Wrap an angle into [-90, 90)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def _shortest_angle_diff_deg(a: float, b: float) -> float:
    """Signed shortest angle difference in [-90, 90) for 180-deg symmetric axes."""
    return _normalize_angle(a - b)


class VisionWorkspace:
    WARP_SCALE = 4.0  # px per mm in top-down workspace image

    def __init__(
        self,
        ws_w: float = CAMERA_WORKSPACE_WIDTH,
        ws_h: float = CAMERA_WORKSPACE_HEIGHT,
        travel_x: float = MACHINE_TRAVEL_X,
        travel_y: float = MACHINE_TRAVEL_Y,
    ):
        self.ws_w = float(ws_w)
        self.ws_h = float(ws_h)
        self.travel_x = float(travel_x)
        self.travel_y = float(travel_y)

        # Keep backward-compatible aliases for workspace dimensions
        self.mach_w = self.ws_w
        self.mach_h = self.ws_h

        self.src_pts = np.zeros((4, 2), dtype=np.float32)
        self.dst_pts = np.array([
            [0.0, 0.0],
            [self.ws_w, 0.0],
            [self.ws_w, self.ws_h],
            [0.0, self.ws_h],
        ], dtype=np.float32)

        self.H_px_to_vision: Optional[np.ndarray] = None
        self.H_vision_to_px: Optional[np.ndarray] = None
        self.M_warp: Optional[np.ndarray] = None
        self.is_calibrated = False

        # Backward-compatible attribute names
        self.M = None
        self.M_inv = None

        # Detection settings (yellow plastic on wood background)
        self.detect_workpiece_enabled = False
        self.min_area_mm2 = 100.0
        self.max_area_mm2 = 20000.0
        self.hsv_lower = np.array([15, 80, 70], dtype=np.uint8)
        self.hsv_upper = np.array([35, 255, 255], dtype=np.uint8)

        # Optional background subtraction
        self.use_background = False
        self.bg_diff_thresh = 30
        self._bg_warped: Optional[np.ndarray] = None

        self.border_margin_mm = 1.5
        self.min_rectangularity = 0.70

        # Temporal smoothing & status tracking
        self.smooth_len = 7
        self.stable_pos_tol_mm = float(STABLE_POS_TOL_MM)
        self.stable_ang_tol_deg = float(STABLE_ANG_TOL_DEG)
        self.stale_timeout_s = float(STALE_TIMEOUT_S)
        self._hist: deque = deque(maxlen=self.smooth_len)
        self._miss_count = 0
        self.max_miss = 5
        self._last_valid_ts: Optional[float] = None
        self._last_info: Optional[WorkpieceInfo] = None

        # Second transform layer: vision_mm -> machine_mm (kept distinct from Homography)
        self.corr_M: Optional[np.ndarray] = None
        self.calibration_id: str = ""

        self.last_mask: Optional[np.ndarray] = None
        self.last_process_ms = 0.0

    # ------------------------------------------------------------------
    # Workspace calibration (Pixel -> Vision_mm)
    # ------------------------------------------------------------------
    def set_workspace_corners(self, pts_uv: list[tuple[float, float]]):
        if len(pts_uv) != 4:
            raise ValueError("Must provide exactly 4 corner points!")
        self.src_pts = np.array(pts_uv, dtype=np.float32)
        self.H_px_to_vision = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.H_vision_to_px = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        self.M = self.H_px_to_vision
        self.M_inv = self.H_vision_to_px

        s = self.WARP_SCALE
        w_px, h_px = int(round(self.ws_w * s)), int(round(self.ws_h * s))
        dst_warp = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float32)
        self.M_warp = cv2.getPerspectiveTransform(self.src_pts, dst_warp)

        self._bg_warped = None
        self._reset_tracking()
        self.corr_M = None
        self.calibration_id = ""
        self.is_calibrated = True
        print(f"[VisionWorkspace] Homography corners set: {self.src_pts.tolist()}")

    def _reset_tracking(self):
        self._hist.clear()
        self._miss_count = 0
        self._last_valid_ts = None
        self._last_info = None

    # ------------------------------------------------------------------
    # Validation & coordinate transforms (strictly separated layers)
    # ------------------------------------------------------------------
    def is_valid_vision_pt(self, x_mm: float, y_mm: float, tol: float = 1e-6) -> bool:
        return (-tol <= x_mm <= self.ws_w + tol) and (-tol <= y_mm <= self.ws_h + tol)

    def is_valid_machine_pt(self, mx: float, my: float, tol: float = 1e-6) -> bool:
        return (-tol <= mx <= self.travel_x + tol) and (-tol <= my <= self.travel_y + tol)

    def pixel_to_vision_mm(self, u: float, v: float) -> Optional[tuple[float, float]]:
        """Layer 1: Pixel (u, v) -> Vision_mm (x, y). Returns None if out of workspace."""
        if not self.is_calibrated or self.H_px_to_vision is None:
            return None
        pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), self.H_px_to_vision)
        x, y = float(pt[0, 0, 0]), float(pt[0, 0, 1])
        if not self.is_valid_vision_pt(x, y):
            return None
        return x, y

    def pixel_to_mm(self, u: float, v: float) -> Optional[tuple[float, float]]:
        """Rejects points outside the calibrated vision workspace instead of clipping."""
        return self.pixel_to_vision_mm(u, v)

    def vision_mm_to_pixel(self, x: float, y: float) -> Optional[tuple[float, float]]:
        if not self.is_calibrated or self.H_vision_to_px is None:
            return None
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), self.H_vision_to_px)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def mm_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        res = self.vision_mm_to_pixel(x, y)
        return res if res is not None else (0.0, 0.0)

    def mm_array_to_pixel(self, pts_mm: np.ndarray) -> np.ndarray:
        if not self.is_calibrated or self.H_vision_to_px is None or len(pts_mm) == 0:
            return np.array([], dtype=np.float32)
        pts = np.asarray(pts_mm, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_vision_to_px).reshape(-1, 2)

    @property
    def has_correction(self) -> bool:
        return self.corr_M is not None

    def set_correction(self, H: np.ndarray, calibration_id: str = "") -> None:
        self.corr_M = np.asarray(H, dtype=np.float64).reshape(3, 3)
        self.calibration_id = calibration_id

    def clear_correction(self) -> None:
        self.corr_M = None
        self.calibration_id = ""

    def vision_to_machine(self, pts_vision_mm, strict_bounds: bool = True) -> Optional[np.ndarray]:
        """Layer 2: Vision_mm -> Machine_mm via corr_M (or identity if no corr_M).
        If strict_bounds=True, returns None when any point exceeds machine travel bounds."""
        pts = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        if self.corr_M is not None:
            mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.corr_M).reshape(-1, 2)
        else:
            mapped = pts.astype(np.float32).copy()

        for pt in mapped:
            if not self.is_valid_machine_pt(float(pt[0]), float(pt[1])):
                if strict_bounds:
                    return None
        return mapped.astype(np.float32)

    # ------------------------------------------------------------------
    # Background & detection
    # ------------------------------------------------------------------
    def _warp(self, frame: np.ndarray) -> np.ndarray:
        s = self.WARP_SCALE
        return cv2.warpPerspective(
            frame,
            self.M_warp,
            (int(round(self.ws_w * s)), int(round(self.ws_h * s))),
        )

    def capture_background(self, frame: np.ndarray):
        if not self.is_calibrated or frame is None or frame.size == 0:
            return
        warped = self._warp(frame)
        self._bg_warped = cv2.GaussianBlur(warped, (5, 5), 0)
        self._reset_tracking()
        print("[VisionWorkspace] Background captured.")

    def clear_background(self):
        self._bg_warped = None

    def _build_mask(self, warped: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(warped, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        if self.use_background and self._bg_warped is not None:
            diff = cv2.absdiff(blur, self._bg_warped)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, fg = cv2.threshold(diff_gray, self.bg_diff_thresh, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, fg)

        m = int(round(self.border_margin_mm * self.WARP_SCALE))
        if m > 0:
            mask[:m, :] = 0
            mask[-m:, :] = 0
            mask[:, :m] = 0
            mask[:, -m:] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _stabilize_angle_against_history(self, candidate_angle: float, long_mm: float, short_mm: float) -> float:
        """Reduce sudden 90-deg flips on near-square or noisy rectangles."""
        ang = _normalize_angle(candidate_angle)
        if len(self._hist) == 0:
            return ang
        prev_ang = self._hist[-1][4]
        diff = _shortest_angle_diff_deg(ang, prev_ang)
        aspect_ratio = max(long_mm, 1e-6) / max(short_mm, 1e-6)
        if aspect_ratio < 1.20 and abs(abs(diff) - 90.0) < 15.0:
            ang = _normalize_angle(ang + (90.0 if diff > 0 else -90.0))
        return ang

    def _measure(self, warped: np.ndarray) -> Optional[tuple]:
        s = self.WARP_SCALE
        mask = self._build_mask(warped)
        self.last_mask = mask

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for c in cnts:
            area_px = cv2.contourArea(c)
            area_mm2 = area_px / (s * s)
            if not (self.min_area_mm2 <= area_mm2 <= self.max_area_mm2):
                continue
            rect = cv2.minAreaRect(c)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area <= 0:
                continue
            if area_px / rect_area < self.min_rectangularity:
                continue
            if area_px > best_area:
                best_area = area_px
                best = (c, rect)

        if best is None:
            return None

        cnt, rect = best
        (cx, cy), (rw, rh), ang = rect
        if rw >= rh:
            long_px, short_px, angle = rw, rh, ang
        else:
            long_px, short_px, angle = rh, rw, ang + 90.0

        long_mm, short_mm = long_px / s, short_px / s
        angle = self._stabilize_angle_against_history(angle, long_mm, short_mm)

        box_mm = cv2.boxPoints(((cx, cy), (long_px, short_px), angle)) / s
        return (cx / s, cy / s, long_mm, short_mm, angle, box_mm, cnt.reshape(-1, 2) / s)

    def _smooth(self, meas: tuple, now_ts: float) -> WorkpieceInfo:
        cx, cy, lng, sht, ang, box_mm, cnt_mm = meas
        self._hist.append((cx, cy, lng, sht, ang))
        arr = np.array(self._hist, dtype=np.float64)

        med_cx, med_cy = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        med_l, med_s = float(np.median(arr[:, 2])), float(np.median(arr[:, 3]))

        a2 = np.radians(arr[:, 4] * 2.0)
        mean_ang = np.degrees(np.arctan2(np.sin(a2).mean(), np.cos(a2).mean())) / 2.0
        med_a = _normalize_angle(float(mean_ang))

        in_bounds = self.is_valid_vision_pt(med_cx, med_cy)
        is_stable = False
        if len(self._hist) >= self.smooth_len and in_bounds:
            pos_spread = max(float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1])))
            dang = np.abs([_shortest_angle_diff_deg(float(a), med_a) for a in arr[:, 4]])
            is_stable = bool(pos_spread <= self.stable_pos_tol_mm and float(dang.max()) <= self.stable_ang_tol_deg)

        status = DetectionStatus.STABLE if is_stable else DetectionStatus.DETECTED
        rect = ((med_cx, med_cy), (med_l, med_s), med_a)
        box_smooth = cv2.boxPoints(rect)

        return WorkpieceInfo(
            center_mm=(med_cx, med_cy),
            size_mm=(med_l, med_s),
            angle=med_a,
            box_mm=box_smooth.astype(np.float32),
            contour_mm=cnt_mm,
            status=status,
            age_s=0.0,
            in_bounds=in_bounds,
        )

    def detect_workpiece(self, frame: np.ndarray, frame_ts: Optional[float] = None) -> Optional[WorkpieceInfo]:
        if not self.is_calibrated or not self.detect_workpiece_enabled:
            self._last_info = None
            return None
        if frame is None or frame.size == 0:
            print("[VisionDebug] Empty frame received in detect_workpiece!")
            return self._handle_stale_or_lost(time.monotonic())

        now_ts = float(frame_ts) if frame_ts is not None else time.monotonic()
        t0 = cv2.getTickCount()
        try:
            warped = self._warp(frame)
            meas = self._measure(warped)
        except Exception as e:
            print(f"[VisionDebug] Detection error: {e}")
            self.last_process_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
            return self._handle_stale_or_lost(now_ts)

        if meas is None:
            self.last_process_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
            return self._handle_stale_or_lost(now_ts)

        self._miss_count = 0
        if len(self._hist) > 0:
            prev = self._hist[-1]
            if np.hypot(meas[0] - prev[0], meas[1] - prev[1]) > 8.0:
                self._hist.clear()

        info = self._smooth(meas, now_ts)
        self._last_valid_ts = now_ts
        self._last_info = info
        self.last_process_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
        return info

    def _handle_stale_or_lost(self, now_ts: float) -> Optional[WorkpieceInfo]:
        self._miss_count += 1
        age = (now_ts - self._last_valid_ts) if self._last_valid_ts is not None else float("inf")

        if self._miss_count > self.max_miss or age > self.stale_timeout_s * 2.0:
            self._hist.clear()
            self._last_info = None
            return None

        if self._last_info is not None:
            is_stale = age > self.stale_timeout_s
            status = DetectionStatus.STALE if is_stale else DetectionStatus.DETECTED
            return WorkpieceInfo(
                center_mm=self._last_info.center_mm,
                size_mm=self._last_info.size_mm,
                angle=self._last_info.angle,
                box_mm=self._last_info.box_mm.copy(),
                contour_mm=np.empty((0, 2), dtype=np.float32),
                status=status,
                age_s=age,
                in_bounds=self._last_info.in_bounds,
            )
        return None

    def can_generate_path(self, workpiece: Optional[WorkpieceInfo]) -> tuple[bool, str]:
        if workpiece is None:
            return False, "Workpiece status: LOST"
        if workpiece.status == DetectionStatus.STALE or workpiece.age_s > self.stale_timeout_s:
            return False, f"Workpiece data is STALE (age={workpiece.age_s:.2f}s)"
        if workpiece.status != DetectionStatus.STABLE:
            return False, f"Workpiece is not STABLE (status={workpiece.status.value})"
        if not workpiece.in_bounds or not self.is_valid_vision_pt(*workpiece.center_mm):
            return False, "Workpiece center is outside vision workspace"
        return True, "OK"

    # ------------------------------------------------------------------
    # Engraving path (rejects out-of-bounds paths instead of clipping)
    # ------------------------------------------------------------------
    def calculate_engraving_path(
        self,
        workpiece: WorkpieceInfo,
        width_mm: float,
        height_mm: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> Optional[np.ndarray]:
        ok, _ = self.can_generate_path(workpiece)
        if not ok:
            return None
        if width_mm <= 0 or height_mm <= 0:
            return None

        cx, cy = workpiece.center_mm
        theta = np.radians(workpiece.angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

        local_pts = np.array([
            [-width_mm / 2.0, -height_mm / 2.0],
            [width_mm / 2.0, -height_mm / 2.0],
            [width_mm / 2.0, height_mm / 2.0],
            [-width_mm / 2.0, height_mm / 2.0],
        ], dtype=np.float64)

        local_off = np.array([offset_x, offset_y], dtype=np.float64)
        pts = (local_pts + local_off) @ R.T + np.array([cx, cy], dtype=np.float64)

        for pt in pts:
            if not self.is_valid_vision_pt(float(pt[0]), float(pt[1])):
                return None

        return pts.astype(np.float32)

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------
    def draw_workspace_overlay(
        self,
        img: np.ndarray,
        target_vision_mm: Optional[tuple[float, float]] = None,
        workpiece: Optional[WorkpieceInfo] = None,
        pattern_path_vision_mm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_calibrated:
            cv2.putText(img, "WORKSPACE NOT CALIBRATED!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img

        pts_int = self.src_pts.astype(np.int32)
        cv2.polylines(img, [pts_int], True, (0, 255, 255), 2)
        cv2.circle(img, tuple(int(v) for v in pts_int[0]), 6, (0, 0, 255), -1)
        cv2.putText(img, "Origin (0,0)", (int(pts_int[0][0]) + 10, int(pts_int[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if workpiece is not None:
            box_px = self.mm_array_to_pixel(workpiece.box_mm)
            color_map = {
                DetectionStatus.STABLE: (0, 255, 0),
                DetectionStatus.DETECTED: (255, 0, 255),
                DetectionStatus.STALE: (0, 165, 255),
                DetectionStatus.LOST: (0, 0, 255),
            }
            color = color_map.get(workpiece.status, (255, 0, 255))
            if len(box_px) > 0:
                cv2.polylines(img, [box_px.astype(np.int32)], True, color, 2)
            cx, cy = self.mm_to_pixel(*workpiece.center_mm)
            cv2.drawMarker(img, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                img,
                f"WP {workpiece.size_mm[0]:.0f}x{workpiece.size_mm[1]:.0f}mm "
                f"{workpiece.angle:.1f}deg [{workpiece.status.value}] {self.last_process_ms:.0f}ms",
                (int(cx) + 10, int(cy) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if pattern_path_vision_mm is not None and len(pattern_path_vision_mm) > 0:
            path_px = self.mm_array_to_pixel(pattern_path_vision_mm)
            if len(path_px) > 0:
                cv2.polylines(img, [path_px.astype(np.int32)], True, (0, 255, 0), 2)
                p0 = (int(path_px[0][0]), int(path_px[0][1]))
                cv2.circle(img, p0, 5, (0, 255, 0), -1)
                cv2.putText(img, "Engraving Target Path", (p0[0] + 10, p0[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        if target_vision_mm is not None and self.is_valid_vision_pt(*target_vision_mm):
            u, v = self.mm_to_pixel(target_vision_mm[0], target_vision_mm[1])
            cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, f"Target ({target_vision_mm[0]:.1f}, {target_vision_mm[1]:.1f})",
                        (int(u) + 10, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img