# main.py
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calibration import (
    CAL_FILE,
    delete_correction,
    fit_correction,
    load_correction,
    make_grid,
    save_correction,
)
from camera_grabber import FrameGrabber
from config import (
    ALLOW_LASER_FIRING,
    CAMERA_WORKSPACE_HEIGHT,
    CAMERA_WORKSPACE_WIDTH,
    DEFAULT_BAUDRATE,
    DEFAULT_CAM_URL,
    MACHINE_TRAVEL_X,
    MACHINE_TRAVEL_Y,
)
from serial_link import GrblLink
from vision_core import DetectionStatus, VisionWorkspace, WorkpieceInfo

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None


def _dspin(lo, hi, val, step=1.0, dec=1, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


def _btn(text, slot, style="") -> QPushButton:
    b = QPushButton(text)
    if style:
        b.setStyleSheet(style)
    b.clicked.connect(slot)
    return b


STYLE_BLUE = "background-color: #2b5b84; color: white; font-weight: bold;"
STYLE_TEAL = "background-color: #00838f; color: white; font-weight: bold;"
STYLE_GREY = "background-color: #37474f; color: white; font-weight: bold;"
STYLE_RED = "background-color: #b00020; color: white; font-weight: bold; padding: 8px;"
STYLE_ORANGE = "background-color: #e65100; color: white; font-weight: bold;"


class CameraView(QLabel):
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__("No Camera Stream Available")
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background-color: #141414; color: #c8c8c8; border: 1px solid #333;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_size = None

    def show_bgr(self, bgr: np.ndarray) -> None:
        h, w, ch = bgr.shape
        self._img_size = (w, h)
        q = QImage(bgr.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.setPixmap(
            QPixmap.fromImage(q).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def mousePressEvent(self, event):
        pm = self.pixmap()
        if self._img_size is None or pm is None or pm.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x() - (self.width() - pm.width()) / 2
        y = event.position().y() - (self.height() - pm.height()) / 2
        if 0 <= x < pm.width() and 0 <= y < pm.height():
            w, h = self._img_size
            self.clicked.emit(min(x * w / pm.width(), w - 1), min(y * h / pm.height(), h - 1))


class MainWindow(QMainWindow):
    log_sig = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision-Based Laser Controller - Phase 1 (P0 + P1)")
        self.resize(1320, 820)

        self.vision = VisionWorkspace(
            ws_w=CAMERA_WORKSPACE_WIDTH,
            ws_h=CAMERA_WORKSPACE_HEIGHT,
            travel_x=MACHINE_TRAVEL_X,
            travel_y=MACHINE_TRAVEL_Y,
        )
        self.link = GrblLink(self)
        self.link.line_received.connect(lambda s: self.log(f"< {s}"))
        self.link.link_error.connect(lambda s: self.log(f"SERIAL ERROR: {s}"))

        self.grabber: Optional[FrameGrabber] = None
        self._last_seq = -1
        self._last_frame_size: Optional[tuple[int, int]] = None
        self.target_vision_mm: Optional[tuple[float, float]] = None
        self.latest_workpiece: Optional[WorkpieceInfo] = None
        self.current_pattern_vision_mm: Optional[np.ndarray] = None

        self._temp_corners: list[tuple[float, float]] = []
        self._abort_event = threading.Event()

        # Machine mapping calibration state (vision_mm -> machine_mm)
        self._cal_grid: list[tuple[float, float]] = []
        self._cal_pairs: list = []  # (grid_index, (vx, vy), (mx, my))
        self._cal_idx = 0
        self._cal_waiting = False

        self.log_sig.connect(self._append_log)
        self.init_ui()
        self.refresh_ports()
        self._load_saved_calibration()

        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self.tick)
        self.dro_timer = QTimer(self)
        self.dro_timer.timeout.connect(self.update_dro)
        self.dro_timer.start(200)

    def init_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.view = CameraView()
        self.view.clicked.connect(self.on_camera_click)
        root.addWidget(self.view, 3)

        right = QWidget()
        right.setMinimumWidth(440)
        right.setMaximumWidth(490)
        rl = QVBoxLayout(right)
        root.addWidget(right, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_connect(), "Connect")
        self.tabs.addTab(self._build_tab_workspace(), "Workspace")
        self.tabs.addTab(self._build_tab_engraving(), "Engrave Setup")
        self.tabs.addTab(self._build_tab_control(), "Control")
        self.tabs.addTab(self._build_tab_machine_cal(), "Machine Cal")
        rl.addWidget(self.tabs, 1)

        self.lbl_dro = QLabel("Machine: -")
        self.lbl_dro.setStyleSheet("font-family: Consolas, monospace; font-weight: bold;")
        rl.addWidget(self.lbl_dro)

        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("font-weight: bold; color: #aaa;")
        rl.addWidget(self.status_label)

        safety_note = QLabel(
            "MVP Safety: Laser firing locked OFF in code. Software stop / soft reset is NOT a physical E-stop."
        )
        safety_note.setStyleSheet("color: #ffab00; font-size: 11px;")
        safety_note.setWordWrap(True)
        rl.addWidget(safety_note)

        rl.addWidget(_btn("SOFTWARE STOP / RESET (Esc)", self.do_stop, STYLE_RED))

        self.logbox = QTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumHeight(130)
        self.logbox.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace;")
        rl.addWidget(self.logbox)

    def _build_tab_connect(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Hardware & Camera Connection")
        gl = QGridLayout(g)
        self.combo_port = QComboBox()
        gl.addWidget(QLabel("Machine Port:"), 0, 0)
        gl.addWidget(self.combo_port, 0, 1)
        gl.addWidget(_btn("Refresh", self.refresh_ports), 0, 2)

        self.combo_baud = QComboBox()
        self.combo_baud.addItems([str(DEFAULT_BAUDRATE), "9600", "57600"])
        gl.addWidget(QLabel("Baudrate:"), 1, 0)
        gl.addWidget(self.combo_baud, 1, 1, 1, 2)

        self.txt_cam_url = QLineEdit(DEFAULT_CAM_URL)
        gl.addWidget(QLabel("Camera URL / ID:"), 2, 0)
        gl.addWidget(self.txt_cam_url, 2, 1, 1, 2)
        lay.addWidget(g)

        bounds_lbl = QLabel(
            f"Camera Workspace: {CAMERA_WORKSPACE_WIDTH:.0f} x {CAMERA_WORKSPACE_HEIGHT:.0f} mm | "
            f"Machine Travel: {MACHINE_TRAVEL_X:.0f} x {MACHINE_TRAVEL_Y:.0f} mm"
        )
        bounds_lbl.setStyleSheet("color: #aaa;")
        bounds_lbl.setWordWrap(True)
        lay.addWidget(bounds_lbl)

        row = QHBoxLayout()
        b = _btn("CONNECT SYSTEM", self.connect_all, STYLE_BLUE)
        b.setFixedHeight(40)
        row.addWidget(b, 3)
        row.addWidget(_btn("Disconnect", self.disconnect_all), 1)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _build_tab_workspace(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g1 = QGroupBox("1. Workspace Homography (Pixel -> Vision mm)")
        g1l = QVBoxLayout(g1)
        g1l.addWidget(
            QLabel(
                f"Click 4 corners in order:\n"
                f"1. Top-Left (0, 0)\n"
                f"2. Top-Right ({CAMERA_WORKSPACE_WIDTH:.0f}, 0)\n"
                f"3. Bottom-Right ({CAMERA_WORKSPACE_WIDTH:.0f}, {CAMERA_WORKSPACE_HEIGHT:.0f})\n"
                f"4. Bottom-Left (0, {CAMERA_WORKSPACE_HEIGHT:.0f})"
            )
        )
        self.lbl_calib_status = QLabel("Status: NOT CALIBRATED")
        self.lbl_calib_status.setStyleSheet("color: red; font-weight: bold;")
        g1l.addWidget(self.lbl_calib_status)
        g1l.addWidget(_btn("Reset Corners", self.reset_corners, STYLE_GREY))
        lay.addWidget(g1)

        g2 = QGroupBox("2. Workpiece Detection (Yellow Plastic on Wood)")
        g2l = QVBoxLayout(g2)
        g2l.addWidget(_btn("Capture Empty Background", self.capture_bg, STYLE_TEAL))

        self.chk_use_bg = QCheckBox("Use Background Subtraction")
        self.chk_use_bg.setChecked(self.vision.use_background)
        self.chk_use_bg.toggled.connect(lambda c: setattr(self.vision, "use_background", c))
        g2l.addWidget(self.chk_use_bg)

        self.chk_wp = QCheckBox("Enable Workpiece Detection")
        self.chk_wp.toggled.connect(lambda checked: setattr(self.vision, "detect_workpiece_enabled", checked))
        g2l.addWidget(self.chk_wp)

        self.lbl_wp_info = QLabel("Workpiece: LOST")
        self.lbl_wp_info.setStyleSheet("font-weight: bold; color: #00e676;")
        g2l.addWidget(self.lbl_wp_info)
        lay.addWidget(g2)

        lay.addStretch(1)
        return w

    def _build_tab_engraving(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Pattern & Position Configuration (Requires STABLE Workpiece)")
        gl = QGridLayout(g)

        self.sp_pat_w = _dspin(5, 200, 40, 1, 1, " mm")
        self.sp_pat_h = _dspin(5, 200, 30, 1, 1, " mm")
        self.sp_off_x = _dspin(-100, 100, 0, 1, 1, " mm")
        self.sp_off_y = _dspin(-100, 100, 0, 1, 1, " mm")
        self.sp_power = _dspin(0, 1000, 0, 10, 0)
        self.sp_feed = _dspin(100, 5000, 1000, 100, 0)

        gl.addWidget(QLabel("Pattern Width:"), 0, 0)
        gl.addWidget(self.sp_pat_w, 0, 1)
        gl.addWidget(QLabel("Pattern Height:"), 1, 0)
        gl.addWidget(self.sp_pat_h, 1, 1)
        gl.addWidget(QLabel("Offset X (local):"), 2, 0)
        gl.addWidget(self.sp_off_x, 2, 1)
        gl.addWidget(QLabel("Offset Y (local):"), 3, 0)
        gl.addWidget(self.sp_off_y, 3, 1)
        gl.addWidget(QLabel("Laser Power (S):"), 4, 0)
        gl.addWidget(self.sp_power, 4, 1)
        gl.addWidget(QLabel("Feed Rate (F):"), 5, 0)
        gl.addWidget(self.sp_feed, 5, 1)

        lay.addWidget(g)
        lay.addWidget(_btn("Trace Pattern Border (Laser OFF)", self.trace_pattern_border, STYLE_TEAL))
        lay.addWidget(_btn("Start Engraving (Locked in MVP)", self.start_engraving_job, STYLE_ORANGE))
        lay.addStretch(1)
        return w

    def _build_tab_control(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(_btn("Home ($H)", lambda: self.send_gcode("$H", "Homing")))
        row.addWidget(_btn("Unlock ($X)", lambda: self.send_gcode("$X", "Unlock")))
        lay.addLayout(row)

        self.lbl_target = QLabel("Selected Target: None")
        self.lbl_target.setStyleSheet("font-weight: bold; color: #00e676;")
        lay.addWidget(self.lbl_target)
        lay.addWidget(_btn("Move Laser to Selected Target", self.do_move_selected, STYLE_TEAL))
        lay.addWidget(_btn("Target = STABLE Workpiece Center", self.target_from_workpiece, STYLE_BLUE))
        lay.addStretch(1)
        return w

    def _build_tab_machine_cal(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("Machine Mapping Calibration (Vision mm -> Machine mm)")
        gl = QGridLayout(g)
        self.sp_cal_cols = _dspin(2, 8, 4, 1, 0)
        self.sp_cal_rows = _dspin(2, 6, 3, 1, 0)
        self.sp_cal_margin = _dspin(5, 40, 15, 1, 1, " mm")
        gl.addWidget(QLabel("Grid columns:"), 0, 0)
        gl.addWidget(self.sp_cal_cols, 0, 1)
        gl.addWidget(QLabel("Grid rows:"), 1, 0)
        gl.addWidget(self.sp_cal_rows, 1, 1)
        gl.addWidget(QLabel("Edge margin:"), 2, 0)
        gl.addWidget(self.sp_cal_margin, 2, 1)
        lay.addWidget(g)

        hint = QLabel(
            "1. Calibrate the 4 workspace corners first.\n"
            "2. Press Start: head moves to each grid point (laser OFF).\n"
            "3. Click the head tip/center at surface height in the camera.\n"
            "4. Press Fit && Save after recording points."
        )
        hint.setStyleSheet("color: #aaa;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.lbl_cal_progress = QLabel("Progress: -")
        self.lbl_cal_progress.setStyleSheet("font-weight: bold; color: #00e676;")
        lay.addWidget(self.lbl_cal_progress)

        self.lbl_cal_status = QLabel("Correction: NONE (identity mapping)")
        self.lbl_cal_status.setStyleSheet("font-weight: bold; color: #ffab00;")
        self.lbl_cal_status.setWordWrap(True)
        lay.addWidget(self.lbl_cal_status)

        lay.addWidget(_btn("Start Grid Calibration", self.cal_start, STYLE_BLUE))
        row = QHBoxLayout()
        row.addWidget(_btn("Skip Point", self.cal_skip, STYLE_GREY))
        row.addWidget(_btn("Undo Last", self.cal_undo, STYLE_GREY))
        lay.addLayout(row)
        lay.addWidget(_btn("Fit && Save Correction", self.cal_fit, STYLE_TEAL))
        lay.addWidget(_btn("Clear Correction", self.cal_clear, STYLE_ORANGE))
        lay.addStretch(1)
        return w

    def _load_saved_calibration(self):
        loaded = load_correction()
        if loaded is None:
            return
        res, corners, meta = loaded
        try:
            if corners and len(corners) == 4:
                pts = [(float(c[0]), float(c[1])) for c in corners]
                self.vision.set_workspace_corners(pts)
                self._temp_corners = list(pts)
                self.lbl_calib_status.setText("Status: RESTORED from saved calibration")
                self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
            self.vision.set_correction(res.matrix, calibration_id=res.calibration_id)
            self._show_cal_result(res)
            self.log(
                f"Loaded {CAL_FILE} [id={res.calibration_id}] "
                f"(RMS {res.rms_mm:.2f} mm, ver={meta.get('config_version', '-')})."
            )
        except Exception as e:
            self.log(f"Could not restore saved calibration: {e}")

    def _show_cal_result(self, res):
        loo = "n/a" if np.isnan(res.loo_rms_mm) else f"{res.loo_rms_mm:.2f} mm"
        val_txt = ""
        if res.validation is not None and res.validation.n_val_points > 0:
            val_txt = f" | Val RMS {res.validation.val_rms_mm:.2f} mm ({res.validation.n_val_points} pts)"
        self.lbl_cal_status.setText(
            f"Correction ACTIVE [{res.calibration_id}]: RMS {res.rms_mm:.2f} mm | "
            f"max {res.max_mm:.2f} mm | LOO {loo}{val_txt}"
        )
        self.lbl_cal_status.setStyleSheet("font-weight: bold; color: #00e676;")

    def cal_start(self):
        if not self.vision.is_calibrated:
            QMessageBox.warning(self, "Not Calibrated", "Calibrate the 4 workspace corners first!")
            return
        if not self.link.connected:
            QMessageBox.warning(self, "Not Connected", "Machine is not connected!")
            return
        self._cal_grid = make_grid(
            MACHINE_TRAVEL_X,
            MACHINE_TRAVEL_Y,
            int(self.sp_cal_cols.value()),
            int(self.sp_cal_rows.value()),
            self.sp_cal_margin.value(),
        )
        self._cal_pairs = []
        self._cal_idx = 0
        self._cal_waiting = True
        self.send_gcode("M5", "Laser OFF")
        self.log(f"Grid calibration started: {len(self._cal_grid)} points.")
        self._cal_goto_current()

    def _cal_goto_current(self):
        n = len(self._cal_grid)
        if self._cal_idx >= n:
            self._cal_waiting = False
            self.lbl_cal_progress.setText(f"All {n} points visited - press 'Fit && Save'")
            self.log(f"Grid pass finished: {len(self._cal_pairs)}/{n} points recorded.")
            return
        mx, my = self._cal_grid[self._cal_idx]
        self.lbl_cal_progress.setText(
            f"Point {self._cal_idx + 1}/{n}: machine ({mx:.1f}, {my:.1f}) - click head in camera"
        )
        self.send_gcode(f"$J=G90X{mx:.2f}Y{my:.2f}F1800", f"Cal point {self._cal_idx + 1}/{n}")

    def cal_record_click(self, u: float, v: float):
        if self._cal_idx >= len(self._cal_grid):
            return
        vis_pt = self.vision.pixel_to_vision_mm(u, v)
        if vis_pt is None:
            self.log("Clicked point is outside calibrated vision workspace; ignored.")
            return
        mx, my = self._cal_grid[self._cal_idx]
        self._cal_pairs.append((self._cal_idx, vis_pt, (mx, my)))
        self.log(
            f"Cal pair {self._cal_idx + 1}: vision ({vis_pt[0]:.1f}, {vis_pt[1]:.1f}) "
            f"<-> machine ({mx:.1f}, {my:.1f})"
        )
        self._cal_idx += 1
        self._cal_goto_current()

    def cal_skip(self):
        if not self._cal_waiting:
            return
        self.log(f"Skipped calibration point {self._cal_idx + 1}.")
        self._cal_idx += 1
        self._cal_goto_current()

    def cal_undo(self):
        if not self._cal_pairs:
            return
        idx, _, _ = self._cal_pairs.pop()
        self._cal_idx = idx
        self._cal_waiting = True
        self.log(f"Undid calibration point {idx + 1}.")
        self._cal_goto_current()

    def cal_fit(self):
        if len(self._cal_pairs) < 4:
            QMessageBox.warning(
                self,
                "Not enough points",
                f"Need at least 4 recorded points (have {len(self._cal_pairs)}).",
            )
            return
        vis = [p[1] for p in self._cal_pairs]
        mach = [p[2] for p in self._cal_pairs]
        try:
            res = fit_correction(vis, mach)
        except Exception as e:
            self.log(f"Calibration fit error: {e}")
            QMessageBox.warning(self, "Fit failed", str(e))
            return
        self._cal_waiting = False
        self.vision.set_correction(res.matrix, calibration_id=res.calibration_id)
        try:
            save_correction(
                res,
                self.vision.src_pts,
                camera_source=self.txt_cam_url.text().strip(),
                frame_size=self._last_frame_size,
                homography_px_to_vision=self.vision.H_px_to_vision,
            )
            self.log(f"Correction saved to {CAL_FILE} [id={res.calibration_id}].")
        except Exception as e:
            self.log(f"Could not save correction: {e}")
        self._show_cal_result(res)
        self.log(
            f"Correction fitted: RMS={res.rms_mm:.3f} mm, max={res.max_mm:.3f} mm, "
            f"LOO={res.loo_rms_mm:.3f} mm, inliers={res.n_inliers}/{res.n_points}"
        )
        self.lbl_cal_progress.setText("Done.")

    def cal_clear(self):
        self.vision.clear_correction()
        delete_correction()
        self._cal_waiting = False
        self.lbl_cal_status.setText("Correction: NONE (identity mapping)")
        self.lbl_cal_status.setStyleSheet("font-weight: bold; color: #ffab00;")
        self.log("Machine correction cleared.")

    def reset_corners(self):
        self._temp_corners.clear()
        self.vision.is_calibrated = False
        self.vision.clear_correction()
        self._cal_waiting = False
        self.lbl_cal_status.setText("Correction: NONE (identity mapping)")
        self.lbl_cal_status.setStyleSheet("font-weight: bold; color: #ffab00;")
        self.lbl_calib_status.setText("Status: RESET - Please calibrate 4 corners")
        self.lbl_calib_status.setStyleSheet("color: red; font-weight: bold;")
        self.log("Workspace corners reset; previous corr_M invalidated.")

    def capture_bg(self):
        if self.grabber is None or self.grabber.latest() is None:
            QMessageBox.warning(self, "No Stream", "Camera stream not available!")
            return
        if not self.vision.is_calibrated:
            QMessageBox.warning(self, "Not Calibrated", "Calibrate workspace before capturing background!")
            return
        fr = self.grabber.latest().image
        self.vision.capture_background(fr)
        self.log("Empty background captured successfully.")
        QMessageBox.information(self, "Success", "Background captured!")

    def target_from_workpiece(self):
        ok, reason = self.vision.can_generate_path(self.latest_workpiece)
        if not ok or self.latest_workpiece is None:
            QMessageBox.warning(self, "Workpiece Not Ready", reason)
            return
        self.target_vision_mm = self.latest_workpiece.center_mm
        self.lbl_target.setText(
            f"Target (Vision): X={self.target_vision_mm[0]:.1f} mm, Y={self.target_vision_mm[1]:.1f} mm"
        )
        self.log(
            f"Target set to STABLE workpiece center: "
            f"X={self.target_vision_mm[0]:.1f}, Y={self.target_vision_mm[1]:.1f} mm"
        )

    def log(self, msg: str):
        self.log_sig.emit(msg)

    def _append_log(self, msg: str):
        self.logbox.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def refresh_ports(self):
        self.combo_port.clear()
        ports_available = None
        if list_ports is not None:
            try:
                ports_available = list_ports.comports()
            except Exception:
                ports_available = None

        if ports_available is not None:
            for p in ports_available:
                self.combo_port.addItem(f"{p.device} ({p.description})", p.device)

        if self.combo_port.count() == 0:
            self.combo_port.addItem("COM3", "COM3")

        self.log("Refreshed COM ports list.")

    def connect_all(self):
        self._abort_event.clear()
        port = self.combo_port.currentData() or self.combo_port.currentText()
        baud = int(self.combo_baud.currentText())
        stream = self.txt_cam_url.text().strip()

        try:
            self.link.open(port, baud)
            self.log(f"Connected to machine controller at {port}")
        except Exception as e:
            self.log(f"Machine connection error: {e}")

        if self.grabber is not None:
            self.grabber.stop()
        self.grabber = FrameGrabber(int(stream) if stream.isdigit() else stream)
        self.grabber.start()
        self.cam_timer.start(30)
        self.status_label.setText("Status: Camera & Machine Connected")
        self.status_label.setStyleSheet("font-weight: bold; color: #00E676;")

    def disconnect_all(self):
        self._abort_event.set()
        if self.grabber:
            self.grabber.stop()
        self.link.close()
        self.cam_timer.stop()
        self.status_label.setText("Status: Disconnected")
        self.status_label.setStyleSheet("font-weight: bold; color: #aaa;")
        self.log("System disconnected.")

    def tick(self):
        if self.grabber is None:
            return
        fr = self.grabber.latest()
        if fr is None or fr.image is None or fr.seq == self._last_seq:
            return
        self._last_seq = fr.seq
        h, w = fr.image.shape[:2]
        self._last_frame_size = (w, h)

        self.latest_workpiece = self.vision.detect_workpiece(fr.image, frame_ts=fr.ts)
        if self.latest_workpiece is not None:
            wp = self.latest_workpiece
            self.lbl_wp_info.setText(
                f"[{wp.status.value}] {wp.size_mm[0]:.0f}x{wp.size_mm[1]:.0f}mm | "
                f"Ang: {wp.angle:.1f} deg | age: {wp.age_s:.2f}s"
            )
            self.current_pattern_vision_mm = self.vision.calculate_engraving_path(
                wp,
                self.sp_pat_w.value(),
                self.sp_pat_h.value(),
                self.sp_off_x.value(),
                self.sp_off_y.value(),
            )
        else:
            self.lbl_wp_info.setText(f"Workpiece: {DetectionStatus.LOST.value}")
            self.current_pattern_vision_mm = None

        disp = fr.image.copy()
        disp = self.vision.draw_workspace_overlay(
            disp,
            target_vision_mm=self.target_vision_mm,
            workpiece=self.latest_workpiece,
            pattern_path_vision_mm=self.current_pattern_vision_mm,
        )
        self.view.show_bgr(disp)

    def update_dro(self):
        state, wpos, _, _, _ = self.link.get_state()
        txt = f"Machine: {state}"
        if wpos is not None and self.link.connected:
            txt += f" | X={wpos[0]:.2f} Y={wpos[1]:.2f}"
        self.lbl_dro.setText(txt)

    def on_camera_click(self, u: float, v: float):
        if self._cal_waiting and self.vision.is_calibrated:
            self.cal_record_click(u, v)
            return

        if len(self._temp_corners) < 4:
            self._temp_corners.append((u, v))
            corner_names = [
                f"Top-Left (0, 0)",
                f"Top-Right ({CAMERA_WORKSPACE_WIDTH:.0f}, 0)",
                f"Bottom-Right ({CAMERA_WORKSPACE_WIDTH:.0f}, {CAMERA_WORKSPACE_HEIGHT:.0f})",
                f"Bottom-Left (0, {CAMERA_WORKSPACE_HEIGHT:.0f})",
            ]
            self.log(f"Recorded corner [{corner_names[len(self._temp_corners) - 1]}]: u={u:.1f}, v={v:.1f}")

            if len(self._temp_corners) == 4:
                try:
                    self.vision.set_workspace_corners(self._temp_corners)
                    self.lbl_calib_status.setText("Status: CALIBRATED SUCCESSFULLY")
                    self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
                    self.log("Homography 4-corner transform ready.")
                except Exception as e:
                    self.log(f"Calibration error: {e}")
                    self._temp_corners.clear()
            return

        if not self.vision.is_calibrated:
            QMessageBox.warning(self, "Not Calibrated", "Please complete the 4-corner calibration first!")
            return

        vis_pt = self.vision.pixel_to_vision_mm(u, v)
        if vis_pt is None:
            self.log(f"Click at ({u:.0f}, {v:.0f}) rejected: outside valid vision workspace.")
            return

        self.target_vision_mm = vis_pt
        self.lbl_target.setText(f"Target (Vision): X={vis_pt[0]:.1f} mm, Y={vis_pt[1]:.1f} mm")
        self.log(f"Pixel ({u:.0f}, {v:.0f}) -> Vision_mm: X={vis_pt[0]:.1f}, Y={vis_pt[1]:.1f}")

    def send_gcode(self, cmd: str, desc: str = ""):
        cmd = cmd.strip()
        if not cmd:
            return
        self.log(f"> {cmd}" + (f" ({desc})" if desc else ""))
        if self.link.connected:
            self.link.send(cmd)

    def do_move_selected(self):
        if self.target_vision_mm is None:
            QMessageBox.warning(self, "No Target", "Click inside the workspace view to select a target point!")
            return
        if not self.vision.is_valid_vision_pt(*self.target_vision_mm):
            QMessageBox.warning(self, "Invalid Target", "Target point is outside vision workspace.")
            return

        mapped = self.vision.vision_to_machine(np.array([self.target_vision_mm]), strict_bounds=True)
        if mapped is None or len(mapped) == 0:
            QMessageBox.warning(
                self,
                "Out of Machine Travel",
                f"Mapped coordinate exceeds machine travel ({MACHINE_TRAVEL_X:.0f} x {MACHINE_TRAVEL_Y:.0f} mm).",
            )
            return

        mx, my = float(mapped[0][0]), float(mapped[0][1])
        self.send_gcode(f"$J=G90X{mx:.2f}Y{my:.2f}F1800", f"Machine X={mx:.2f} Y={my:.2f}")

    def trace_pattern_border(self):
        ok, reason = self.vision.can_generate_path(self.latest_workpiece)
        if not ok or self.current_pattern_vision_mm is None or len(self.current_pattern_vision_mm) == 0:
            QMessageBox.warning(self, "Cannot Trace", f"Pattern unavailable: {reason}")
            return
        if not self.link.connected:
            QMessageBox.warning(self, "Not Connected", "Machine is not connected!")
            return

        mach_pts = self.vision.vision_to_machine(self.current_pattern_vision_mm, strict_bounds=True)
        if mach_pts is None or len(mach_pts) == 0:
            QMessageBox.warning(self, "Out of Bounds", "Mapped pattern exceeds machine travel bounds.")
            return

        feed = int(self.sp_feed.value())
        self._abort_event.clear()

        def run_trace():
            self.send_gcode("M5", "Laser OFF")
            pts_list = list(mach_pts) + [mach_pts[0]]
            for idx, pt in enumerate(pts_list):
                if self._abort_event.is_set():
                    self.log("Trace aborted.")
                    return
                tx, ty = float(pt[0]), float(pt[1])
                cmd = f"G90 G1 X{tx:.2f} Y{ty:.2f} F{feed}" if idx == 0 else f"G1 X{tx:.2f} Y{ty:.2f} F{feed}"
                resp = self.link.send_wait(cmd, timeout=15.0)
                if resp != "ok":
                    self.log(f"Trace command failed ({resp}): {cmd}")
                    return
                if not self.link.wait_for_motion_complete((tx, ty), tol_mm=0.3, timeout=30.0, abort=self._abort_event):
                    self.log("Trace motion did not reach Idle in time or was aborted.")
                    return
            self.log("Trace pattern completed.")

        threading.Thread(target=run_trace, daemon=True).start()

    def start_engraving_job(self):
        if not ALLOW_LASER_FIRING:
            QMessageBox.information(
                self,
                "Laser Disabled in MVP",
                "Laser firing is intentionally locked OFF during MVP positioning & calibration verification.",
            )
            self.log("Engraving start blocked: ALLOW_LASER_FIRING is False in MVP.")
            return

    def do_stop(self):
        self._abort_event.set()
        self.log("SOFTWARE STOP / GRBL SOFT RESET!")
        if self.link.connected:
            self.link.soft_reset()

    def closeEvent(self, event):
        self._abort_event.set()
        if self.grabber:
            self.grabber.stop()
        self.link.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()