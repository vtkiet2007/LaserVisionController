# config.py
from __future__ import annotations

CONFIG_VERSION = "1.1.0-mvp-p1"

# Camera-calibrated workspace dimensions (mm) for the current vision task
CAMERA_WORKSPACE_WIDTH = 210.0
CAMERA_WORKSPACE_HEIGHT = 120.0

# Separate mechanical travel bounds of the machine (mm); do not assume equal to workspace
MACHINE_TRAVEL_X = 210.0
MACHINE_TRAVEL_Y = 120.0

# Default camera & serial settings
DEFAULT_CAM_URL = "http://172.16.129.200:4747/video"
DEFAULT_BAUDRATE = 115200

# Positioning deviation warning threshold (mm)
DEVIATION_WARN_MM = 3.0

# Vision detection, stability, and staleness thresholds
STABLE_POS_TOL_MM = 0.6
STABLE_ANG_TOL_DEG = 1.5
STALE_TIMEOUT_S = 0.50

# MVP safety guard: keep laser firing disabled during positioning & scan verification
ALLOW_LASER_FIRING = False