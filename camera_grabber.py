# camera_grabber.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    seq: int

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.ts)


def open_capture(source, size=(1280, 720)) -> cv2.VideoCapture:
    """Open either a local camera index (int/digit string) or a network stream URL."""
    print(f"[CameraGrabber] Opening source: {source}")
    is_index = isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit())

    if is_index:
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        except Exception as e:
            print(f"[CameraGrabber] Camera resolution set warning: {e}")
    else:
        url = str(source).strip()
        cap = cv2.VideoCapture(url, cv2.CAP_ANY)
        if not cap.isOpened() and hasattr(cv2, "CAP_FFMPEG"):
            print(f"[CameraGrabber] Default backend failed, trying FFMPEG for {url}")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"[CameraGrabber] Buffer size set warning: {e}")

    return cap


class FrameGrabber:
    def __init__(self, source, size=(1280, 720)):
        self.source = source
        self.size = size
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.connected = False
        self.error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FrameGrabberThread")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    @staticmethod
    def _release(cap) -> None:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

    def _loop(self) -> None:
        cap = None
        last = time.monotonic()
        try:
            while not self._stop.is_set():
                if cap is None or not cap.isOpened():
                    self._release(cap)
                    try:
                        cap = open_capture(self.source, self.size)
                    except Exception as e:
                        cap = None
                        with self._lock:
                            self.error = str(e)
                        print(f"[FrameGrabber] Exception opening capture: {e}")

                    if cap is None or not cap.isOpened():
                        with self._lock:
                            self.connected = False
                            if not self.error:
                                self.error = "Failed to open camera stream"
                        self._stop.wait(1.0)
                        continue

                try:
                    ok, img = cap.read()
                except Exception as e:
                    ok, img = False, None
                    with self._lock:
                        self.error = f"OpenCV read exception: {e}"
                    print(f"[FrameGrabber] Read exception: {e}")

                if not ok or img is None or img.size == 0:
                    with self._lock:
                        self.connected = False
                        self.error = "Lost video stream, reconnecting..."
                    self._release(cap)
                    cap = None
                    self._stop.wait(0.5)
                    continue

                now = time.monotonic()
                with self._lock:
                    self.connected = True
                    self.error = ""
                    self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
                    last = now
                    self._seq += 1
                    self._frame = Frame(img, now, self._seq)
        finally:
            self._release(cap)
            with self._lock:
                self.connected = False