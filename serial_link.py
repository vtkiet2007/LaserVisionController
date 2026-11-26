# serial_link.py
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Optional

from PySide6.QtCore import QObject, Signal

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


class _Pending:
    __slots__ = ("cmd", "event", "result")

    def __init__(self, cmd: str):
        self.cmd, self.event, self.result = cmd, threading.Event(), ""


class GrblLink(QObject):
    line_received = Signal(str)
    link_error = Signal(str)

    def __init__(self, parent=None, poll_interval: float = 0.15):
        super().__init__(parent)
        self.poll_interval = poll_interval
        self._ser = None
        self._running = False
        self._reader: Optional[threading.Thread] = None
        self._wlock = threading.Lock()
        self._slock = threading.Lock()
        self._pending: deque[_Pending] = deque()
        self._state, self._mpos, self._wpos = "Disconnected", None, None
        self._wco = (0.0, 0.0, 0.0)
        self._state_ts = 0.0
        self._motion_seq = 0

    @property
    def connected(self) -> bool:
        return bool(self._running and self._ser is not None and getattr(self._ser, "is_open", True))

    def open(self, port: str, baud: int) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed!")
        self.attach(serial.Serial(port, baud, timeout=0.1, write_timeout=1.0))

    def attach(self, ser) -> None:
        self.close()
        self._ser = ser
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="GrblReader")
        self._reader.start()

    def close(self) -> None:
        self._running = False
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=1.0)
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser, self._reader = None, None
        self._fail_pending("disconnected")
        with self._slock:
            self._state, self._mpos, self._wpos = "Disconnected", None, None

    def _write(self, data: bytes) -> None:
        with self._wlock:
            if self._ser is None:
                raise RuntimeError("Serial port is not open")
            self._ser.write(data)

    def send(self, cmd: str) -> Optional[_Pending]:
        cmd = cmd.strip()
        if not cmd:
            return None
        p = _Pending(cmd)
        with self._slock:
            self._pending.append(p)
        try:
            self._write(cmd.encode("utf-8") + b"\n")
        except Exception as e:
            with self._slock:
                if p in self._pending:
                    self._pending.remove(p)
            p.result = f"error: write failed ({e})"
            p.event.set()
            self.link_error.emit(str(e))
        return p

    def send_wait(self, cmd: str, timeout: float = 10.0) -> str:
        p = self.send(cmd)
        if p is None:
            return "ok"
        if not p.event.wait(timeout):
            return "timeout"
        return p.result

    def send_realtime(self, data: bytes) -> None:
        try:
            self._write(data)
        except Exception as e:
            self.link_error.emit(str(e))

    def soft_reset(self) -> None:
        self.send_realtime(b"\x18")
        self._fail_pending("reset")

    def get_state(self):
        with self._slock:
            return self._state, self._wpos, self._mpos, self._state_ts, self._motion_seq

    def wait_for_motion_complete(
        self,
        target_xy: Optional[tuple[float, float]] = None,
        tol_mm: float = 0.25,
        timeout: float = 30.0,
        abort: Optional[threading.Event] = None,
    ) -> bool:
        """Wait until GRBL finishes motion (returns to Idle) rather than merely receiving 'ok'."""
        start_ts = time.monotonic()
        saw_active = False

        while time.monotonic() - start_ts < timeout:
            if abort is not None and abort.is_set():
                return False
            if not self.connected:
                return False

            state, wpos, _, ts, _ = self.get_state()
            st_upper = state.upper()
            if st_upper.startswith("ALARM") or st_upper == "DISCONNECTED":
                return False

            if st_upper in ("RUN", "JOG", "HOLD"):
                saw_active = True

            if st_upper == "IDLE" and ts >= start_ts:
                if target_xy is not None and wpos is not None and len(wpos) >= 2:
                    dist = math.hypot(wpos[0] - target_xy[0], wpos[1] - target_xy[1])
                    if dist <= tol_mm:
                        return True
                elif saw_active or (time.monotonic() - start_ts) >= max(self.poll_interval * 1.5, 0.25):
                    return True

            time.sleep(0.04)

        return False

    def wait_until_reached(
        self,
        x: float,
        y: float,
        tol: float = 0.15,
        timeout: float = 60.0,
        abort: Optional[threading.Event] = None,
    ) -> bool:
        return self.wait_for_motion_complete((x, y), tol_mm=tol, timeout=timeout, abort=abort)

    def _fatal(self, e: Exception) -> None:
        self._running = False
        self._fail_pending("disconnected")
        self.link_error.emit(f"Serial connection lost: {e}")

    def _read_loop(self) -> None:
        buf = bytearray()
        last_poll = 0.0
        while self._running:
            now = time.monotonic()
            if now - last_poll >= self.poll_interval:
                last_poll = now
                try:
                    self._write(b"?")
                except Exception as e:
                    self._fatal(e)
                    return
            try:
                data = self._ser.read(max(1, getattr(self._ser, "in_waiting", 0) or 1))
            except Exception as e:
                self._fatal(e)
                return
            if not data:
                continue
            buf.extend(data)
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(buf[:i]).decode("utf-8", "ignore").strip()
                del buf[:i + 1]
                if line:
                    self._handle_line(line)

    def _fail_pending(self, result: str) -> None:
        with self._slock:
            items = list(self._pending)
            self._pending.clear()
        for p in items:
            p.result = result
            p.event.set()

    def _resolve_one(self, result: str) -> None:
        with self._slock:
            p = self._pending.popleft() if self._pending else None
        if p is not None:
            p.result = result
            p.event.set()

    def _handle_line(self, line: str) -> None:
        if line.startswith("<"):
            self._parse_status(line)
        elif line == "ok":
            self._resolve_one("ok")
        elif line.lower().startswith("error"):
            self._resolve_one(line)
            self.line_received.emit(line)
        elif line.upper().startswith("ALARM"):
            with self._slock:
                self._state = "Alarm"
                self._state_ts = time.monotonic()
            self._fail_pending(line)
            self.line_received.emit(line)
        elif line.startswith("Grbl") or "Grbl" in line[:12]:
            self._fail_pending("reset")
            self.line_received.emit(line)
        else:
            self.line_received.emit(line)

    def _parse_status(self, line: str) -> None:
        parts = line.strip("<>").split("|")
        state = parts[0].split(":")[0]
        mpos = wpos = wco = None
        for f in parts[1:]:
            k, _, v = f.partition(":")
            if k in ("MPos", "WPos", "WCO"):
                try:
                    vals = tuple(float(t) for t in v.split(","))
                except ValueError:
                    continue
                if k == "MPos":
                    mpos = vals
                elif k == "WPos":
                    wpos = vals
                else:
                    wco = vals
        with self._slock:
            prev_state = self._state.upper()
            if wco is not None:
                self._wco = wco
            if mpos is not None:
                self._mpos = mpos
                if wpos is None:
                    wpos = tuple(m - o for m, o in zip(mpos, self._wco))
            if wpos is not None:
                self._wpos = wpos
            self._state = state
            self._state_ts = time.monotonic()
            if state.upper() in ("RUN", "JOG") and prev_state != state.upper():
                self._motion_seq += 1