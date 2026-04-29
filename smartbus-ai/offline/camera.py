
"""
camera.py — PiCamera2 Manager
Handles both PiCamera2 (primary) and USB webcam (fallback).
Produces JPEG frames for MJPEG streaming + detector access.

Why PiCamera2?
  - Official Raspberry Pi library for Camera Module v2/v3
  - Hardware ISP pipeline → better image quality than OpenCV direct
  - Tunable resolution for lightweight inference (320x240 for YOLO)
"""

import threading
import time
import logging
import numpy as np
import cv2
from datetime import datetime

log = logging.getLogger('Camera')

# Detection resolution — keeps CPU/RAM light on Pi
DETECT_W, DETECT_H = 320, 240
# Stream preview resolution
STREAM_W, STREAM_H = 640, 480


class CameraManager:
    def __init__(self):
        self.is_running = False
        self._lock = threading.Lock()
        self._raw_frame = None        # numpy BGR for detector
        self._annotated_jpeg = None   # JPEG bytes for MJPEG stream
        self._thread = None
        self._use_picamera = False
        self._cap = None              # OpenCV VideoCapture fallback

    # ── Start
    def start(self):
        """Try PiCamera2 first, fall back to OpenCV VideoCapture."""
        try:
            from picamera2 import Picamera2
            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"format": "BGR888", "size": (STREAM_W, STREAM_H)},
                lores={"format": "YUV420",  "size": (DETECT_W, DETECT_H)},
            )
            self._picam.configure(config)
            self._picam.start()
            self._use_picamera = True
            log.info(f'PiCamera2 started ({STREAM_W}x{STREAM_H} stream, {DETECT_W}x{DETECT_H} detect)')
        except Exception as e:
            log.warning(f'PiCamera2 unavailable ({e}), falling back to OpenCV VideoCapture')
            self._use_picamera = False
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                # Try index 1 or use synthetic frames in demo mode
                self._cap = cv2.VideoCapture(1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info('Camera thread started')

    # ── Stop
    def stop(self):
        self.is_running = False
        if self._use_picamera and hasattr(self, '_picam'):
            self._picam.stop()
        if self._cap:
            self._cap.release()

    # ── Capture loop
    def _capture_loop(self):
        while self.is_running:
            try:
                frame = self._grab_frame()
                if frame is not None:
                    with self._lock:
                        self._raw_frame = frame
            except Exception as e:
                log.error(f'Capture error: {e}')
            time.sleep(0.033)  # ~30fps capture, detector runs slower

    def _grab_frame(self):
        if self._use_picamera:
            return self._picam.capture_array("main")  # BGR888
        else:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    return frame
            # Demo/test mode: synthetic frame with crowd simulation
            return self._synthetic_frame()

    def _synthetic_frame(self):
        """Generate a fake frame for demo / unit testing."""
        frame = np.zeros((STREAM_H, STREAM_W, 3), dtype=np.uint8)
        frame[:] = (20, 20, 35)  # dark bg
        # Draw fake "people" as coloured rectangles
        import random
        n = random.randint(5, 30)
        for _ in range(n):
            x = random.randint(0, STREAM_W - 40)
            y = random.randint(60, STREAM_H - 80)
            cv2.rectangle(frame, (x, y), (x+30, y+60), (80, 180, 80), -1)
        # timestamp
        ts = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, f'DEMO {ts}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    # ── Public accessors
    def get_raw_frame(self):
        """Returns latest BGR numpy frame (for detector)."""
        with self._lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def get_detect_frame(self):
        """Returns resized low-res frame for YOLO inference."""
        frame = self.get_raw_frame()
        if frame is None:
            return None
        return cv2.resize(frame, (DETECT_W, DETECT_H))

    def set_annotated_frame(self, jpeg_bytes: bytes):
        """Called by detector after drawing bounding boxes."""
        with self._lock:
            self._annotated_jpeg = jpeg_bytes

    def get_annotated_frame(self) -> bytes | None:
        """Returns latest JPEG for MJPEG stream."""
        with self._lock:
            if self._annotated_jpeg:
                return self._annotated_jpeg
            # Fall back: encode raw frame
            if self._raw_frame is not None:
                _, enc = cv2.imencode('.jpg', self._raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                return enc.tobytes()
        return None
