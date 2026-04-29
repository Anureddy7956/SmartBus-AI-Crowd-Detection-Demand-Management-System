"""
crowd_detector.py — YOLOv8n Person Detector
Runs inference on PiCamera2 frames, counts crowd, triggers alerts.

Why YOLOv8n?
  - 'n' = nano variant, ~3.2M params
  - INT8 quantized ONNX runs ~8-12 fps on Pi 4B (1 core)
  - Only detects class 0 (person) → discard rest → saves RAM
  - No GPU needed — CPU inference is sufficient for bus stops

Why OpenCV DNN instead of ultralytics runtime?
  - ultralytics pulls PyTorch (~1.5 GB) — too heavy for 4GB Pi
  - cv2.dnn loads ONNX directly — ~80MB, starts in <2s
  - Same accuracy, fraction of memory
"""

import cv2
import numpy as np
import threading
import time
import logging
from datetime import datetime

log = logging.getLogger('Detector')

# YOLO config
MODEL_PATH  = 'models/yolov8n.onnx'
INPUT_SIZE  = 320          # YOLOv8 input (must match export)
CONF_THRESH = 0.40
NMS_THRESH  = 0.45
PERSON_CLASS = 0

# Crowd thresholds (persons at stop)
CROWD_LOW  = 15
CROWD_MED  = 35
CROWD_HIGH = 60

# Inference interval — 1 frame every N seconds
INFER_INTERVAL = 2.0       # 0.5 fps inference, plenty for bus stops


class CrowdDetector:
    def __init__(self, camera, db, ai, announce, bus_mgr):
        self.camera   = camera
        self.db       = db
        self.ai       = ai
        self.announce = announce
        self.bus_mgr  = bus_mgr

        self.is_running   = False
        self._thread      = None
        self._start_time  = None
        self._net         = None          # cv2 DNN net
        self._last_count  = 0
        self._boxes       = []            # last detected boxes for annotation

        # Current active stop (set by dashboard selection)
        self.active_stop  = 'stop_majestic'

    # ── Start
    def start(self):
        self._load_model()
        self._start_time = time.time()
        self.is_running  = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()
        log.info('Crowd detector started')

    def stop(self):
        self.is_running = False

    def uptime_seconds(self):
        return int(time.time() - self._start_time) if self._start_time else 0

    # ── Load ONNX model via cv2.dnn
    def _load_model(self):
        import os
        if not os.path.exists(MODEL_PATH):
            log.warning(f'Model not found at {MODEL_PATH}. Running in DEMO mode.')
            self._net = None
            return
        try:
            self._net = cv2.dnn.readNetFromONNX(MODEL_PATH)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            log.info('YOLOv8n ONNX model loaded via cv2.dnn')
        except Exception as e:
            log.error(f'Model load failed: {e}. Running in DEMO mode.')
            self._net = None

    # ── Main detection loop
    def _detect_loop(self):
        while self.is_running:
            t0 = time.time()
            try:
                frame = self.camera.get_raw_frame()
                if frame is not None:
                    count, boxes = self._run_inference(frame)
                    self._last_count = count
                    self._boxes = boxes
                    self._process_count(count, frame, boxes)
            except Exception as e:
                log.error(f'Detect loop error: {e}')

            elapsed = time.time() - t0
            sleep   = max(0, INFER_INTERVAL - elapsed)
            time.sleep(sleep)

    # ── YOLO inference
    def _run_inference(self, frame):
        if self._net is None:
            return self._demo_count(frame)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False
        )
        self._net.setInput(blob)
        outputs = self._net.forward()                   # shape: (1, 84, 8400)

        # YOLOv8 output: [cx,cy,w,h, class_scores×80]
        pred = outputs[0].T                             # (8400, 84)
        boxes, scores = [], []
        for row in pred:
            obj_scores = row[4:]
            cls_id     = int(np.argmax(obj_scores))
            score      = float(obj_scores[cls_id])
            if cls_id != PERSON_CLASS or score < CONF_THRESH:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = int((cx - bw/2) * w / INPUT_SIZE)
            y1 = int((cy - bh/2) * h / INPUT_SIZE)
            x2 = int((cx + bw/2) * w / INPUT_SIZE)
            y2 = int((cy + bh/2) * h / INPUT_SIZE)
            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(score)

        # NMS
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
        final_boxes = []
        if len(idxs) > 0:
            for i in (idxs.flatten() if isinstance(idxs, np.ndarray) else idxs):
                x, y, bw, bh = boxes[i]
                final_boxes.append((x, y, x+bw, y+bh, scores[i]))

        count = len(final_boxes)
        annotated = self._annotate(frame, final_boxes, count)
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        self.camera.set_annotated_frame(jpeg.tobytes())

        return count, final_boxes

    def _demo_count(self, frame):
        """Simulated count for demo / no-model mode."""
        import random
        base = 20 + 15 * np.sin(time.time() / 30)
        count = max(0, int(base + random.gauss(0, 3)))
        annotated = frame.copy()
        # Draw fake boxes
        boxes = []
        for i in range(min(count, 15)):
            x = (i * 45) % (frame.shape[1] - 50)
            y = 80 + (i * 20) % (frame.shape[0] - 120)
            boxes.append((x, y, x+30, y+70, 0.9))
        annotated = self._annotate(annotated, boxes, count)
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        self.camera.set_annotated_frame(jpeg.tobytes())
        return count, boxes

    # ── Annotate frame
    def _annotate(self, frame, boxes, count):
        out = frame.copy()
        color_map = {
            'low':  (50, 200, 50),
            'med':  (50, 180, 240),
            'high': (30, 30, 240),
        }
        level = crowd_level(count)
        color = color_map[level]

        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f'{conf:.0%}', (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Overlay HUD
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
        ts = datetime.now().strftime('%H:%M:%S')
        cv2.putText(out, f'SmartBus AI  |  Crowd: {count}  |  {ts}',
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        level_text = level.upper()
        col = color
        cv2.putText(out, level_text, (out.shape[1] - 80, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        return out

    # ── Process count: DB write + alerts + announcements
    def _process_count(self, count, frame, boxes):
        stop_id  = self.active_stop
        level    = crowd_level(count)
        ts       = datetime.now().isoformat()

        # Save to DB
        self.db.update_stop_crowd(stop_id, count, level, ts)

        # Emergency overcrowding
        if count >= CROWD_HIGH:
            self.db.add_alert({
                'type':    'emergency',
                'stop_id': stop_id,
                'message': f'Overcrowding at {stop_id}: {count} people detected',
                'count':   count,
                'ts':      ts
            })
            self.bus_mgr.auto_dispatch(stop_id, 'overcrowding')
            self.announce.announce_overcrowding(stop_id, count)

        # Update AI prediction
        self.ai.update_observation(stop_id, count, ts)


# ── Helper
def crowd_level(count: int) -> str:
    if count < CROWD_LOW:
        return 'low'
    elif count < CROWD_HIGH:
        return 'med'
    return 'high'
