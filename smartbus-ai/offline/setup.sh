#!/bin/bash
# SmartBus AI — Raspberry Pi 4B Setup Script
# Run: chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────

set -e
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  SmartBus AI — Pi 4B Setup Script   ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. System packages
echo ">>> Installing system packages..."
sudo apt update -qq
sudo apt install -y \
  python3-pip \
  python3-picamera2 \
  espeak-ng \
  espeak-ng-data \
  libespeak-ng1 \
  python3-opencv \
  libatlas-base-dev \
  libopenblas-dev \
  --no-install-recommends

# ── 2. Python packages (break-system-packages for Pi OS)
echo ">>> Installing Python packages..."
pip3 install --break-system-packages \
  flask==3.0.3 \
  flask-cors==4.0.0 \
  opencv-python-headless==4.9.0.80 \
  numpy==1.26.4 \
  pyttsx3==2.90 \
  psutil==5.9.8

# ── 3. YOLO model (export once then delete ultralytics)
echo ">>> Setting up YOLOv8n model..."
mkdir -p models

if [ ! -f models/yolov8n.onnx ]; then
  echo ">>> Downloading + exporting YOLOv8n to ONNX (runs once)..."
  pip3 install --break-system-packages ultralytics
  python3 -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=320, simplify=True, dynamic=False)
shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
print('Model exported to models/yolov8n.onnx')
"
  echo ">>> Removing ultralytics + PyTorch to free ~2GB RAM..."
  pip3 uninstall -y ultralytics torch torchvision torchaudio 2>/dev/null || true
  echo ">>> YOLO model ready at models/yolov8n.onnx"
else
  echo ">>> models/yolov8n.onnx already exists, skipping."
fi

# ── 4. Logs directory
mkdir -p logs

# ── 5. Run!
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  Setup complete! Starting SmartBus  ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Dashboard will be at: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
python3 app.py
