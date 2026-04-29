# SmartBus-AI-Crowd-Detection-Demand-Management-System
# 🚌 SmartBus AI System

## 📌 Overview
SmartBus AI is an edge-based intelligent transport system that monitors crowd density at bus stops and helps in efficient bus dispatch.

---

## 📂 Project Structure

### 🌐 Online Mode (Real Detection)
- Folder: `online/`
- Uses:
  - Raspberry Pi Camera
  - TFLite AI model
- Detects real people in real-time

### 💻 Offline Mode (Demo System)
- Folder: `offline/`
- Uses:
  - Simulated data
- Used for UI demo and fallback

---

## 🧠 Technologies Used
- Python
- OpenCV
- Flask
- NumPy
- TFLite Runtime
- PiCamera2

---

## ⚙️ How to Run

### Online Mode
```bash
cd online
python3 detect_crowd_picam.py
cd offline
python3 app.py
