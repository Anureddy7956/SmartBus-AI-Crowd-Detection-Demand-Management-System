"""
SmartBus AI — Main Flask Application
Raspberry Pi 4B optimized · Lightweight · Offline-first
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import threading
import time
import json
import logging
from datetime import datetime

from camera import CameraManager
from crowd_detector import CrowdDetector
from ai_engine import AIEngine
from announcement_engine import AnnouncementEngine
from database import Database
from bus_manager import BusManager

# ── Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/smartbus.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('SmartBus')

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'smartbus-pi-secret'

# ── Global singletons (lazy-init to save RAM)
db = Database()
ai = AIEngine()
announce = AnnouncementEngine()
bus_mgr = BusManager(db)
camera = CameraManager()
detector = CrowdDetector(camera, db, ai, announce, bus_mgr)

# ─────────────────────────────────────────────
# REST API ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ── Live dashboard data (polling every 2s)
@app.route('/api/live')
def api_live():
    """Returns all live data for dashboard refresh."""
    try:
        data = db.get_live_snapshot()
        return jsonify({'ok': True, 'data': data})
    except Exception as e:
        log.error(f'api_live error: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500

# ── Per-stop detailed data
@app.route('/api/stop/<stop_id>')
def api_stop(stop_id):
    try:
        stop = db.get_stop_detail(stop_id)
        prediction = ai.predict_crowd(stop_id)
        seats = ai.estimate_seats(stop_id)
        return jsonify({'ok': True, 'stop': stop, 'prediction': prediction, 'seats': seats})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# ── AI dispatch suggestion
@app.route('/api/dispatch', methods=['POST'])
def api_dispatch():
    data = request.get_json()
    stop_id = data.get('stop_id')
    reason = data.get('reason', 'manual')
    result = bus_mgr.dispatch_bus(stop_id, reason)
    announce.announce_dispatch(stop_id, result)
    db.log_dispatch(stop_id, result)
    return jsonify({'ok': True, 'result': result})

# ── Passenger signal (bus request)
@app.route('/api/signal', methods=['POST'])
def api_signal():
    data = request.get_json()
    stop_id = data.get('stop_id')
    db.log_passenger_signal(stop_id)
    count = db.get_signal_count(stop_id)
    bus_mgr.check_signal_threshold(stop_id, count)
    announce.announce_signal_received(stop_id)
    return jsonify({'ok': True, 'signal_count': count})

# ── Force announcement
@app.route('/api/announce', methods=['POST'])
def api_announce():
    data = request.get_json()
    text = data.get('text', '')
    lang = data.get('lang', 'en')
    announce.speak(text, lang)
    return jsonify({'ok': True})

# ── MJPEG video stream endpoint
@app.route('/api/stream')
def video_stream():
    """Streams annotated camera frames as MJPEG."""
    def gen():
        while True:
            frame = camera.get_annotated_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # ~10 fps — lightweight
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── Camera snapshot
@app.route('/api/snapshot')
def snapshot():
    frame = camera.get_annotated_frame()
    if frame:
        return Response(frame, mimetype='image/jpeg')
    return jsonify({'ok': False, 'error': 'No frame'}), 503

# ── Heatmap history
@app.route('/api/heatmap')
def api_heatmap():
    data = db.get_heatmap_data()
    return jsonify({'ok': True, 'data': data})

# ── Fleet status
@app.route('/api/fleet')
def api_fleet():
    return jsonify({'ok': True, 'buses': bus_mgr.get_fleet_status()})

# ── Alerts
@app.route('/api/alerts')
def api_alerts():
    return jsonify({'ok': True, 'alerts': db.get_active_alerts()})

# ── Stats summary
@app.route('/api/stats')
def api_stats():
    return jsonify({'ok': True, 'stats': db.get_stats_summary()})

# ── Health check
@app.route('/api/health')
def health():
    return jsonify({
        'ok': True,
        'camera': camera.is_running,
        'detector': detector.is_running,
        'uptime': detector.uptime_seconds(),
        'time': datetime.now().isoformat()
    })

# ── System info
@app.route('/api/sysinfo')
def sysinfo():
    import psutil, os
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    temp = 0
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = int(f.read().strip()) / 1000
    except:
        pass
    return jsonify({
        'cpu': cpu,
        'ram_used_mb': round(mem.used / 1024 / 1024),
        'ram_total_mb': round(mem.total / 1024 / 1024),
        'temp_c': round(temp, 1)
    })

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

def startup():
    log.info('=' * 50)
    log.info('SmartBus AI starting on Raspberry Pi 4B...')
    db.init()
    camera.start()
    detector.start()
    announce.speak("SmartBus system online", "en")
    log.info('All systems ready. Dashboard at http://0.0.0.0:5000')

if __name__ == '__main__':
    startup()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
