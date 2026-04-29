#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
import time
import requests
import json

# Configuration
CAMERA_ID = "bus_stop_01"
THRESHOLD_ALERT = 5
FIREBASE_URL = "https://crowd-detection-d2c61-default-rtdb.asia-southeast1.firebasedatabase.app/"  # UPDATE THIS!
CONFIDENCE_THRESHOLD = 0.5

class CrowdDetector:
    def __init__(self, model_path='detect.tflite'):
        print("🔧 Loading TFLite model...")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        print(f"✅ Model loaded! Input size: {self.width}x{self.height}")
    
    def detect_people(self, frame):
        """Detect people in frame"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.width, self.height))
        input_data = np.expand_dims(img_resized, axis=0)
        
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        person_count = 0
        detections = []
        
        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD and int(classes[i]) == 0:
                person_count += 1
                ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
                xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
                ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
                xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))
                
                detections.append({
                    'box': (xmin, ymin, xmax, ymax),
                    'confidence': float(scores[i])
                })
        
        return person_count, detections

def send_to_firebase(data):
    """Send data to Firebase"""
    if "your-project" in FIREBASE_URL:
        print("⚠️  Firebase URL not configured! Skipping upload...")
        return False
    
    try:
        response = requests.put(
            f"{FIREBASE_URL}/bus_stops/{CAMERA_ID}.json",
            data=json.dumps(data),
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️  Firebase error: {e}")
        return False

def save_to_local_file(data):
    """Fallback: Save to local JSON file"""
    try:
        with open('/tmp/bus_data.json', 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"⚠️  Local save error: {e}")
        return False

def main():
    print("🚌 Initializing Bus Stop Monitoring System...")
    
    # Initialize Picamera2
    print("📷 Starting camera...")
    picam = Picamera2()
    
    config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    
    print("⏳ Camera warming up (2s)...")
    time.sleep(2)
    
    # Initialize detector
    detector = CrowdDetector()
    
    print("✅ System ready! Starting detection...\n")
    print("=" * 50)
    
    frame_count = 0
    last_upload_time = 0
    
    try:
        while True:
            # Capture frame
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect people
            start_time = time.time()
            person_count, detections = detector.detect_people(frame)
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_text = f"{det['confidence']:.2f}"
                cv2.putText(frame, conf_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add overlay
            cv2.putText(frame, f"People: {person_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Upload every 3 seconds
            current_time = time.time()
            if current_time - last_upload_time >= 3:
                data = {
                    'count': person_count,
                    'timestamp': int(current_time * 1000),
                    'alert': person_count > THRESHOLD_ALERT,
                    'status': 'crowded' if person_count > THRESHOLD_ALERT else 'normal',
                    'fps': round(fps, 1)
                }
                
                # Try Firebase first, fallback to local
                success = send_to_firebase(data)
                if not success:
                    save_to_local_file(data)
                
                status_icon = "🚨" if data['alert'] else "✅"
                print(f"{status_icon} Count: {person_count:2d} | Status: {data['status']:8s} | FPS: {fps:4.1f}")
                
                last_upload_time = current_time
            
            # Save preview every 30 frames
            if frame_count % 30 == 0:
                cv2.imwrite(f'/tmp/preview.jpg', frame)
            
            frame_count += 1
            time.sleep(0.05)  # Small delay
            
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Stopping monitoring system...")
    finally:
        picam.stop()
        print("✅ Camera stopped. Goodbye!")

if __name__ == "__main__":
    main()
