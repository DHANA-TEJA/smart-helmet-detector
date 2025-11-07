from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import os
import datetime
import threading
import queue
import time
from playsound import playsound

app = Flask(__name__)

# ---------------- CONFIG ----------------
MODEL_PATH = "helmet_detection_v11.pt"
ALERT_SOUND = "static/alert.mp3"
SNAPSHOT_DIR = "snapshots"
CONFIDENCE_THRESHOLD = 0.5
ALERT_COOLDOWN = 5  # seconds

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
print("Loading YOLOv11 model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ---------------- SOUND MANAGER (no echo) ----------------
_audio_queue = queue.Queue(maxsize=1)

def _audio_worker():
    """Worker that plays alert sound, ensures no overlap"""
    last_time = 0
    while True:
        _ = _audio_queue.get()
        now = time.time()
        if now - last_time >= ALERT_COOLDOWN:
            try:
                playsound(ALERT_SOUND)
            except Exception as e:
                print("Audio error:", e)
            last_time = now
        _audio_queue.task_done()

# start background sound thread
threading.Thread(target=_audio_worker, daemon=True).start()

def play_alert():
    """Add sound request safely"""
    if _audio_queue.empty():
        _audio_queue.put(1)

# ---------------- FRAME GENERATOR ----------------
def generate_frames():
    cap = cv2.VideoCapture(0)  # change to video file path if testing

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame, verbose=False)
            detections = results[0].boxes

            helmet_count = 0
            no_helmet_count = 0
            no_helmet_detected = False

            for box in detections:
                conf = float(box.conf)
                label = model.names[int(box.cls)]
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if "helmet" in label.lower() and "no" not in label.lower():
                    color = (0, 255, 0)
                    helmet_count += 1
                else:
                    color = (0, 0, 255)
                    no_helmet_count += 1
                    no_helmet_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Overlay counts box
            cv2.rectangle(frame, (0, 0), (360, 90), (255, 255, 255), -1)
            cv2.putText(frame, f"Helmet: {helmet_count}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)
            cv2.putText(frame, f"No Helmet: {no_helmet_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Play alert if violation detected
            if no_helmet_detected:
                play_alert()
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SNAPSHOT_DIR, f"no_helmet_{ts}.jpg")
                cv2.imwrite(filename, frame)

            # Encode frame for web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
