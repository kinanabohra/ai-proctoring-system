from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Track last known face center
prev_center = None

def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

def is_moving(prev, curr, threshold=20):
    if prev is None or curr is None:
        return False
    dx = abs(curr[0] - prev[0])
    dy = abs(curr[1] - prev[1])
    return dx > threshold or dy > threshold

@app.route('/detect', methods=['POST'])
def detect_face_movement():
    global prev_center
    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        prev_center = None
        return jsonify({"status": "Candidate not in frame", "faces_detected": 0})

    # Choose largest face
    (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
    center = get_center(x, y, w, h)

    if is_moving(prev_center, center):
        status = "Suspicious movement"
    else:
        status = "Stable"

    prev_center = center

    return jsonify({
        "status": status,
        "faces_detected": len(faces)
    })

if __name__ == '__main__':
    app.run(debug=True)
