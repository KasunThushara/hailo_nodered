import sys
import os
import numpy as np
import cv2
import argparse
import threading
import time
from flask import Flask, Response
from hailo_inference import HailoInference
from object_detector import ObjectDetector  
from landmark_predictor import LandmarkPredictor  
from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS

# Global variables for Flask
output_frame = None
lock = threading.Lock()
stop_event = threading.Event()

app = Flask(__name__)

@app.route('/')
def index():
    return "MJPEG Streamer Running"

@app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame
        while not stop_event.is_set():
            with lock:
                if output_frame is None:
                    continue
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['GET'])
def stop():
    stop_event.set()
    time.sleep(1)
    return "Stopping stream...", 200

def run_flask():
    app.run(host='0.0.0.0', port=5001, threaded=True)

def main():
    global output_frame

    # Initialize Hailo inference
    hailo_infer = HailoInference()

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--detection', type=str, default="hand",  
                    choices=["hand", "face"],
                    help="Application (hand, face). Default is hand")
    ap.add_argument('-m', '--model1', type=str, 
                    help='Path of detection model')
    ap.add_argument('-n', '--model2', type=str, 
                    help='Path of landmark model')
    args = ap.parse_args()

    # Set up models based on application type
    if args.detection == "hand":
        detector_type = "palm"
        landmark_type = "hand"
        default_detector_model = 'models/palm_detection_lite.hef'
        default_landmark_model = 'models/hand_landmark_lite.hef'
    elif args.detection == "face":
        detector_type = "face"
        landmark_type = "face"
        default_detector_model = 'models/face_detection_short_range.hef'
        default_landmark_model = 'models/face_landmark.hef'
    else:
        print(f"[ERROR] Invalid application: {args.detection}. Must be one of hand,face.")
        exit(1)

    # Use default models if none specified
    args.model1 = args.model1 or default_detector_model
    args.model2 = args.model2 or default_landmark_model

    # Initialize detectors
    detector = ObjectDetector(detector_type, hailo_infer)
    detector.load_model(args.model1)

    landmark_predictor = LandmarkPredictor(landmark_type, hailo_infer)
    landmark_predictor.load_model(args.model2)

    # Start Flask app in another thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Open default camera
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print("Press Ctrl+C to quit")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Convert frame to RGB and process
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = frame.copy()

            # Detection pipeline
            img1, scale1, pad1 = detector.resize_pad(image)
            normalized_detections = detector.predict_on_image(img1)

            if len(normalized_detections) > 0:
                detections = detector.denormalize_detections(normalized_detections, scale1, pad1)
                xc, yc, scale, theta = detector.detection2roi(detections)
                roi_img, roi_affine, roi_box = landmark_predictor.extract_roi(image, xc, yc, theta, scale)
                flags, normalized_landmarks = landmark_predictor.predict(roi_img)
                landmarks = landmark_predictor.denormalize_landmarks(normalized_landmarks, roi_affine)

                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    if args.detection == "hand":
                        draw_landmarks(output, landmark[:, :2], HAND_CONNECTIONS, size=2)
                    elif args.detection == "face":
                        draw_landmarks(output, landmark[:, :2], FACE_CONNECTIONS, size=1)
                draw_roi(output, roi_box)
                draw_detections(output, detections)

            # Update the global output frame
            with lock:
                output_frame = output.copy()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    finally:
        cap.release()
        stop_event.set()
        print("Stream ended")

if __name__ == "__main__":
    main()
