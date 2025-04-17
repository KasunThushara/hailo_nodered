#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading 
import cv2
from typing import List
from object_detection_utils import ObjectDetectionUtils
import socket
import pickle
import time
from flask import Flask, Response

# Global variables for frame streaming and stop event
output_frame = None
lock = threading.Lock()
stop_event = threading.Event()  # Global stop event

app = Flask(__name__)

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches

CAMERA_CAP_WIDTH = 640
CAMERA_CAP_HEIGHT = 480

# Socket setup
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
server_ip = "127.0.0.1"  # Replace with server IP
server_port = 9999
address = (server_ip, server_port)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument("-n", "--net", help="Path for the network in HEF format.", default="yolov7.hef")
    parser.add_argument("-i", "--input", default="zidane.jpg", help="Path to the input - either an image or a folder of images.")
    parser.add_argument("-b", "--batch_size", default=1, type=int, required=False, help="Number of images in one batch")
    parser.add_argument("-l", "--labels", default="coco.txt", help="Path to labels file.")
    parser.add_argument("-s", "--save_stream_output", action="store_true", help="Save stream output.")
    args = parser.parse_args()

    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    return args

def preprocess_from_cap(cap: cv2.VideoCapture, batch_size: int, input_queue: queue.Queue, 
                       width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """Process frames from camera and add to input queue."""
    frames = []
    processed_frames = []
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or stop_event.is_set():
            break
        
        frames.append(frame)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = utils.preprocess(processed_frame, width, height)
        processed_frames.append(processed_frame)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            processed_frames, frames = [], []

    # Add any remaining frames
    if frames:
        input_queue.put((frames, processed_frames))

def preprocess_images(images: List[np.ndarray], batch_size: int, input_queue: queue.Queue,
                     width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """Process images and add to input queue."""
    for batch in divide_list_to_batches(images, batch_size):
        if stop_event.is_set():
            break
        input_tuple = ([image for image in batch], [utils.preprocess(image, width, height) for image in batch])
        input_queue.put(input_tuple)

def preprocess(images: List[np.ndarray], cap: cv2.VideoCapture, batch_size: int,
              input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """Main preprocessing function."""
    if cap is None:
        preprocess_images(images, batch_size, input_queue, width, height, utils)
    else:
        preprocess_from_cap(cap, batch_size, input_queue, width, height, utils)
    
    input_queue.put(None)  # Sentinel value

def postprocess(output_queue: queue.Queue, cap: cv2.VideoCapture, utils: ObjectDetectionUtils) -> None:
    """Postprocess inference results."""
    prev_time = 0
    while not stop_event.is_set():
        try:
            result = output_queue.get(timeout=1)
            if result is None:
                break

            original_frame, infer_results = result
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            if len(infer_results) == 1:
                infer_results = infer_results[0]

            detections = utils.extract_detections(infer_results)
            frame_with_detections = utils.draw_detections(detections, original_frame)
            
            cv2.putText(frame_with_detections, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Send frame via socket
            ret, buffer = cv2.imencode(".jpg", frame_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            client_socket.sendto(pickle.dumps(buffer), address)
            
            # Update MJPEG stream
            global output_frame
            with lock:
                output_frame = frame_with_detections.copy()

            output_queue.task_done()
        except queue.Empty:
            if stop_event.is_set():
                break

def infer(input_source: str, net_path: str, labels_path: str, batch_size: int) -> None:
    """Main inference pipeline."""
    utils = ObjectDetectionUtils(labels_path)
    cap = None
    images = []
    
    if input_source == "camera":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
    elif any(input_source.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
        cap = cv2.VideoCapture(input_source)
    else:
        images = load_images_opencv(input_source)
        validate_images(images, batch_size)

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, batch_size, input_queue, width, height, utils))
    postprocess_thread = threading.Thread(
        target=postprocess,
        args=(output_queue, cap, utils))

    preprocess_thread.start()
    postprocess_thread.start()
    hailo_inference.run()

    preprocess_thread.join()
    output_queue.put(None)  # Signal postprocess to exit
    postprocess_thread.join()

    if cap is not None:
        cap.release()
    logger.info('Inference stopped gracefully')
    sys.exit(0)

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
                (flag, encoded) = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['GET'])
def stop():
    """Trigger graceful shutdown"""
    stop_event.set()
    time.sleep(1)  # Allow final frames to process
    return "Stopping stream...", 200

def run_flask():
    app.run(host='0.0.0.0', port=5001, threaded=True)

def main() -> None:
    """Main function"""
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    args = parse_args()
    infer(args.input, args.net, args.labels, args.batch_size)

if __name__ == "__main__":
    main()
