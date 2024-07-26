#!/usr/bin/env python
# coding: utf-8

import onnxruntime
import time
import utils
import os
import argparse
import cv2
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO object detection on webcam stream")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detection results")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--image_size", type=int, default=640, help="Size to resize image")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Read in list of yolo class labels
    with open("yolo_labels.txt", 'r') as file:
        yolo_labels = [line.strip() for line in file]

    # Create an ONNX Runtime inference session with GPU support
    ort_session = onnxruntime.InferenceSession("./yolov7-tiny.onnx",
                                               providers=['CUDAExecutionProvider'])
    if not os.path.exists(args.input_dir):
        os.mkdir(args.input_dir)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    json_detections_path = os.path.join(args.output_dir, "detections")
    if not os.path.exists(json_detections_path):
        os.makedirs(json_detections_path)

    image_detections_path = os.path.join(args.output_dir, "images")
    if not os.path.exists(image_detections_path):
        os.makedirs(image_detections_path)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Generate a filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"webcam-{timestamp}.jpg"

        # Resize and process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = utils.process_image(image, args.image_size, ort_session, yolo_labels, args.confidence)

        if detections:
            # Save the frame with detections
            image_path = os.path.join(image_detections_path, filename)
            cv2.imwrite(image_path, frame)

            utils.log_detection(filename, json_detections_path, detections)
            print(f"Detected {len(detections)} objects in {filename}")
            for _, label, confidence in detections:
                print(f"- {label} with confidence {confidence:.2f}")
        else:
            print(f"No detection above threshold for {filename}")

        cv2.imshow('Raw Camera Feed', frame)

        # Wait for 1 second (1 FPS)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()