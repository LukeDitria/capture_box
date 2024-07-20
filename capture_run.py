#!/usr/bin/env python
# coding: utf-8

import onnxruntime
import time
import utils
import os
import argparse
import shutil
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO object detection on periodic image scans")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to scan for new images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to move processed images")
    parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds (default: 300)")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold (default: 0.5)")
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

    json_detections_path = os.path.join(args.output_dir, "detections")
    if not os.path.exists(json_detections_path):
        os.mkdir(json_detections_path)

    image_detections_path = os.path.join(args.output_dir, "images")
    if not os.path.exists(image_detections_path):
        os.mkdir(image_detections_path)

    while True:
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(args.input_dir, filename)
                coordinates, label, confidence = utils.process_image(image_detections_path, args.image_size,
                                                                     ort_session, yolo_labels, args.confidence)

                if coordinates is not None:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    new_filename = f"{label}-{timestamp}-1.jpg"
                    new_path = os.path.join(json_detections_path, new_filename)
                    shutil.move(image_path, new_path)
                    utils.log_detection(new_path, label, confidence, coordinates, timestamp)
                    print(f"Detected {label} with confidence {confidence:.2f}")
                else:
                    print(f"No detection above threshold for {filename}")

                # Delete the original image
                os.remove(image_path)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()