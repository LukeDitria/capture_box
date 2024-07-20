#!/usr/bin/env python
# coding: utf-8

import onnxruntime
import time
import utils
import os
import argparse
import shutil


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
        file_list = os.listdir(args.input_dir)
        if len(file_list) > 0:
            for filename in file_list:
                if filename.lower().endswith('.jpg'):
                    image_path = os.path.join(args.input_dir, filename)
                    detections = utils.process_image(image_path, args.image_size,
                                                     ort_session, yolo_labels, args.confidence)

                    if detections:
                        # Keep the original filename
                        new_path = os.path.join(image_detections_path, filename)
                        shutil.move(image_path, new_path)

                        utils.log_detection(filename, json_detections_path, detections)
                        print(f"Detected {len(detections)} objects in {filename}")
                        for _, label, confidence in detections:
                            print(f"- {label} with confidence {confidence:.2f}")
                    else:
                        print(f"No detection above threshold for {filename}")
                        # Delete the original image
                        try:
                            os.remove(image_path)
                            print("Delete image")
                        except FileNotFoundError:
                            print(f"File {image_path} was not found when trying to remove it.")
                        except PermissionError:
                            print(f"Permission denied when trying to remove {image_path}.")
        else:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()