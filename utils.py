import numpy as np
from PIL import Image, ImageDraw
import os
import json
import cv2


def process_image(image, image_size, ort_session, yolo_labels, confidence_threshold):
    image = crop_resize(image, image_size)
    norm_image = image_normalise_reshape(image)

    onnxruntime_input = {ort_session.get_inputs()[0].name: norm_image}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    detections = onnxruntime_outputs[0]

    results = []
    if len(detections) > 0:
        for det in detections:
            if det[6] >= confidence_threshold:
                coordinates = det[1:5]  # xyxy
                label = yolo_labels[int(det[5])]
                confidence = det[6]
                results.append((coordinates, label, confidence))

    return results


def log_detection(filename, json_detections_path, detections):
    all_detections = {}
    for i, (coordinates, label, confidence) in enumerate(detections):
        detection = {
            "image": os.path.basename(filename),
            "timestamp": filename.split('-')[1],
            "label": label,
            "confidence": float(confidence),
            "bbox": [float(coord) for coord in coordinates]
        }
        all_detections[i] = detection

    json_name = os.path.basename(filename).split(".")[0]
    json_file = os.path.join(json_detections_path, f"{json_name}.json")
    with open(json_file, "a") as f:
        json.dump(all_detections, f)
        f.write("\n")


def crop_resize(image, new_size):
    height, width = image.shape[:2]
    min_dim = min(width, height)

    # Calculate cropping boundaries
    left = (width - min_dim) // 2
    upper = (height - min_dim) // 2
    right = left + min_dim
    lower = upper + min_dim

    # Crop the image
    cropped_image = image[upper:lower, left:right]
    resized_image = cv2.resize(cropped_image, (new_size, new_size))

    return resized_image


def image_normalise_reshape(image):
    # Move channel dimension to the front (assuming PyTorch format) and normalize pixel values by 255
    norm_image = image.transpose((2, 0, 1)) / 255.0

    # Expand the dimension at index 0 to create a batch dimension (assuming batch size of 1)
    # and cast the data type to float32 for compatibility with most models
    return np.expand_dims(norm_image, 0).astype(np.float32)


def draw_bounding_box(image, coordinates, box_color="red", box_width=3):
    draw = ImageDraw.Draw(image)
    draw.rectangle(coordinates, outline=box_color, width=box_width)
    return image


def read_txt_file(txt_file):
    with open(txt_file, 'r') as file:
        return [line.strip() for line in file]
