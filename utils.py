import numpy as np
from PIL import Image, ImageDraw
import os
import json


def process_image(image_path, image_size, ort_session, yolo_labels, confidence_threshold):
    test_image = crop_resize(Image.open(image_path), image_size)
    np_image = np.array(test_image)
    norm_image = image_normalise_reshape(np_image)

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
    for coordinates, label, confidence in detections:
        detection = {
            "image": os.path.basename(filename),
            "timestamp": filename.split('-')[1],
            "label": label,
            "confidence": float(confidence),
            "bbox": [float(coord) for coord in coordinates]
        }

        json_name = os.path.basename(filename).split(".")[0]
        json_file = os.path.join(json_detections_path, f"{json_name}.json")
        with open(json_file, "a") as f:
            json.dump(detection, f)
            f.write("\n")


def crop_resize(image, new_size):
    width, height = image.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    upper = (height - min_dim) // 2
    right = left + min_dim
    lower = upper + min_dim

    square_image = image.crop((left, upper, right, lower))

    return square_image.resize((new_size, new_size))


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