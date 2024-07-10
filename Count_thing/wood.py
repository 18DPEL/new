import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
import numpy as np

class ObjectDetector_wood:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def predict(self, image_path):
        # Read the image using OpenCV
        original_img = cv2.imread(image_path)
        img = cv2.resize(original_img, (400, 300))

        # Perform detection
        result = self.model(img, conf=self.confidence_threshold)[0]

        # Extract detections
        detections = sv.Detections.from_ultralytics(result)

        # Create a DotAnnotator and annotate the image
        dot_annotator = sv.DotAnnotator()
        annotated_img = dot_annotator.annotate(scene=img.copy(), detections=detections)

        # Get the number of detections
        num_detections = len(detections)

        return original_img, annotated_img, num_detections

