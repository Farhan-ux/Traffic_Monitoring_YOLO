import cv2
from ultralytics import YOLO
import numpy as np

class DetectionEngine:
    def __init__(self, model_path='models/yolov8n.pt'):
        self.model = YOLO(model_path)
        # Class names in COCO that are relevant to traffic
        self.target_classes = [0, 1, 2, 3, 5, 7] # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.class_names = self.model.names

    def process_frame(self, frame):
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False, classes=self.target_classes)[0]

        # Get detections
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        counts = {
            'person': 0,
            'bicycle': 0,
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0
        }

        # Annotate frame
        for box, cls_idx, score in zip(boxes, classes, scores):
            name = self.class_names[int(cls_idx)]
            if name in counts:
                counts[name] += 1

            x1, y1, x2, y2 = map(int, box)
            label = f"{name} {score:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, counts

    def process_video_stream(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            processed_frame, counts = self.process_frame(frame)
            yield processed_frame, counts

        cap.release()

if __name__ == "__main__":
    # Test script
    engine = DetectionEngine('Traffic_Monitoring_YOLO/models/yolov8n.pt')
    img = cv2.imread('Traffic_Monitoring_YOLO/data/test_bus.jpg')
    if img is not None:
        processed_frame, counts = engine.process_frame(img)
        print(counts)
    else:
        print("Test image not found")
