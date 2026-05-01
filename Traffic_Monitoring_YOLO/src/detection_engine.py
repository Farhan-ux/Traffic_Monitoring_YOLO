import cv2
from ultralytics import YOLO
import numpy as np

class DetectionEngine:
    def __init__(self, model_path='models/yolov8n.pt'):
        self.model = YOLO(model_path)
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 6: train, 7: truck, 9: traffic light, 11: stop sign
        self.target_classes = [0, 1, 2, 3, 5, 6, 7, 9, 11]
        self.class_names = self.model.names
        self.tracked_ids = set() # Set of (class_name, track_id)
        self.cumulative_counts = {name: 0 for name in [self.class_names[i] for i in self.target_classes]}

    def reset_cumulative(self):
        self.tracked_ids = set()
        for key in self.cumulative_counts:
            self.cumulative_counts[key] = 0

    def process_frame(self, frame, persist=True):
        # Run YOLOv8 tracking
        # persist=True keeps tracks across calls
        results = self.model.track(frame, verbose=False, classes=self.target_classes, persist=persist)[0]

        counts = {name: 0 for name in [self.class_names[i] for i in self.target_classes]}
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)

            # Annotate frame
            for box, cls_idx, score, track_id in zip(boxes, classes, scores, ids):
                name = self.class_names[int(cls_idx)]
                if name in counts:
                    counts[name] += 1
                
                # Update cumulative
                track_key = (name, track_id)
                if track_key not in self.tracked_ids:
                    self.tracked_ids.add(track_key)
                    self.cumulative_counts[name] += 1

                x1, y1, x2, y2 = map(int, box)
                label = f"{name} {track_id} {score:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Fallback to normal detection if tracking fails or no objects
            results_det = self.model(frame, verbose=False, classes=self.target_classes)[0]
            boxes = results_det.boxes.xyxy.cpu().numpy()
            classes = results_det.boxes.cls.cpu().numpy()
            scores = results_det.boxes.conf.cpu().numpy()
            
            for box, cls_idx, score in zip(boxes, classes, scores):
                name = self.class_names[int(cls_idx)]
                if name in counts:
                    counts[name] += 1
                
                x1, y1, x2, y2 = map(int, box)
                label = f"{name} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, counts, self.cumulative_counts

if __name__ == "__main__":
    # Test script
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models/yolov8n.pt')
    img_path = os.path.join(base_dir, 'data/sample_traffic.jpg')
    
    engine = DetectionEngine(model_path)
    img = cv2.imread(img_path)
    if img is not None:
        processed_frame, counts, cumulative = engine.process_frame(img, persist=False)
        print(f"Detected counts: {counts}")
        print(f"Cumulative counts: {cumulative}")
    else:
        print(f"Test image not found at {img_path}")
