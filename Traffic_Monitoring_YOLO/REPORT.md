# Project Report: Real-Time Traffic Monitoring and Object Detection

## 1. Project Title
**Real-Time Traffic Monitoring and Object Detection using YOLO on Urban Road Scenes**

## 2. Problem Statement
This project addresses the challenge of real-time object detection in complex traffic environments. Urban centers face increasing congestion, and traditional manual monitoring is inefficient and non-scalable. The goal is to automate the identification and localization of road entities—such as cars, buses, trucks, motorcycles, and pedestrians—to provide actionable data for traffic management.

**Target Beneficiaries:**
- Traffic management authorities
- Law enforcement agencies
- Urban planners
- Daily commuters

## 3. Motivation & Practical Relevance
Efficient traffic monitoring is a cornerstone of "Smart Cities." Automating this process improves road safety and optimizes traffic flow.
**Practical Impact:**
- Automated surveillance reducing manual labor.
- Improved traffic signal control based on real-time density.
- Early detection of accidents or violations.
- Data-driven urban planning decisions.

## 4. Literature Review (Summary)
1. **YOLO (You Only Look Once):** A single-stage detector chosen for this project due to its superior speed-to-accuracy ratio, essential for real-time applications.
2. **Faster R-CNN:** While highly accurate, its two-stage region proposal approach is computationally expensive and often too slow for real-time edge processing on CPUs.
3. **SSD (Single Shot MultiBox Detector):** Offers multi-scale detection but generally lacks the refinement and ecosystem support of modern YOLO variants (v8/v9/v10).

## 5. Proposed Methodology
The solution follows a systematic computer vision pipeline:
1. **Model Selection:** YOLOv8n (nano) was selected for its extremely low latency and high efficiency on CPU-bound hardware.
2. **Preprocessing:** Frames are extracted from video streams and resized to the model's native resolution (640x640) for optimal inference.
3. **Inference:** The pretrained model identifies entities across six critical classes: person, bicycle, car, motorcycle, bus, and truck.
4. **Visualization:** Bounding boxes and confidence scores are overlaid on the frames, and counts are extracted to update a live dashboard.
5. **GUI Implementation:** A modern desktop interface was developed using the Flet framework (based on Flutter) to provide a user-friendly experience.

## 6. Dataset Description
- **Primary Source:** The project is designed to be compatible with traffic datasets from Dhaka and other high-density urban environments.
- **Validation:** Testing was performed using standard urban traffic footage and the COCO (Common Objects in Context) dataset labels relevant to transportation.

## 7. Expected Outcomes
- **Functional Prototype:** A desktop application capable of real-time traffic analysis.
- **Entity Counting:** Automated tracking of vehicle and pedestrian density.
- **Performance:** Stable inference speeds on standard laptop hardware without requiring a dedicated GPU.
- **Visual Evidence:** Real-time annotated video output demonstrating the model's accuracy.

## 8. Conclusion
The "Real-Time Traffic Monitoring" system provides a scalable, cost-effective solution for urban traffic analysis. By leveraging YOLOv8 and modern GUI frameworks, it bridges the gap between complex computer vision models and practical, usable tools for traffic authorities.
