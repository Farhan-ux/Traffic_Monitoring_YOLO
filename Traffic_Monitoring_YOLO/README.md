# Real-Time Traffic Monitoring and Object Detection using YOLO

A modern, high-performance desktop application for monitoring urban road scenes and detecting various traffic entities in real-time using YOLOv8.

## Features
- **Real-Time Detection:** High-speed object detection for cars, buses, trucks, motorcycles, and pedestrians.
- **Modern GUI:** Built with Flet for a sleek, responsive desktop experience.
- **Live Counting:** Real-time dashboard showing the count of each detected entity.
- **CPU Optimized:** Uses the YOLOv8-nano model, making it suitable for laptops without dedicated GPUs.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Farhan-ux/Traffic_Monitoring_YOLO.git
   cd Traffic_Monitoring_YOLO
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   ```bash
   python src/app.py
   ```

2. **Interface:**
   - Click **"Start Monitoring"** to begin processing the default traffic video.
   - Use the dashboard to monitor live vehicle and pedestrian counts.
   - Click **"Stop"** to halt processing.

## Project Structure
- `src/`: Source code for the application and detection engine.
- `models/`: Contains the YOLOv8 weights.
- `data/`: Sample video and test data.
- `assets/`: UI assets and styling.

## Methodology
The project leverages the YOLO (You Only Look Once) architecture for single-stage object detection. It is designed for intelligent transportation systems to assist in urban planning and automated traffic management.
