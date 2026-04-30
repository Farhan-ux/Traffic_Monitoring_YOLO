import flet as ft
import cv2
import base64
import time
import os
import threading
from detection_engine import DetectionEngine

def main(page: ft.Page):
    page.title = "Real-Time Traffic Monitoring"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_width = 1200
    page.window_height = 800

    # Initialize detection engine
    # Adjust path if running from different locations
    model_path = os.path.join(os.path.dirname(__file__), "../models/yolov8n.pt")
    engine = DetectionEngine(model_path)

    # UI Elements
    video_image = ft.Image(
        src_base64="",
        width=800,
        height=450,
        fit=ft.ImageFit.CONTAIN,
    )

    status_text = ft.Text("Ready", size=18, color=ft.colors.BLUE_200)

    # Count Display Cards
    def create_count_card(label, icon):
        return ft.Container(
            content=ft.Column([
                ft.Icon(icon, size=30, color=ft.colors.AMBER_400),
                ft.Text(label, size=16, weight=ft.FontWeight.BOLD),
                ft.Text("0", size=24, key=f"count_{label.lower()}")
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.SURFACE_VARIANT,
            width=120
        )

    counts_row = ft.Row([
        create_count_card("Cars", ft.icons.DIRECTIONS_CAR),
        create_count_card("Buses", ft.icons.DIRECTIONS_BUS),
        create_count_card("Trucks", ft.icons.LOCAL_SHIPPING),
        create_count_card("Motorcycles", ft.icons.TWO_WHEELER),
        create_count_card("Pedestrians", ft.icons.PERSON),
    ], alignment=ft.MainAxisAlignment.CENTER)

    def update_counts(counts):
        page.get_control(f"count_cars").value = str(counts.get('car', 0))
        page.get_control(f"count_buses").value = str(counts.get('bus', 0))
        page.get_control(f"count_trucks").value = str(counts.get('truck', 0))
        page.get_control(f"count_motorcycles").value = str(counts.get('motorcycle', 0))
        page.get_control(f"count_pedestrians").value = str(counts.get('person', 0))
        page.update()

    is_processing = False

    def video_processing_thread():
        nonlocal is_processing

        # For this prototype, we use the sample data
        # Check for video, fallback to image if video is missing/broken
        video_path = os.path.join(os.path.dirname(__file__), "../data/sample_traffic.mp4")
        image_path = os.path.join(os.path.dirname(__file__), "../data/sample_traffic.jpg")

        if os.path.exists(video_path) and cv2.VideoCapture(video_path).isOpened():
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened() and is_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                process_and_update_ui(frame)
            cap.release()
        elif os.path.exists(image_path):
            frame = cv2.imread(image_path)
            if frame is not None:
                # In image mode, just process once or loop
                process_and_update_ui(frame)
                time.sleep(2) # Show for 2 seconds
        else:
            status_text.value = "Error: No valid media found"
            page.update()

        is_processing = False
        status_text.value = "Processing Complete"
        status_text.color = ft.colors.BLUE_200
        page.update()

    def process_and_update_ui(frame):
        # Resize for display
        display_frame = cv2.resize(frame, (800, 450))
        processed_frame, counts = engine.process_frame(display_frame)

        # Convert to base64 for Flet Image
        _, buffer = cv2.imencode(".jpg", processed_frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        video_image.src_base64 = img_base64
        update_counts(counts)

    def handle_process_video(e):
        nonlocal is_processing
        if is_processing:
            return

        is_processing = True
        status_text.value = "Processing..."
        status_text.color = ft.colors.GREEN_400
        page.update()

        # Run processing in a separate thread to keep UI responsive
        threading.Thread(target=video_processing_thread, daemon=True).start()

    def handle_stop(e):
        nonlocal is_processing
        is_processing = False
        status_text.value = "Stopped"
        status_text.color = ft.colors.RED_400
        page.update()

    # Layout
    page.add(
        ft.Column([
            ft.Text("Urban Road Scenes: Real-Time Traffic Monitoring", size=32, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Row([
                ft.ElevatedButton("Start Monitoring", icon=ft.icons.PLAY_ARROW, on_click=handle_process_video),
                ft.ElevatedButton("Stop", icon=ft.icons.STOP, on_click=handle_stop, color=ft.colors.RED),
                status_text
            ], alignment=ft.MainAxisAlignment.CENTER),
            ft.Container(
                content=video_image,
                border=ft.border.all(2, ft.colors.OUTLINE),
                border_radius=10,
                padding=5
            ),
            ft.Text("Live Object Detection Summary", size=24, weight=ft.FontWeight.W_500),
            counts_row
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )

if __name__ == "__main__":
    ft.app(target=main)
