import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
from detection_engine import DetectionEngine

# Page config
st.set_page_config(
    page_title="Urban Road Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Custom CSS for themes
def apply_theme(theme):
    if theme == 'dark':
        primary_color = "#1E90FF" # Blue
        bg_color = "#000000"      # Black
        secondary_bg = "#262730"  # Grey
        text_color = "#FFFFFF"
        button_bg = "#1E1E1E"
    else:
        primary_color = "#FF69B4" # Pink
        bg_color = "#FFFFFF"      # White
        secondary_bg = "#F0F2F6"  # Light Grey
        text_color = "#000000"
        button_bg = "#FFE4E1"     # Misty Rose

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        [data-testid="stSidebar"] {{
            background-color: {secondary_bg};
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background-color: {button_bg};
            color: {text_color};
            border: 1px solid {primary_color};
        }}
        .count-button-active {{
            background-color: {primary_color} !important;
            color: white !important;
            font-weight: bold;
        }}
        .count-button-inactive {{
            background-color: {button_bg};
            color: {text_color};
        }}
        </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# Sidebar
with st.sidebar:
    st.title("Settings")

    # Theme Toggle
    st.subheader("Theme")
    col1, col2 = st.columns(2)
    if col1.button("Dark Mode"):
        st.session_state.theme = 'dark'
        st.rerun()
    if col2.button("Light Mode"):
        st.session_state.theme = 'light'
        st.rerun()

    st.divider()

    # Video Source
    st.subheader("Video Source")
    source_type = st.radio("Select Source", ["Local Video", "IP Camera / Link"])

    video_file = None
    ip_link = ""

    if source_type == "Local Video":
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    else:
        ip_link = st.text_input("Enter IP/URL", placeholder="http://192.168.1.x:8080/video")
        st.info("Tip: Use 'IP Webcam' app on mobile for local LAN link.")

    st.divider()
    start_btn = st.button("Start Monitoring", type="primary")
    stop_btn = st.button("Stop Monitoring")

# Main Content
st.title("🚦 Urban Road Scenes: Real-Time Traffic Monitoring")

# Placeholder for counts
count_cols = st.columns(5)
placeholders = {}
classes = ['car', 'bus', 'truck', 'motorcycle', 'person']
labels = ['Cars', 'Buses', 'Trucks', 'Motorcycles', 'Pedestrians']

for i, cls in enumerate(classes):
    placeholders[cls] = count_cols[i].empty()

# Video Display
video_placeholder = st.empty()

# Detection Engine Initialization
@st.cache_resource
def get_engine():
    model_path = os.path.join(os.path.dirname(__file__), "../models/yolov8n.pt")
    return DetectionEngine(model_path)

engine = get_engine()

def update_count_buttons(counts):
    for i, cls in enumerate(classes):
        val = counts.get(cls, 0)
        btn_class = "count-button-active" if val > 0 else "count-button-inactive"
        # We can't easily change button style per-button in standard Streamlit without custom components
        # So we use markdown with HTML to simulate buttons
        color = "#1E90FF" if st.session_state.theme == 'dark' else "#FF69B4"
        bg = color if val > 0 else ("#1E1E1E" if st.session_state.theme == 'dark' else "#FFE4E1")
        txt_color = "white" if val > 0 else ("white" if st.session_state.theme == 'dark' else "black")

        placeholders[cls].markdown(f"""
            <div style="
                background-color: {bg};
                color: {txt_color};
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                border: 1px solid {color};
                font-weight: bold;
                margin-bottom: 10px;
            ">
                {labels[i]}: {val}
            </div>
        """, unsafe_allow_html=True)

# Initial counts
update_count_buttons({})

if start_btn:
    cap = None
    tfile_path = None
    if source_type == "Local Video" and video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile_path = tfile.name
        tfile.close()
        cap = cv2.VideoCapture(tfile_path)
    elif source_type == "IP Camera / Link" and ip_link:
        cap = cv2.VideoCapture(ip_link)
    else:
        # Fallback to sample if nothing provided but button clicked?
        # Or just show error
        st.error("Please provide a video source.")

    if cap is not None and cap.isOpened():
        st.success("Connection established. Processing...")

        # Add a stop flag in session state
        st.session_state.running = True

        while cap.isOpened() and st.session_state.get('running', False):
            ret, frame = cap.read()
            if not ret:
                st.info("End of video stream.")
                break

            # Process frame
            processed_frame, counts = engine.process_frame(frame)

            # Convert BGR to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Update UI
            video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
            update_count_buttons(counts)

            # Check for stop
            # Note: In Streamlit, it's hard to catch the stop button click inside a loop
            # unless we use some tricks. But let's try this.
            # Actually, we might need a small sleep to allow UI interactions?
            # time.sleep(0.01)

        cap.release()
        if tfile_path and os.path.exists(tfile_path):
            os.remove(tfile_path)
        st.session_state.running = False
    elif cap is not None:
        st.error("Could not open video source.")

if stop_btn:
    st.session_state.running = False
    st.write("Stopped.")
