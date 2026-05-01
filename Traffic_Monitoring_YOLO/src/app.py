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
if 'running' not in st.session_state:
    st.session_state.running = False

# Custom CSS for themes
def apply_theme(theme):
    if theme == 'dark':
        primary_color = "#1E90FF" # Blue
        bg_color = "#000000"      # Black
        secondary_bg = "#262730"  # Grey
        text_color = "#FFFFFF"
        button_bg = "#1E1E1E"
        sidebar_text = "#FFFFFF"
    else:
        primary_color = "#FF69B4" # Pink
        bg_color = "#FFFFFF"      # White
        secondary_bg = "#FFE4E1"  # Misty Rose / Light Pink
        text_color = "#000000"
        button_bg = "#FFF0F5"     # Lavender Blush
        sidebar_text = "#000000"

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        [data-testid="stSidebar"] {{
            background-color: {secondary_bg} !important;
        }}
        [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {{
            color: {sidebar_text} !important;
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background-color: {button_bg};
            color: {text_color};
            border: 1px solid {primary_color};
        }}
        </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme Toggle
    st.subheader("🎨 Theme")
    col1, col2 = st.columns(2)
    if col1.button("🌙 Dark Mode"):
        st.session_state.theme = 'dark'
        st.rerun()
    if col2.button("☀️ Light Mode"):
        st.session_state.theme = 'light'
        st.rerun()
    
    st.divider()
    
    # Video Source
    st.subheader("📹 Video Source")
    source_type = st.radio("Select Source", ["Local Video", "IP Camera / Link"])
    
    video_file = None
    ip_link = ""
    
    if source_type == "Local Video":
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    else:
        ip_link = st.text_input("Enter IP/URL", placeholder="http://192.168.1.x:8080/video")
        st.info("Tip: Use 'IP Webcam' app on mobile for local LAN link.")

    st.divider()
    if not st.session_state.running:
        if st.button("🚀 Start Monitoring", type="primary"):
            st.session_state.running = True
            st.rerun()
    else:
        if st.button("🛑 Stop Monitoring", type="secondary"):
            st.session_state.running = False
            st.rerun()

    if st.button("🔄 Reset Totals"):
        if 'engine' in st.session_state:
            st.session_state.engine.reset_cumulative()
        st.rerun()

# Main Content
st.title("🚦 Urban Road Scenes: Real-Time Traffic Monitoring")

# Detection Engine Initialization
if 'engine' not in st.session_state:
    model_path = os.path.join(os.path.dirname(__file__), "../models/yolov8n.pt")
    st.session_state.engine = DetectionEngine(model_path)

engine = st.session_state.engine

# Layout
col_vid, col_stats = st.columns([2, 1])

with col_vid:
    video_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Session Statistics")
    # Categories and labels
    classes = ['car', 'bus', 'truck', 'motorcycle', 'person', 'train', 'bicycle', 'traffic light', 'stop sign']
    labels = ['Cars', 'Buses', 'Trucks', 'Motorcycles', 'Pedestrians', 'Trains', 'Bicycles', 'Traffic Lights', 'Stop Signs']
    
    placeholders = {}
    for i, cls in enumerate(classes):
        placeholders[cls] = st.empty()

def update_ui(counts, cumulative):
    color = "#1E90FF" if st.session_state.theme == 'dark' else "#FF69B4"
    bg_inactive = "#1E1E1E" if st.session_state.theme == 'dark' else "#FFF0F5"
    text_inactive = "white" if st.session_state.theme == 'dark' else "black"
    
    for i, cls in enumerate(classes):
        val = counts.get(cls, 0)
        total = cumulative.get(cls, 0)
        
        # Highlight if currently detected
        bg = color if val > 0 else bg_inactive
        txt_color = "white" if val > 0 else text_inactive
        
        placeholders[cls].markdown(f"""
            <div style="
                background-color: {bg};
                color: {txt_color};
                padding: 10px;
                border-radius: 10px;
                text-align: left;
                border: 1px solid {color};
                font-weight: bold;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
            ">
                <span>{labels[i]}</span>
                <span>Live: {val} | Total: {total}</span>
            </div>
        """, unsafe_allow_html=True)

# Initial UI state
update_ui({}, engine.cumulative_counts)

if st.session_state.running:
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
    elif source_type == "Local Video" and video_file is None:
        st.warning("Please upload a video file first.")
        st.session_state.running = False
        st.rerun()
        
    if cap is not None and cap.isOpened():
        st.toast("Connection established. Monitoring started!")
        try:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, counts, cumulative = engine.process_frame(frame)
                
                # Convert BGR to RGB
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update UI
                video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                update_ui(counts, cumulative)
        finally:
            cap.release()
            if tfile_path and os.path.exists(tfile_path):
                os.remove(tfile_path)
        
        if st.session_state.running:
            st.session_state.running = False
            st.rerun()
    elif cap is not None:
        st.error("Could not open video source.")
        st.session_state.running = False
