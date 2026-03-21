"""
DeepVision – AI Crowd Monitor (Cloud Version)
Features:
- Video upload mode (fully cloud-compatible)
- Live person count per frame
- Entered / Exited counts (crossing center line)
- Crowd alert UI when visible >= threshold
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
import math
from ultralytics import YOLO

st.set_page_config(
    page_title="DeepVision – AI Crowd Monitor",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = "yolov8n.pt"

# =========================
# Load YOLO model (cached)
# =========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}. Make sure '{MODEL_PATH}' is in the repository root.")
    st.stop()

# =========================
# Simple Centroid Tracker
# =========================
def match_detections_to_tracks(detections, tracks, max_dist=50.0):
    if tracks:
        next_id_start = max(tracks.keys()) + 1
    else:
        next_id_start = 1

    new_tracks = {}
    id_assignments = []
    used_ids = set()
    next_id = next_id_start

    for (cx, cy) in detections:
        best_id = None
        best_dist = float("inf")
        for tid, (tx, ty) in tracks.items():
            if tid in used_ids:
                continue
            d = math.dist((cx, cy), (tx, ty))
            if d < best_dist and d < max_dist:
                best_dist = d
                best_id = tid

        if best_id is None:
            best_id = next_id
            next_id += 1

        used_ids.add(best_id)
        new_tracks[best_id] = (cx, cy)
        id_assignments.append(best_id)

    return new_tracks, id_assignments, next_id

# =========================
# Process ONE FRAME
# =========================
def process_frame(frame, model, tracks, last_y, next_id_global, entered_total, exited_total, threshold):
    frame = cv2.resize(frame, (960, 540))
    h, w = frame.shape[:2]
    line_y = h // 2

    results = model(frame, classes=[0], verbose=False)

    detections = []
    rects = []

    if results is not None and len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                rects.append((x1, y1, x2, y2))
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append((cx, cy))

    tracks, assigned_ids, next_id_global = match_detections_to_tracks(detections, tracks, max_dist=50.0)
    visible_now = len(tracks)

    for (cx, cy), tid in zip(detections, assigned_ids):
        prev_y = last_y.get(tid, cy)
        if prev_y >= line_y and cy < line_y:
            entered_total += 1
        elif prev_y <= line_y and cy > line_y:
            exited_total += 1
        last_y[tid] = cy

    for (x1, y1, x2, y2), tid in zip(rects, assigned_ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.rectangle(frame, (0, 0), (340, 90), (5, 5, 5), -1)
    cv2.putText(frame, f"Visible: {visible_now}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Entered: {entered_total}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Exited: {exited_total}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

    alert_triggered = visible_now >= threshold
    if alert_triggered:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 3)

    return frame, visible_now, entered_total, exited_total, tracks, last_y, next_id_global, alert_triggered

# =========================
# Streamlit UI
# =========================
st.markdown("<h1 style='text-align:center;'>🧠 DeepVision – AI Crowd Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Real-time person detection & counting using YOLOv8</p>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.number_input("Alert Threshold (Visible People)", min_value=1, max_value=500, value=5, step=1)

alert_ph = st.empty()
col1, col2, col3 = st.columns(3)
visible_ph = col1.empty()
entered_ph = col2.empty()
exited_ph = col3.empty()
visible_ph.metric("Currently Visible", 0)
entered_ph.metric("Total Entered", 0)
exited_ph.metric("Total Exited", 0)
video_placeholder = st.empty()

uploaded_file = st.file_uploader("📂 Upload a video to analyze", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("▶ Start Processing", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmp_video_path = tmpfile.name

        cap = cv2.VideoCapture(tmp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        progress_bar = st.progress(0)

        tracks = {}
        last_y = {}
        next_id_global = 1
        entered_total = 0
        exited_total = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            frame_out, visible_now, entered_total, exited_total, tracks, last_y, next_id_global, alert_triggered = \
                process_frame(frame, model, tracks, last_y, next_id_global, entered_total, exited_total, threshold)

            rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, use_container_width=True)
            visible_ph.metric("Currently Visible", visible_now)
            entered_ph.metric("Total Entered", entered_total)
            exited_ph.metric("Total Exited", exited_total)

            if alert_triggered:
                alert_ph.error(f"🚨 ALERT! Visible Count ({visible_now}) exceeded Threshold ({threshold})!")
            else:
                alert_ph.empty()

            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        try:
            os.unlink(tmp_video_path)
        except:
            pass
        st.success("✅ Video processing complete!")
else:
    st.info("👆 Upload a video file above to get started.")
    st.markdown("""
    ### How it works:
    1. **Upload** any crowd or street video (mp4, avi, mov, mkv)
    2. **Click Start Processing** to run YOLOv8 person detection
    3. **View** real-time count of people visible, entered, and exited
    4. **Alert** triggers when crowd exceeds your threshold
    """)