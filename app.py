"""
DeepVision - LOCAL Crowd Monitor (Webcam + Video)
Features (LOCAL):
- Live Webcam mode
- Video Upload mode
- Live people count (visible)
- Entered / Exited counts (crossing center line)
- Alarm sound when visible >= threshold
- Email alert when visible >= threshold
- UI Alert in Main Screen
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
import math
import base64
from ultralytics import YOLO

# NEW: email imports
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

MODEL_PATH = "yolov8n.pt"
ALERT_SOUND_PATH = "alert_sound.mp3"

st.set_page_config(
    page_title="DeepVision Local - Webcam + Video",
    layout="wide",
)

# =========================
# Load YOLO model (cached)
# =========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure '{MODEL_PATH}' is in the directory.")
    st.stop()

# =========================
# Email sending (async)
# =========================
def send_email_async(sender, app_password, receiver, subject, body):
    """Send email in a background thread so UI doesn't freeze."""
    if not (sender and app_password and receiver):
        return  # missing details → skip

    def _send():
        try:
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = receiver
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender, app_password)
                server.send_message(msg)
            print("✅ Email alert sent.")
        except Exception as e:
            print("Email failed:", e)

    th = threading.Thread(target=_send, daemon=True)
    th.start()

# =========================
# Play alert sound
# =========================
def play_alert_sound():
    if not os.path.exists(ALERT_SOUND_PATH):
        return
    try:
        with open(ALERT_SOUND_PATH, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        st.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio tag.
            </audio>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        pass

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
def process_frame(
    frame,
    model,
    tracks,
    last_y,
    next_id_global,
    entered_total,
    exited_total,
    threshold,
    enable_sound,
    enable_email,
    email_sender,
    email_app_password,
    email_receiver,
    last_alert_time,
    cooldown_seconds,
):
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

    tracks, assigned_ids, next_id_global = match_detections_to_tracks(
        detections, tracks, max_dist=50.0
    )
    visible_now = len(tracks)

    # Count Entered / Exited
    for (cx, cy), tid in zip(detections, assigned_ids):
        prev_y = last_y.get(tid, cy)
        if prev_y >= line_y and cy < line_y:
            entered_total += 1
        elif prev_y <= line_y and cy > line_y:
            exited_total += 1
        last_y[tid] = cy

    # Draw boxes + IDs
    for (x1, y1, x2, y2), tid in zip(rects, assigned_ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw center line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

    # Overlay stats
    cv2.rectangle(frame, (0, 0), (340, 90), (5, 5, 5), -1)
    cv2.putText(frame, f"Visible: {visible_now}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2,cv2.LINE_AA)
    cv2.putText(frame, f"Entered: {entered_total}",(10, 55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,cv2.LINE_AA)
    cv2.putText(frame, f"Exited: {exited_total}",(10, 80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 165, 255),2,cv2.LINE_AA)

    # ALERT logic
    now = time.time()
    alert_triggered = False
    
    # Check if threshold crossed
    if visible_now >= threshold:
        # Check cooldown
        if (now - last_alert_time) >= cooldown_seconds:
            last_alert_time = now
            alert_triggered = True

            # Red border on video
            cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 3)

            # Play Sound
            if enable_sound:
                play_alert_sound()

            # Send Email
            if enable_email:
                subject = "DeepVision Local Alert: Crowd Threshold Exceeded"
                body = (
                    f"DeepVision Local detected high crowd.\n"
                    f"Visible now: {visible_now}\n"
                    f"Threshold: {threshold}"
                )
                send_email_async(
                    email_sender,
                    email_app_password,
                    email_receiver,
                    subject,
                    body,
                )

    return (
        frame,
        visible_now,
        entered_total,
        exited_total,
        tracks,
        last_y,
        next_id_global,
        last_alert_time,
        alert_triggered,
    )

# =========================
# Streamlit UI
# =========================
st.markdown(
    "<h1 style='text-align: center;'>🧠 DeepVision Local – Webcam & Video Crowd Monitor</h1>",
    unsafe_allow_html=True,
)

# === ALERT PLACEHOLDER (Main Screen) ===
# This placeholder sits at the top to show the alert message
alert_main_ph = st.empty()

mode = st.sidebar.radio("Mode", ["Live Webcam", "Upload Video"])

threshold = st.sidebar.number_input(
    "Alert Threshold (Visible People)",
    min_value=1,
    max_value=500,
    value=5,
    step=1,
)

cooldown_seconds = st.sidebar.number_input(
    "Alert Cooldown (seconds)",
    min_value=3,
    max_value=3600,
    value=10,
    step=1,
)

enable_sound = st.sidebar.checkbox("Enable Sound Alert", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📧 Email Alert (Gmail)")
enable_email = st.sidebar.checkbox("Enable Email Alert", value=False)
email_sender = st.sidebar.text_input("Sender Gmail", value="")
email_app_password = st.sidebar.text_input("Gmail App Password", type="password", value="")
email_receiver = st.sidebar.text_input("Receiver Email", value="")

col1, col2, col3 = st.columns(3)
visible_ph = col1.empty()
entered_ph = col2.empty()
exited_ph = col3.empty()

visible_ph.metric("Currently Visible", 0)
entered_ph.metric("Total Entered", 0)
exited_ph.metric("Total Exited", 0)

video_placeholder = st.empty()

if "last_alert_time_local" not in st.session_state:
    st.session_state["last_alert_time_local"] = 0.0

# -------------------------
# MODE 1: LIVE WEBCAM
# -------------------------
if mode == "Live Webcam":
    st.info("This mode uses your laptop webcam.")
    start_btn = st.button("▶ Start Webcam")
    stop_placeholder = st.empty()

    if start_btn:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
        else:
            stop_placeholder.button("■ Stop Webcam", key="stop_webcam")

            tracks = {}
            last_y = {}
            next_id_global = 1
            entered_total = 0
            exited_total = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                (
                    frame_out,
                    visible_now,
                    entered_total,
                    exited_total,
                    tracks,
                    last_y,
                    next_id_global,
                    st.session_state["last_alert_time_local"],
                    alert_triggered,
                ) = process_frame(
                    frame,
                    model,
                    tracks,
                    last_y,
                    next_id_global,
                    entered_total,
                    exited_total,
                    threshold,
                    enable_sound,
                    enable_email,
                    email_sender,
                    email_app_password,
                    email_receiver,
                    st.session_state["last_alert_time_local"],
                    cooldown_seconds,
                )

                rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb, use_container_width=True)

                visible_ph.metric("Currently Visible", visible_now)
                entered_ph.metric("Total Entered", entered_total)
                exited_ph.metric("Total Exited", exited_total)

                # === ALERT UI LOGIC ===
                if alert_triggered:
                    # Pop-up toast
                    st.toast("🚨 ALERT: Crowd Threshold Crossed!", icon="🚨")
                    # Big Red Message in Main Area
                    alert_main_ph.error(f"🚨 ALERT! Visible Count ({visible_now}) exceeded Threshold ({threshold})!")
                elif visible_now < threshold:
                    # Clear the alert if things are normal
                    alert_main_ph.empty()

            cap.release()
            stop_placeholder.empty()

# -------------------------
# MODE 2: UPLOAD VIDEO
# -------------------------
else:
    uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("▶ Start Processing Video"):
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

                (
                    frame_out,
                    visible_now,
                    entered_total,
                    exited_total,
                    tracks,
                    last_y,
                    next_id_global,
                    st.session_state["last_alert_time_local"],
                    alert_triggered,
                ) = process_frame(
                    frame,
                    model,
                    tracks,
                    last_y,
                    next_id_global,
                    entered_total,
                    exited_total,
                    threshold,
                    enable_sound,
                    enable_email,
                    email_sender,
                    email_app_password,
                    email_receiver,
                    st.session_state["last_alert_time_local"],
                    cooldown_seconds,
                )

                rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb, use_container_width=True)

                visible_ph.metric("Currently Visible", visible_now)
                entered_ph.metric("Total Entered", entered_total)
                exited_ph.metric("Total Exited", exited_total)

                # === ALERT UI LOGIC ===
                if alert_triggered:
                    st.toast("🚨 ALERT: Crowd Threshold Crossed!", icon="🚨")
                    alert_main_ph.error(f"🚨 ALERT! Visible Count ({visible_now}) exceeded Threshold ({threshold})!")
                elif visible_now < threshold:
                    alert_main_ph.empty()

                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            try:
                os.unlink(tmp_video_path)
            except:
                pass
            st.success("Video processing complete.")