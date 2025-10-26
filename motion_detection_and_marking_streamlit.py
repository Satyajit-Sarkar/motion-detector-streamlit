import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import tempfile
import os

st.set_page_config(page_title="Motion Detection App", layout="centered")
st.title("🎥 Motion Detection and Marking")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
min_area = st.sidebar.slider("Minimum contour area for motion detection", 1, 500, 20, 1)
source_option = st.sidebar.radio("Select video source:", ("Webcam (Live)", "Upload Video"))

# --- Common explanation ---
st.markdown(
    """
    This app detects motion in video streams using OpenCV.
    You can either **upload a video file** or **use your webcam** for live motion detection.
    Bounding boxes indicate detected motion areas.
    """
)

# --- Option 1: Uploaded video processing ---
if source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file.name)

        ret, prev_frame = cap.read()
        if not ret:
            st.error("Could not read the uploaded video file.")
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_count = 0
            motion_frames = []

            st.info("⏳ Processing video... please wait.")
            progress = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break

                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, current_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if frame_count % 10 == 0:
                    motion_frames.append(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))

                prev_gray = current_gray
                frame_count += 1
                progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            os.unlink(temp_file.name)
            st.success(f"✅ Motion detected in {len(motion_frames)} sampled frames.")

            for idx, frame in enumerate(motion_frames):
                st.image(frame, caption=f"Motion Frame {idx*10}", use_column_width=True)

# --- Option 2: Live webcam via WebRTC ---
elif source_option == "Webcam (Live)":
    st.info("📸 Allow webcam access to start live motion detection.")

    class MotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.prev_gray = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.prev_gray is not None:
                diff = cv2.absdiff(self.prev_gray, gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.prev_gray = gray
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="motion-live",
        video_processor_factory=MotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )