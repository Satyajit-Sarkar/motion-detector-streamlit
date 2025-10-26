import cv2
import numpy as np
import streamlit as st

st.title("Motion Detection and Marking")

option = st.radio("Select video source:", ("Webcam", "Upload"))

if option == "Upload":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
elif option == "Webcam":
    st.info("Click 'Start' to use your webcam.")
    if st.button("Start"):
        cap = cv2.VideoCapture(0)
    else:
        cap = None
else:
    cap = None

if cap is not None and cap.isOpened():
    ret, prev_frame = cap.read()
    if not ret:
        st.error("Could not read video source.")
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        motion_frames = []

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, current_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < 20:
                    continue
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if motion_detected and frame_count % 10 == 0:
                motion_frames.append(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))

            prev_gray = current_gray
            frame_count += 1

        cap.release()

        st.write(f"Total frames with detected motion (every 10th): {len(motion_frames)}")
        for idx, frame in enumerate(motion_frames):
            st.image(frame, caption=f"Motion Frame {idx*10}", use_column_width=True)