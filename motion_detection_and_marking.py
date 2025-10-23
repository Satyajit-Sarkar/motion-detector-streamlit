import cv2
import streamlit as st
import numpy as np

st.title("Motion Detection and Marking")

mode = st.radio("Select input source:", ("Webcam", "Upload Video"))
frame_placeholder = st.empty()
save_motion = st.checkbox("Save frames with motion", value=False)
min_area = st.slider("Minimum contour area for motion", min_value=1, max_value=1000, value=20, step=1)

cap = None

if mode == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(video_file.read())
        tfile.close()
        cap = cv2.VideoCapture("temp_video.mp4")
elif mode == "Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)

if cap is not None:
    ret, prev_frame = cap.read()
    if not ret:
        st.error("Could not read video or webcam.")
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0

        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret:
                break

            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, current_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cap.release()