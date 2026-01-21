import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
import os


@st.cache_resource
def load_models():
    model = load_model("accident_model.h5")
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model, base


def process_video(video_path, model, base_model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    sec = 0

    accident_timestamps = []
    accident_frames = []
    annotated_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % math.floor(fps) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = resize(frame_rgb, (224,224), preserve_range=True).astype(int)
            inp = preprocess_input(np.array([resized]), data_format=None)

            feat = base_model.predict(inp)
            feat = feat.reshape(1, 7*7*512)
            feat = feat/feat.max()
            pred = model.predict(feat)[0]

            accident = pred[0] > pred[1]

            if accident:
                accident_timestamps.append(sec)
                accident_frames.append(frame_rgb)

                annotated = frame_rgb.copy()
                cv2.putText(annotated, "ACCIDENT", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255,0,0), 2, cv2.LINE_AA)
                annotated_frames.append(annotated)

            sec += 1

        frame_idx += 1

    cap.release()

    return accident_timestamps, accident_frames, annotated_frames, fps



st.title("🚨 Accident Detection AI - Full Visual Version")

model, base_model = load_models()

uploaded_video = st.file_uploader("Upload MP4 Video", type=["mp4"])

if uploaded_video:
    # preview original
    st.subheader("Original Video")
    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    with st.spinner("Processing AI detection..."):
        timestamps, frames, annotated, fps = process_video(video_path, model, base_model)

    st.success("AI processing completed!")

    # timestamps result
    if timestamps:
        st.subheader("⚠ Accident Detected At:")
        st.write(", ".join([f"{t}s" for t in timestamps]))
    else:
        st.subheader("✔ No Accident Detected")
        st.stop()

    # show evidence frames
    st.subheader("🖼 Accident Evidence Frames")
    for i, f in enumerate(frames):
        st.image(f, caption=f"{timestamps[i]}s", use_column_width=True)

    # build annotated output video
    st.subheader("🎥 Annotated Video Output")
    out_path = video_path.replace(".mp4", "_processed.mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    sec = 0
    frame_idx = 0

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        accident = sec in timestamps and frame_idx % math.floor(fps) == 0

        if accident:
            cv2.putText(frame, "ACCIDENT", (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0,0,255), 3, cv2.LINE_AA)

        out.write(frame)

        if frame_idx % math.floor(fps) == 0:
            sec += 1
        frame_idx += 1

    cap.release()
    out.release()

    st.video(out_path)

    with open(out_path, "rb") as f:
        st.download_button(
            label="⬇ Download Annotated Video",
            data=f,
            file_name="accident_annotated.mp4",
            mime="video/mp4"
        )
