import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize


@st.cache_resource
def load_models():
    model = load_model("accident_model.h5")
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model, base


def predict_accident(video_path, model, base_model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    sec = 0
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # sample 1 fps
        if frame_idx % math.floor(fps) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = resize(frame_rgb, (224, 224), preserve_range=True).astype(int)
            inp = preprocess_input(np.array([frame_resized]), data_format=None)

            feat = base_model.predict(inp)
            feat = feat.reshape(1, 7 * 7 * 512)
            feat = feat / feat.max()

            pred = model.predict(feat)[0]

            # your notebook logic:
            # index 0 = Accident, index 1 = No Accident
            if pred[0] > pred[1]:
                timestamps.append(sec)

            sec += 1

        frame_idx += 1

    cap.release()
    return timestamps


st.title("🚨 Accident Detection")
st.write("Upload an MP4 video → Detect timestamps where accident occurs")

uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

model, base_model = load_models()

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        path = tmp.name

    with st.spinner("Processing video..."):
        results = predict_accident(path, model, base_model)

    st.success("Done!")

    if len(results) == 0:
        st.write("### ✔ No accident detected")
    else:
        st.write("### ⚠ Accident detected at:")
        for t in results:
            st.write(f"- {t} sec")
