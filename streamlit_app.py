import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from twilio.rest import Client


# ==============================
# 🔐 LOAD TWILIO SECRETS
# ==============================
ACCOUNT_SID = st.secrets["TWILIO_SID"]
AUTH_TOKEN = st.secrets["TWILIO_AUTH"]
TWILIO_NUMBER = st.secrets["TWILIO_NUMBER"]
DESTINATION_NUMBER = st.secrets["DESTINATION_NUMBER"]


# ==============================
# 📦 Load Models
# ==============================
@st.cache_resource
def load_models():
    model = load_model("accident_model.h5")
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model, base


# ==============================
# 📲 Send SMS (SAFE + DEBUG)
# ==============================
def send_sms(timestamp):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        message = client.messages.create(
            body=f"🚨 ALERT! Accident detected at {timestamp} seconds.",
            from_=TWILIO_NUMBER,
            to=DESTINATION_NUMBER
        )

        st.success(f"SMS Sent! SID: {message.sid}")
        return True

    except Exception as e:
        st.error(f"SMS Error: {str(e)}")
        return False


# ==============================
# 🎥 Accident Prediction
# ==============================
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

        if fps > 0 and frame_idx % max(1, math.floor(fps)) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = resize(frame_rgb, (224, 224), preserve_range=True).astype(int)
            inp = preprocess_input(np.array([frame_resized]))

            feat = base_model.predict(inp, verbose=0)
            feat = feat.reshape(1, 7 * 7 * 512)

            if feat.max() != 0:
                feat = feat / feat.max()

            pred = model.predict(feat, verbose=0)[0]

            if pred[0] > pred[1]:
                timestamps.append(sec)

            sec += 1

        frame_idx += 1

    cap.release()
    return timestamps


# ==============================
# 🚨 UI
# ==============================
st.title("🚨 Accident Detection + SMS Alert")

uploaded_video = st.file_uploader("Upload MP4 Video", type=["mp4"])

model, base_model = load_models()

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        path = tmp.name

    with st.spinner("Analyzing video..."):
        results = predict_accident(path, model, base_model)

    if len(results) == 0:
        st.success("✔ No accident detected.")
    else:
        st.warning("⚠ Accident detected!")

        for t in results:
            st.write(f"Detected at: {t} sec")

        # Send SMS once
        send_sms(results[0])

