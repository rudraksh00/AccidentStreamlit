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
# 🔐 TWILIO CONFIG
# ==============================
ACCOUNT_SID = "AC6d6bfad7869fc35bd540b550a2b5ccdd"
AUTH_TOKEN = "c090c6bec04d8024c3ba583d98361a27"
TWILIO_NUMBER = "+13133074835"
DESTINATION_NUMBER = "+917045239053"


# ==============================
# 📦 Load Models
# ==============================
@st.cache_resource
def load_models():
    model = load_model("accident_model.h5")
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model, base


# ==============================
# 📲 Send SMS
# ==============================
def send_sms(timestamp):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        message = client.messages.create(
            body=f"🚨 ALERT! Accident detected at {timestamp} seconds.",
            from_=TWILIO_NUMBER,
            to=DESTINATION_NUMBER
        )

        return True

    except Exception as e:
        st.error(f"SMS Failed: {e}")
        return False


# ==============================
# 🎥 Prediction Logic
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

        if fps > 0 and frame_idx % math.floor(fps) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = resize(frame_rgb, (224, 224), preserve_range=True).astype(int)
            inp = preprocess_input(np.array([frame_resized]))

            feat = base_model.predict(inp, verbose=0)
            feat = feat.reshape(1, 7 * 7 * 512)
            feat = feat / feat.max()

            pred = model.predict(feat, verbose=0)[0]

            if pred[0] > pred[1]:  # Accident
                timestamps.append(sec)

            sec += 1

        frame_idx += 1

    cap.release()
    return timestamps


# ==============================
# 🚨 Streamlit UI
# ==============================
st.title("🚨 Accident Detection with SMS Alert")
st.write("Upload video → Detect accident → Send SMS alert")

uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

model, base_model = load_models()

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        path = tmp.name

    with st.spinner("Processing video..."):
        results = predict_accident(path, model, base_model)

    st.success("Processing Complete!")

    if len(results) == 0:
        st.write("### ✔ No accident detected")
    else:
        st.write("### ⚠ Accident detected at:")
        for t in results:
            st.write(f"- {t} sec")

        # Send SMS only once (first detection)
        sms_sent = send_sms(results[0])

        if sms_sent:
            st.success("📲 SMS Alert Sent Successfully!")

