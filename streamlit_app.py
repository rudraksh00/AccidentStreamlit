import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from twilio.rest import Client
from streamlit_js_eval import get_geolocation


# ======================================================
# 🔐 LOAD SECRETS
# ======================================================
ACCOUNT_SID = st.secrets["TWILIO_SID"]
AUTH_TOKEN = st.secrets["TWILIO_AUTH"]
TWILIO_NUMBER = st.secrets["TWILIO_NUMBER"]
DESTINATION_NUMBER = st.secrets["DESTINATION_NUMBER"]


# ======================================================
# 📦 LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    model = load_model("accident_model.h5")
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    return model, base_model


# ======================================================
# 📲 SEND SMS
# ======================================================
def send_sms(timestamp, lat=None, lon=None):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        if lat and lon:
            maps_link = f"https://maps.google.com/?q={lat},{lon}"
            body = (
                f"🚨 ACCIDENT ALERT!\n\n"
                f"Detected at: {timestamp} sec\n\n"
                f"📍 Location:\n{maps_link}"
            )
        else:
            body = (
                f"🚨 ACCIDENT ALERT!\n\n"
                f"Detected at: {timestamp} sec\n\n"
                f"📍 Location: Not Available"
            )

        client.messages.create(
            body=body,
            from_=TWILIO_NUMBER,
            to=DESTINATION_NUMBER
        )

        st.success("📲 SMS Sent Successfully!")

    except Exception as e:
        st.error(f"SMS Failed: {str(e)}")


# ======================================================
# 🎥 ACCIDENT DETECTION
# ======================================================
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

        if fps > 0 and frame_idx % max(1, int(fps)) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = resize(
                frame_rgb,
                (224, 224),
                preserve_range=True
            ).astype(int)

            inp = preprocess_input(np.array([frame_resized]))

            features = base_model.predict(inp, verbose=0)
            features = features.reshape(1, 7 * 7 * 512)

            if features.max() != 0:
                features = features / features.max()

            prediction = model.predict(features, verbose=0)[0]

            # Index 0 = Accident
            if prediction[0] > prediction[1]:
                timestamps.append(sec)

            sec += 1

        frame_idx += 1

    cap.release()
    return timestamps


# ======================================================
# 🚨 STREAMLIT UI
# ======================================================
st.title("🚨 Accident Detection + SMS + Location")

st.write("Step 1: Capture location → Step 2: Upload video → Step 3: SMS alert")


# ------------------------------------------------------
# 📍 MANUAL LOCATION CAPTURE (RELIABLE)
# ------------------------------------------------------
if "user_location" not in st.session_state:
    st.session_state.user_location = None

if st.button("📍 Capture My Location"):
    location = get_geolocation()

    if location and isinstance(location, dict) and "coords" in location:
        coords = location["coords"]
        lat = coords.get("latitude")
        lon = coords.get("longitude")

        if lat and lon:
            st.session_state.user_location = (lat, lon)
            st.success("Location captured successfully!")
        else:
            st.warning("Coordinates not available.")
    else:
        st.warning("Location permission denied or unavailable.")


# Display stored location
lat = None
lon = None

if st.session_state.user_location:
    lat, lon = st.session_state.user_location
    st.info(f"Using Location: {lat}, {lon}")
else:
    st.info("No location captured yet.")


# ------------------------------------------------------
# 📤 VIDEO UPLOAD
# ------------------------------------------------------
uploaded_video = st.file_uploader("Upload MP4 Video", type=["mp4"])

model, base_model = load_models()

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    with st.spinner("Analyzing video..."):
        results = predict_accident(video_path, model, base_model)

    if len(results) == 0:
        st.success("✔ No accident detected.")
    else:
        st.error("⚠ Accident detected!")

        for t in results:
            st.write(f"Detected at: {t} sec")

        # Send SMS once
        send_sms(results[0], lat, lon)

