import streamlit as st
import torch
import cv2
import numpy as np
from models.model import LipNetModel
import os

# ----------------------------------------------------
# Load vocabulary from classes.txt
# ----------------------------------------------------
def load_classes(path="classes.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
        st.success(f"Loaded {len(classes)} classes")
        return classes
    except FileNotFoundError:
        st.error("❌ classes.txt not found! Run training first.")
        return []


# ----------------------------------------------------
# Video → Frames → Tensor
# ----------------------------------------------------
def preprocess_video(video_bytes, max_frames=75):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        st.error("❌ Unable to read video frames!")
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

    # Pad or trim
    T = frames.shape[0]
    if T < max_frames:
        pad = torch.zeros((max_frames - T, 3, 64, 64))
        frames = torch.cat((frames, pad), dim=0)
    else:
        frames = frames[:max_frames]

    return frames.unsqueeze(0)  # [1, T, C, H, W]


# ----------------------------------------------------
# Load trained model safely
# ----------------------------------------------------
def load_model(num_classes):
    model = LipNetModel(num_classes=len(classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ BEST MODEL FIRST
    if os.path.exists("checkpoints/best_lip_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_lip_model.pth", map_location=device), strict=False)
        st.success("✅ Loaded: checkpoints/best_lip_model.pth")

    # 2️⃣ FALLBACK: FINAL MODEL
    elif os.path.exists("lip_model_final.pth"):
        model.load_state_dict(torch.load("lip_model_final.pth", map_location=device), strict=False)
        st.warning("⚠ Loaded fallback: lip_model_final.pth")

    else:
        st.error("❌ No trained model found! Train first.")
        st.stop()

    model.to(device)
    model.eval()
    return model, device


# ================================================================
#                          STREAMLIT UI
# ================================================================
st.title("👄 Real-Time Lip Reading Demo (Custom Model)")
st.write("Upload a **mute video** and get predicted lip-read text.")

# Load class names
classes = load_classes()

if len(classes) == 0:
    st.stop()

# Load the trained model
model, device = load_model(len(classes))

# File uploader
uploaded = st.file_uploader("Upload a video (.mp4 / .avi)", type=["mp4", "avi", "mov", "mpg"])

if uploaded is not None:
    st.video(uploaded)

    if st.button("🔍 Predict"):
        with st.spinner("Processing video..."):
            frames = preprocess_video(uploaded.read())

            if frames is None:
                st.stop()

            frames = frames.to(device)

            with torch.no_grad():
                output = model(frames)
                pred_idx = torch.argmax(output, dim=1).item()

            if pred_idx < len(classes):
                st.success(f"### 🎯 Predicted Word: **{classes[pred_idx]}**")
            else:
                st.error("❌ Prediction index out of range.")
