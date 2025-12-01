import streamlit as st
import torch
import cv2
import numpy as np
from models.model import LipReadingModel
import os

# --------------------------------------------------------
# Load custom classes from your dataset folder
# --------------------------------------------------------
def load_classes(root="datasets/my_dataset"):
    classes = sorted(os.listdir(root))
    st.success(f"Loaded {len(classes)} classes: {classes}")
    return classes


# --------------------------------------------------------
# Preprocess uploaded video → frames → tensor [1, T, 3, 64, 64]
# --------------------------------------------------------
def preprocess_video(video_bytes, max_frames=75):
    temp_path = "temp_input.avi"

    # Save uploaded file to disk
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
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        st.error("❌ Unable to read frames!")
        return None

    frames = np.array(frames)  # [T,H,W,3]
    T = frames.shape[0]

    # Pad or trim
    if T < max_frames:
        pad = np.zeros((max_frames - T, 64, 64, 3), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:max_frames]

    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # [T,3,64,64]
    return frames.unsqueeze(0)  # [1,T,3,64,64]


# --------------------------------------------------------
# Load model
# --------------------------------------------------------
def load_model(num_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LipReadingModel(num_classes=num_classes).to(device)

    # Try best model first
    if os.path.exists("checkpoints/best_lip_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_lip_model.pth", map_location=device))
        st.success("Loaded best_lip_model.pth")
    else:
        model.load_state_dict(torch.load("lip_model_final.pth", map_location=device))
        st.warning("Loaded lip_model_final.pth")

    model.eval()
    return model, device


# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------
st.title("👄 Real-Time Lip Reading Demo (Custom Model)")
st.write("Upload a mouth-cropped video clip recorded with your Tkinter tool.")

# Load custom classes
classes = load_classes()

model, device = load_model(num_classes=len(classes))

uploaded_file = st.file_uploader("Upload a .mp4 / .avi video", type=["mp4", "avi"])

if uploaded_file is not None:
    # Show uploaded video
    st.video(uploaded_file)

    if st.button("🔍 Predict Word"):
        with st.spinner("Processing video..."):
            frames = preprocess_video(uploaded_file.read())

            if frames is None:
                st.error("Video processing failed.")
            else:
                frames = frames.to(device)

                with torch.no_grad():
                    output = model(frames)
                    pred_idx = torch.argmax(output, dim=1).item()

                st.success(f"### 🎯 Predicted Word: **{classes[pred_idx]}**")
