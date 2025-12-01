import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CustomLipDataset(Dataset):
    """
    Custom Lip-Reading Dataset Loader
    Folder Structure:
        datasets/my_dataset/
            ├── hello/
            │     ├── hello_1.mp4
            │     ├── hello_2.mp4
            ├── yes/
            ├── no/
            └── ...

    Each folder name = class label
    Each video = 1 sample
    """

    def __init__(self, root="datasets/my_dataset", max_frames=75):
        self.root = root
        self.max_frames = max_frames

        # ------------------------------------------
        # Load class folders
        # ------------------------------------------
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        print(f"📝 Found Classes: {self.classes}")
        print(f"📚 Total Classes: {len(self.classes)}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Save vocabulary for Streamlit
        with open("classes.txt", "w") as f:
            for cls in self.classes:
                f.write(cls + "\n")

        # ------------------------------------------
        # Load all (video_path, label_idx)
        # ------------------------------------------
        self.samples = []
        video_extensions = (".mp4", ".avi", ".mov")

        for cls in self.classes:
            cls_folder = os.path.join(root, cls)
            for file in os.listdir(cls_folder):
                if file.lower().endswith(video_extensions):
                    vpath = os.path.join(cls_folder, file)
                    self.samples.append((vpath, self.class_to_idx[cls]))

        print(f"🎥 Total videos found: {len(self.samples)}")

    # ---------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ---------------------------------------------------
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # ------------------------------
        # 1. Load video frames
        # ------------------------------
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (64, 64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # ------------------------------
        # 2. Convert to tensor
        # ------------------------------
        frames = np.array(frames, dtype=np.float32) / 255.0

        # shape → [T, 64, 64, 3] → [T, 3, 64, 64]
        frames = torch.tensor(frames).permute(0, 3, 1, 2)

        # ------------------------------
        # 3. Pad or trim to max_frames
        # ------------------------------
        T = frames.shape[0]

        if T < self.max_frames:
            pad = torch.zeros((self.max_frames - T, 3, 64, 64))
            frames = torch.cat([frames, pad], dim=0)
        else:
            frames = frames[:self.max_frames]

        return frames, torch.tensor(label, dtype=torch.long)
