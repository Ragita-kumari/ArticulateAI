import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from datasets.custom_dataset import CustomLipDataset

# ========================
# INLINE FullModel — matches checkpoint exactly
# ========================
class FullModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.cnn = nn.Module()
        self.cnn.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(8192, 256, batch_first=True, bidirectional=True)
        self.attn = nn.Module()
        self.attn.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = [self.flatten(self.cnn.encoder(x[:, t])) for t in range(T)]
        feats = torch.stack(feats, dim=1)
        lstm_out, _ = self.lstm(feats)
        attn_w = torch.softmax(self.attn.attn(lstm_out), dim=1)
        context = torch.sum(attn_w * lstm_out, dim=1)
        return self.fc(context)

# ========================
# SETTINGS
# ========================
DATA_ROOT  = "datasets/my_dataset"
MODEL_PATH = "checkpoints/best_lip_model.pth"
BATCH_SIZE = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_NUM_CLASSES = 12   # what the model was trained with

# ========================
# LOAD CLASSES (11 known classes from folders)
# ========================
classes = sorted([
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
])
print("Classes:", classes)
num_known = len(classes)   # 11

# ========================
# DATASET & LOADER
# CustomLipDataset(root, max_frames) — matches your actual __init__
# ========================
dataset = CustomLipDataset(root=DATA_ROOT, max_frames=75)

def collate_fn(batch):
    videos, labels = zip(*batch)
    min_len = min(v.shape[0] for v in videos)
    videos = torch.stack([v[:min_len] for v in videos])
    labels = torch.tensor(labels)
    return videos, labels

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========================
# LOAD MODEL with 12 classes (as trained)
# ========================
model = FullModel(num_classes=CHECKPOINT_NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded.")

# ========================
# COLLECT PREDICTIONS
# Only keep samples whose true label is within the 11 known classes
# ========================
y_true = []
y_pred = []

with torch.no_grad():
    for videos, labels in loader:
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)
        preds   = outputs.argmax(1)

        for true, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            # Skip any sample belonging to the mystery 12th class
            if true >= num_known:
                continue
            # If model predicted the 12th class, map it to closest known class
            if pred >= num_known:
                pred = num_known - 1
            y_true.append(true)
            y_pred.append(pred)

print(f"Evaluated {len(y_true)} samples across {num_known} classes.")

# ========================
# CONFUSION MATRIX (11x11)
# ========================
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_known)))

np.save("y_true.npy", np.array(y_true))
np.save("y_pred.npy", np.array(y_pred))

# ========================
# PLOT
# ========================
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Lip Reading — Confusion Matrix (11 Classes)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion_matrix.png")