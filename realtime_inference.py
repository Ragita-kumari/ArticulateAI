import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn

# ========================
# INLINE MODEL DEFINITION (FullModel)
# Must match exactly what was trained
# ========================
class FullModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.cnn = nn.Module()
        self.cnn.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(8192, 256, batch_first=True, bidirectional=True)
        self.attn = nn.Module()
        self.attn.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.cnn.encoder(x[:, t])
            f = self.flatten(f)
            feats.append(f)
        feats = torch.stack(feats, dim=1)
        lstm_out, _ = self.lstm(feats)
        attn_weights = torch.softmax(self.attn.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)

# ========================
# LOAD CLASSES
# ========================
with open("classes.txt") as f:
    vocab = [line.strip() for line in f.readlines()]

num_classes = len(vocab)
print(f"Loaded {num_classes} classes: {vocab}")

# ========================
# LOAD MODEL (12 classes as trained)
# ========================
CHECKPOINT_NUM_CLASSES = 12
model = FullModel(num_classes=CHECKPOINT_NUM_CLASSES)
state_dict = torch.load("checkpoints/best_lip_model.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()
print("Model loaded successfully.")

# ========================
# MEDIAPIPE
# ========================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================
# WEBCAM
# ========================
cap = cv2.VideoCapture(0)

# ========================
# VARIABLES — match training: max_frames=75
# ========================
SEQ_LEN = 75
frames = []
last_word = ""
collecting = False
no_motion_count = 0
prev_lip_gray = None

print("Ready! Start speaking a word...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            lip_ids = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
            coords = [(int(face_landmarks.landmark[i].x * w),
                       int(face_landmarks.landmark[i].y * h)) for i in lip_ids]

            cx = int(np.mean([c[0] for c in coords]))
            cy = int(np.mean([c[1] for c in coords]))

            box_size = 80
            x_min = max(0, cx - box_size // 2)
            x_max = min(w, cx + box_size // 2)
            y_min = max(0, cy - box_size // 2)
            y_max = min(h, cy + box_size // 2)

            lip_crop = frame[y_min:y_max, x_min:x_max]
            if lip_crop.size == 0:
                continue

            # ── Preprocess matching CustomLipDataset EXACTLY ──
            # Training: resize → BGR2RGB → /255 → permute
            lip_64  = cv2.resize(lip_crop, (64, 64))
            lip_rgb = cv2.cvtColor(lip_64, cv2.COLOR_BGR2RGB)
            lip_norm = lip_rgb.astype(np.float32) / 255.0
            lip_chw  = np.transpose(lip_norm, (2, 0, 1))   # (3,64,64)

            # ── Motion detection ──
            lip_gray = cv2.cvtColor(lip_64, cv2.COLOR_BGR2GRAY).astype(np.float32)
            motion = 0.0
            if prev_lip_gray is not None:
                motion = np.mean(np.abs(lip_gray - prev_lip_gray))
            prev_lip_gray = lip_gray.copy()

            MOTION_THRESHOLD = 5.0
            if motion > MOTION_THRESHOLD:
                collecting = True
                no_motion_count = 0
            else:
                if collecting:
                    no_motion_count += 1

            if no_motion_count > 20:
                collecting = False
                no_motion_count = 0
                frames = []

            if collecting:
                frames.append(lip_chw)

            # ── Predict once we have SEQ_LEN frames ──
            if collecting and len(frames) >= SEQ_LEN:
                seq = np.stack(frames[-SEQ_LEN:], axis=0)        # (75,3,64,64)
                input_tensor = torch.tensor(seq).unsqueeze(0).float()  # (1,75,3,64,64)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs  = torch.softmax(output, dim=1)
                    pred   = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item()

                top3 = torch.topk(probs[0], min(3, CHECKPOINT_NUM_CLASSES))
                top3_str = " | ".join(
                    f"{vocab[i] if i < len(vocab) else 'class_'+str(i)}:{p:.2f}"
                    for i, p in zip(top3.indices.tolist(), top3.values.tolist())
                )
                print(f"Top3: {top3_str}")

                if confidence > 0.6:
                    last_word = vocab[pred] if pred < len(vocab) else f"class_{pred}"
                    print(f">>> PREDICTED: {last_word} ({confidence:.2f})")
                    frames = []
                    collecting = False

    # ── Display ──
    status = "Speaking..." if collecting else "Waiting..."
    cv2.putText(frame, f"Word: {last_word}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, status, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("Lip Reading AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()