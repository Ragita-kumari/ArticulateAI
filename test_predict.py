# test_predict.py
import os, torch, cv2, numpy as np
from models.model import LipReadingModel

sample_path = "dataset/train/clean/clean_1.avi"   # change only if your file differs
if not os.path.exists(sample_path):
    print("SAMPLE NOT FOUND:", sample_path)
    raise SystemExit

# read frames, RGB, resize 64x64, pad to 75
frames = []
cap = cv2.VideoCapture(sample_path)
while True:
    ret, f = cap.read()
    if not ret:
        break
    try:
        f = cv2.resize(f, (64,64))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    except Exception as e:
        continue
    frames.append(f.astype('float32')/255.0)
cap.release()

if len(frames) == 0:
    print("NO FRAMES READ from", sample_path)
    raise SystemExit

MAX_FRAMES = 75
if len(frames) < MAX_FRAMES:
    frames = frames + [np.zeros((64,64,3), dtype='float32')] * (MAX_FRAMES - len(frames))
else:
    frames = frames[:MAX_FRAMES]

arr = np.stack(frames, axis=0)
tensor = torch.tensor(arr).permute(0,3,1,2).unsqueeze(0)  # shape (1,T,C,H,W)

# load classes
with open("classes.txt", "r", encoding="utf-8") as f:
    classes = [l.strip() for l in f.readlines() if l.strip()]

print("Loaded classes:", classes)
print("Loading model checkpoint...")

model = LipReadingModel(num_classes=len(classes))
ckpt = "checkpoints/best_lip_model.pth"
if not os.path.exists(ckpt):
    print("Checkpoint not found:", ckpt)
    raise SystemExit

# load (strict) — will raise if mismatch
state = torch.load(ckpt, map_location='cpu')
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    out = model(tensor)
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]

top3 = probs.argsort()[-3:][::-1]
print("Top-3 predictions:")
for i in top3:
    idx = int(i)
    label = classes[idx] if idx < len(classes) else str(idx)
    print(f" - {label}: {probs[idx]:.4f}")
