import cv2
import os
import time

# -------------------------
# SETTINGS
# -------------------------
DATASET_ROOT = "datasets/my_dataset"
CLASS_NAME = "hello"      # change this to record other words
RECORD_SECONDS = 2
FPS = 30
WIDTH, HEIGHT = 640, 480
# -------------------------

os.makedirs(os.path.join(DATASET_ROOT, CLASS_NAME), exist_ok=True)
class_folder = os.path.join(DATASET_ROOT, CLASS_NAME)

# Auto filename
existing = [f for f in os.listdir(class_folder) if f.endswith(".mp4")]
next_id = len(existing) + 1
file_path = os.path.join(class_folder, f"{CLASS_NAME}_{next_id}.mp4")

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(file_path, fourcc, FPS, (WIDTH, HEIGHT))

print("\n📸 Starting in 3 sec...")
time.sleep(3)

print(f"🎬 Recording {RECORD_SECONDS} sec for word: '{CLASS_NAME}'")
start = time.time()

while time.time() - start < RECORD_SECONDS:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    # We skip cv2.imshow because Windows GUI backend doesn't work in some venvs

cap.release()
out.release()
print(f"✅ Saved: {file_path}")
