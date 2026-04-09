import cv2
import os
import time
import tkinter as tk
from PIL import Image, ImageTk

# ===============================
# SETTINGS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "datasets", "my_dataset")

CLASS_NAME = "green"
RECORD_SECONDS = 2
FPS = 20
CROP_SIZE = 64
WIDTH, HEIGHT = 640, 480
# ===============================

# ===============================
# Dataset folder
# ===============================
class_folder = os.path.join(DATASET_ROOT, CLASS_NAME)
os.makedirs(class_folder, exist_ok=True)

def get_next_file():
    max_idx = 0
    for f in os.listdir(class_folder):
        if f.startswith(CLASS_NAME) and f.endswith(".avi"):
            try:
                idx = int(f.split("_")[1].split(".")[0])
                max_idx = max(max_idx, idx)
            except:
                pass
    return os.path.join(class_folder, f"{CLASS_NAME}_{max_idx + 1}.avi")

# ===============================
# Face Detector
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_face = None

def crop_mouth(frame):
    global last_face
    h, w = frame.shape[:2]

    if last_face is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            last_face = max(faces, key=lambda x: x[2] * x[3])

    if last_face is not None:
        x, y, fw, fh = last_face
        my = y + int(fh * 0.55)
        mh = int(fh * 0.45)
        crop = frame[my:my+mh, x:x+fw]
    else:
        crop = frame[int(h*0.6):h, int(w*0.3):int(w*0.7)]

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

# ===============================
# Tkinter UI
# ===============================
root = tk.Tk()
root.title("Lip Dataset Recorder")
root.geometry("700x560")

label_class = tk.Label(
    root,
    text=f"Recording Class: {CLASS_NAME.upper()}",
    font=("Arial", 16, "bold"),
    fg="blue"
)
label_class.pack(pady=8)

panel = tk.Label(root)
panel.pack()

label_status = tk.Label(root, text="Press SPACE to record", font=("Arial", 12))
label_status.pack(pady=6)

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

recording = False

# ===============================
# Safe image display
# ===============================
def show(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(
        image=Image.fromarray(rgb),
        master=root   # 🔥 FIX
    )
    panel.configure(image=img)
    panel.image = img

# ===============================
# RECORD FUNCTION
# ===============================
def start_recording():
    global recording, last_face

    if recording:
        return

    recording = True
    last_face = None
    save_path = get_next_file()

    # Countdown
    for i in [3, 2, 1]:
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, str(i), (300, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            show(frame)
        root.update_idletasks()
        time.sleep(1)

    label_status.config(text="Recording...")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(save_path, fourcc, FPS, (CROP_SIZE, CROP_SIZE))

    total_frames = FPS * RECORD_SECONDS
    frames = 0

    while frames < total_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        mouth = crop_mouth(frame)
        writer.write(mouth)
        frames += 1
        show(cv2.resize(mouth, (WIDTH, HEIGHT)))
        root.update_idletasks()

    writer.release()

    label_status.config(
        text=f"✅ Saved: {os.path.basename(save_path)} ({frames} frames)"
    )
    recording = False

# ===============================
# Key binding
# ===============================
root.bind("<space>", lambda e: start_recording())

# ===============================
# Live preview
# ===============================
def preview():
    if not recording:
        ret, frame = cap.read()
        if ret:
            show(frame)
    root.after(30, preview)

preview()
root.mainloop()
cap.release()
