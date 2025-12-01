import cv2
import os
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk

# -------------------------------
# SETTINGS (change class name only)
# -------------------------------
DATASET_ROOT = "datasets/my_dataset"
CLASS_NAME = "hello"       # change this word to record another class
RECORD_SECONDS = 2
FPS = 20
CROP_SIZE = 64
WIDTH, HEIGHT = 640, 480
# -------------------------------


# Ensure output directory
class_folder = os.path.join(DATASET_ROOT, CLASS_NAME)
os.makedirs(class_folder, exist_ok=True)


# Auto filename
def get_next_file():
    existing = [f for f in os.listdir(class_folder) if f.endswith(".avi")]
    next_id = len(existing) + 1
    return os.path.join(class_folder, f"{CLASS_NAME}_{next_id}.avi")


save_path = get_next_file()


# -------------------------------
# Face Detector
# -------------------------------
haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(haar_path)


def detect_face(frame):
    """Returns face bbox or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    return faces[0]  # x,y,w,h


def crop_mouth(frame):
    """Crops mouth from face OR fallback to lower-face center."""
    h, w = frame.shape[:2]
    face = detect_face(frame)

    if face is not None:
        x, y, fw, fh = face
        # lower half of face (mouth region)
        mx = x
        my = y + int(fh * 0.45)
        mw = fw
        mh = int(fh * 0.55)
        mx, my = max(0, mx), max(0, my)
        crop = frame[my:my + mh, mx:mx + mw]
    else:
        # fallback lower center crop
        cx = w // 2
        cy = int(h * 0.6)
        size = min(w, h) // 3
        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(w, cx + size)
        y2 = min(h, cy + size)
        crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        crop = frame[h//4: 3*h//4, w//4: 3*w//4]

    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    return crop


# -------------------------------
# Tkinter UI
# -------------------------------
root = tk.Tk()
root.title("Lip Reading Dataset Recorder (SPACE to Record)")
root.geometry("700x560")

label_title = tk.Label(root, text="📸 Lip/Mouth Recorder", font=("Arial", 16))
label_title.pack(pady=6)

panel = tk.Label(root)  # webcam preview
panel.pack()

label_info = tk.Label(root, text=f"Class: {CLASS_NAME}", font=("Arial", 12))
label_info.pack(pady=4)

label_next = tk.Label(root, text=f"Next File: {os.path.basename(save_path)}", font=("Arial", 10))
label_next.pack(pady=4)

label_status = tk.Label(root, text="Press SPACE to start recording", font=("Arial", 12))
label_status.pack(pady=6)


# -------------------------------
# Webcam Setup
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


recording = False


# -------------------------------
# RECORD FUNCTION
# -------------------------------
def record_clip():
    global recording, save_path
    recording = True

    # Countdown
    for num in [3, 2, 1]:
        ret, frame = cap.read()
        if not ret:
            continue

        disp = frame.copy()
        cv2.putText(disp, f"{num}", (WIDTH//2 - 20, HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        panel.configure(image=img)
        panel.imgtk = img
        root.update()
        time.sleep(1)

    label_status.config(text="🎬 Recording...")
    root.update()

    # GUARANTEED WORKING CODEC ON WINDOWS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(save_path, fourcc, FPS, (CROP_SIZE, CROP_SIZE))

    if not writer.isOpened():
        print("❌ ERROR: Writer not opened:", save_path)
        label_status.config(text="❌ Save failed: writer not opened")
        recording = False
        return

    start = time.time()
    frames_written = 0

    while time.time() - start < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue

        mouth = crop_mouth(frame)  # mouth crop (64x64)
        writer.write(mouth)         # BGR → correct format
        frames_written += 1

        # Show mouth during recording
        disp = cv2.resize(mouth, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.putText(disp, "Recording...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        panel.configure(image=img)
        panel.imgtk = img
        root.update()

    writer.release()
    label_status.config(text=f"✅ Saved: {save_path} ({frames_written} frames)")

    # Prepare next filename
    save_path = get_next_file()
    label_next.config(text=f"Next File: {os.path.basename(save_path)}")
    recording = False


# -------------------------------
# SPACE KEY EVENT
# -------------------------------
def key_pressed(event):
    if event.keysym == "space" and not recording:
        threading.Thread(target=record_clip, daemon=True).start()

root.bind("<KeyPress>", key_pressed)


# -------------------------------
# LIVE PREVIEW LOOP
# -------------------------------
def update_preview():
    if not recording:
        ret, frame = cap.read()
        if ret:
            # draw face box just for preview
            face = detect_face(frame)
            preview = frame.copy()

            if face is not None:
                x, y, fw, fh = face
                cv2.rectangle(preview, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
            panel.configure(image=img)
            panel.imgtk = img

    panel.after(30, update_preview)

update_preview()
root.mainloop()
cap.release()
