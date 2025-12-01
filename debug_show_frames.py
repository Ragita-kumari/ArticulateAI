# debug_show_frames.py
import cv2, os, sys
import numpy as np

def show_and_save(video_path, out_dir="debug_frames"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64,64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    print("Total frames read:", len(frames))
    for i, f in enumerate(frames[:12]):
        p = os.path.join(out_dir, f"frame_{i:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        print("Saved", p)
    if len(frames) == 0:
        print("No frames extracted — video read failed or codec unsupported.")
    return frames

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_show_frames.py path/to/video.mp4")
        sys.exit(1)
    show_and_save(sys.argv[1])
