import cv2
import os
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ===============================
# Initialize Mediapipe Face Mesh
# ===============================
mp_face_mesh = mp.solutions.face_mesh

# Global face mesh (for real-time)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==========================================================
# OFFLINE: extract lips from VIDEO (unchanged logic)
# ==========================================================
def extract_lips_from_video(video_path, output_folder, target_size=(112, 112)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as fm:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                lm = results.multi_face_landmarks[0].landmark
                lip_ids = list(range(61, 88))

                xs = [int(lm[i].x * w) for i in lip_ids]
                ys = [int(lm[i].y * h) for i in lip_ids]

                x1, x2 = max(min(xs)-10, 0), min(max(xs)+10, w)
                y1, y2 = max(min(ys)-10, 0), min(max(ys)+10, h)

                lip = frame[y1:y2, x1:x2]
                if lip.size == 0:
                    continue

                lip = cv2.resize(lip, target_size)
                cv2.imwrite(
                    os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"),
                    lip
                )
                frame_count += 1

    cap.release()
    print(f"✅ Saved {frame_count} frames → {output_folder}")


# ==========================================================
# REAL-TIME: extract lip ROI from SINGLE FRAME (FIXED)
# ==========================================================
def extract_lip_roi(frame, target_size=(64, 64)):
    """
    Returns: (3, H, W) tensor-ready numpy array
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = frame.shape
    lm = results.multi_face_landmarks[0].landmark
    lip_ids = list(range(61, 88))

    xs = [int(lm[i].x * w) for i in lip_ids]
    ys = [int(lm[i].y * h) for i in lip_ids]

    x1, x2 = max(min(xs)-10, 0), min(max(xs)+10, w)
    y1, y2 = max(min(ys)-10, 0), min(max(ys)+10, h)

    lip = frame[y1:y2, x1:x2]
    if lip.size == 0:
        return None

    lip = cv2.resize(lip, target_size)
    lip = lip.astype(np.float32) / 255.0

    # 🔥 CRITICAL FIX: HWC → CHW
    lip = np.transpose(lip, (2, 0, 1))  # (3, H, W)

    return lip
