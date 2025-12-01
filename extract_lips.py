import cv2
import os
import mediapipe as mp
from tqdm import tqdm

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

def extract_lips_from_video(video_path, output_folder, target_size=(112, 112)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_face_landmarks[0].landmark

                # Lip landmark indices (Mediapipe FaceMesh specification)
                lip_indices = list(range(61, 88))

                x_coords = [int(landmarks[i].x * w) for i in lip_indices]
                y_coords = [int(landmarks[i].y * h) for i in lip_indices]

                x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
                y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

                # Crop and resize lips
                lip_crop = frame[y_min:y_max, x_min:x_max]
                lip_crop = cv2.resize(lip_crop, target_size)

                # Save frame
                frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, lip_crop)
                frame_count += 1

    cap.release()
    print(f"✅ Saved {frame_count} lip frames from {video_path} → {output_folder}")


def process_all_videos(input_dir, output_dir):
    for label in tqdm(os.listdir(input_dir), desc="Processing classes"):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        for video in os.listdir(label_path):
            if not video.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(label_path, video)
            video_name = os.path.splitext(video)[0]
            output_folder = os.path.join(output_dir, label, video_name)

            extract_lips_from_video(video_path, output_folder)
if __name__ == "__main__":
    input_dir = "dataset"          # folder containing your raw videos
    output_dir = "dataset_lips"    # where cropped lips will be saved
    process_all_videos(input_dir, output_dir)