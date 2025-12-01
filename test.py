# test_predict_auto.py
import os
import sys
import torch
import cv2
import numpy as np

from models.model import LipReadingModel

SEARCH_DIRS = ["dataset", "datasets"]
CKPT_CANDIDATES = [
    "checkpoints/best_lip_model.pth",
    "checkpoints/lip_model_final.pth",
    "lip_model_final.pth",
    "lip_model.pth",
    "checkpoints/best_model.pth"
]

def find_sample(name_keyword="clean"):
    """Search recursively in SEARCH_DIRS and return first matching video path or None."""
    exts = (".mp4", ".avi", ".mov", ".mpg")
    for base in SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            # prefer folder named exactly (e.g., dataset/train/clean/...)
            parts = root.replace("\\", "/").split("/")
            if any(p.lower() == name_keyword.lower() for p in parts):
                for f in files:
                    if f.lower().endswith(exts):
                        return os.path.join(root, f)
        # fallback: filename contains keyword
        for root, dirs, files in os.walk(base):
            for f in files:
                if name_keyword.lower() in f.lower() and f.lower().endswith(exts):
                    return os.path.join(root, f)
    return None

def preprocess_video(path, max_frames=75):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame = cv2.resize(frame, (64,64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            continue
        frames.append(frame.astype("float32")/255.0)
    cap.release()
    if len(frames) == 0:
        return None
    if len(frames) < max_frames:
        pad = [np.zeros((64,64,3), dtype="float32")] * (max_frames - len(frames))
        frames = frames + pad
    else:
        frames = frames[:max_frames]
    arr = np.stack(frames, 0)
    tensor = torch.tensor(arr).permute(0,3,1,2).unsqueeze(0)  # (1, T, C, H, W)
    return tensor

def pick_checkpoint():
    for p in CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def try_strict_load(model, ckpt):
    try:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        return True, ""
    except Exception as e:
        return False, str(e)

def try_relaxed_load_with_summary(model, ckpt):
    # load checkpoint dict
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip module.
    sd2 = {}
    for k,v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        sd2[nk] = v
    model_sd = model.state_dict()
    matched = 0
    mapped_keys = []
    for k_ck, v_ck in sd2.items():
        if k_ck in model_sd and model_sd[k_ck].shape == v_ck.shape:
            model_sd[k_ck] = v_ck
            matched += 1
            mapped_keys.append(k_ck)
    model.load_state_dict(model_sd)  # now should load
    total_ck = len(sd2)
    return matched, total_ck, mapped_keys

def main():
    print("Working dir:", os.getcwd())
    sample = find_sample("clean")
    if sample is None:
        print("No 'clean' sample found in dataset/ or datasets/. Run 'ls dataset' and 'ls datasets' to inspect.")
        sys.exit(1)
    print("Using sample:", sample)

    inp = preprocess_video(sample)
    if inp is None:
        print("Could not read frames from sample (codec issue?). Try transcoding with ffmpeg.")
        sys.exit(1)
    print("Read frames -> tensor shape:", tuple(inp.shape))  # (1,T,C,H,W)

    # load classes
    if not os.path.exists("classes.txt"):
        print("classes.txt not found")
        sys.exit(1)
    with open("classes.txt","r",encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    print("Loaded classes:", classes)

    ckpt = pick_checkpoint()
    if ckpt is None:
        print("No checkpoint found among candidates:", CKPT_CANDIDATES)
        sys.exit(1)
    print("Found checkpoint:", ckpt)

    model = LipReadingModel(num_classes=len(classes))
    print("Attempting strict load...")
    ok, err = try_strict_load(model, ckpt)
    if ok:
        print("Strict load OK.")
    else:
        print("Strict load FAILED. Error:", err)
        print("Attempting relaxed/mapped load (match shapes)...")
        matched, total_ck, mapped_keys = try_relaxed_load_with_summary(model, ckpt)
        print(f"Mapped {matched}/{total_ck} checkpoint tensors by exact-name+shape into model.")
        if matched == 0:
            print("No matching params found — likely architecture mismatch. You must use the exact model.py used during training.")
        else:
            print("Proceeding with partially-loaded model (some params random).")

    model.eval()
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    top3 = probs.argsort()[-3:][::-1]
    print("Top-3 predictions (label, prob):")
    for i in top3:
        idx = int(i)
        label = classes[idx] if idx < len(classes) else str(idx)
        print(f" - {label}: {probs[idx]:.4f}")

if __name__ == "__main__":
    main()
