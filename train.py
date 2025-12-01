import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets.custom_dataset import CustomLipDataset
from models.model import LipReadingModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/checkpoint_lip_model.pth"
BEST_MODEL_PATH = "checkpoints/best_lip_model.pth"
os.makedirs("checkpoints", exist_ok=True)

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
TOTAL_EPOCHS = 60
ADDITIONAL_IF_PAST = 10  # if checkpoint epoch >= TOTAL_EPOCHS, continue by this many more

# ----------------------------------------------------------------------
# Collate function (batch sync)
# ----------------------------------------------------------------------
def collate_fn(batch):
    videos, labels = zip(*batch)
    min_len = min(v.shape[0] for v in videos)
    videos = [v[:min_len] for v in videos]
    videos = torch.stack(videos)  # [B,T,C,H,W]
    labels = torch.tensor(labels)
    return videos, labels

# ----------------------------------------------------------------------
# Training function
# ----------------------------------------------------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # --------------------------------------------------------------
    # Load Custom Dataset
    # --------------------------------------------------------------
    DATA_ROOT = "datasets/my_dataset"

    classes = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])

    if len(classes) == 0:
        raise RuntimeError(f"No class folders found under {DATA_ROOT}")

    print("📝 Discovered classes:", classes)

    dataset = CustomLipDataset(
        root_dir=DATA_ROOT,
        classes=classes,
        max_frames=75,
        augment=True
    )

    num_classes = len(classes)
    print(f"📚 Found {num_classes} word classes")

    total_size = len(dataset)
    if total_size == 0:
        raise RuntimeError("No video samples found in dataset.")
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # --------------------------------------------------------------
    # Model, Loss, Optimizer
    # --------------------------------------------------------------
    model = LipReadingModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # --------------------------------------------------------------
    # Checkpoint Resume (also restore training history)
    # --------------------------------------------------------------
    start_epoch = 0
    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    global TOTAL_EPOCHS
    if os.path.exists(CHECKPOINT_PATH):
        print("🔄 Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception as e:
            print("⚠️ Warning: could not fully load optimizer/scheduler state:", e)

        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

        train_losses = checkpoint.get("train_losses", [])
        val_losses   = checkpoint.get("val_losses", [])
        train_accs   = checkpoint.get("train_accs", [])
        val_accs     = checkpoint.get("val_accs", [])

        print(f"🔁 Resumed from epoch {start_epoch} (checkpoint epoch was {checkpoint.get('epoch', 'N/A')})")
        print(f"🔁 Previously recorded {len(train_losses)} train epochs and {len(val_losses)} val epochs")

        if start_epoch >= TOTAL_EPOCHS:
            old_total = TOTAL_EPOCHS
            TOTAL_EPOCHS = start_epoch + ADDITIONAL_IF_PAST
            print(f"⚠️ TOTAL_EPOCHS ({old_total}) <= start_epoch ({start_epoch}). Extending TOTAL_EPOCHS -> {TOTAL_EPOCHS}")

    # --------------------------------------------------------------
    # Stats and training loop bounds
    # --------------------------------------------------------------
    num_epochs = TOTAL_EPOCHS
    print(f"🔢 Will train from epoch {start_epoch} to {num_epochs-1} (inclusive).")

    # --------------------------------------------------------------
    # Main Training Loop
    # --------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs):

        # ------------------ TRAINING ------------------
        model.train()
        run_loss = 0.0
        correct = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for videos, labels in train_pbar:
            # videos: [B, T, C, H, W], labels: list/1D tensor of length B
            # Move to device
            videos = videos.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)

            # --- Ensure channel count matches model expectation:
            # If dataset gives C==1 but model expects 3, repeat channels.
            # We can't inspect model first conv reliably here, but repeating to 3 is safe if model was trained with 3 channels.
            if videos.dim() == 5 and videos.size(2) == 1:
                videos = videos.repeat(1, 1, 3, 1, 1)  # (B,T,3,H,W)

            # Forward
            outputs = model(videos)  # model must accept (B,T,C,H,W)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * videos.size(0)  # accumulate sum
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            train_pbar.set_postfix(loss=loss.item())

        # compute averages
        train_loss = run_loss / (train_size if train_size > 0 else 1)
        train_acc = correct / (train_size if train_size > 0 else 1)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        scheduler.step()

        # ------------------ VALIDATION ------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)

                if videos.dim() == 5 and videos.size(2) == 1:
                    videos = videos.repeat(1, 1, 3, 1, 1)

                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * videos.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / (val_size if val_size > 0 else 1)
        val_acc = val_correct / (val_size if val_size > 0 else 1)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"📘 Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ------------------ SAVE CHECKPOINT ------------------
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        }, CHECKPOINT_PATH)

        # ------------------ SAVE BEST MODEL ------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 Updated Best Model: Val Acc = {val_acc:.4f}")

    # --------------------------------------------------------------
    # Save Final Model (also save final checkpoint with history)
    # --------------------------------------------------------------
    torch.save(model.state_dict(), "lip_model_final.pth")
    torch.save({
        "epoch": num_epochs - 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }, CHECKPOINT_PATH)
    print("🎉 Training completed — saved lip_model_final.pth and checkpoint with training history")

    # --------------------------------------------------------------
    # Plot Training Curves (robust)
    # --------------------------------------------------------------
    print("DEBUG: train_losses (len={}): {}".format(len(train_losses), train_losses[-10:]))
    print("DEBUG: val_losses   (len={}): {}".format(len(val_losses), val_losses[-10:]))
    print("DEBUG: train_accs   (len={}): {}".format(len(train_accs), train_accs[-10:]))
    print("DEBUG: val_accs     (len={}): {}".format(len(val_accs), val_accs[-10:]))

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    if len(train_losses) > 0:
        plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    if len(val_losses) > 0:
        plt.plot(epochs, val_losses, marker='o', label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if len(train_losses) + len(val_losses) > 0:
        plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    if len(train_accs) > 0:
        plt.plot(epochs, train_accs, marker='o', label="Train Acc")
    if len(val_accs) > 0:
        plt.plot(epochs, val_accs, marker='o', label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    if len(train_accs) + len(val_accs) > 0:
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
