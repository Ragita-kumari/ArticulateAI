import torch
from model import LipReadingModel  # ✅ same model as used during training
from datasets.grid_dataset import GRIDDataset

def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Create same model as in training
    model = LipReadingModel(num_classes=10).to(device)
    model.load_state_dict(torch.load("lip_model.pth", map_location=device))
    model.eval()

    print("✅ Model loaded successfully!")

    # ✅ Example: test one sample
    dataset = GRIDDataset(root_dir="datasets")
    video, label = dataset[0]  # take first video
    video = video.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        output = model(video)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"Predicted class index: {predicted_class}")
    print(f"Actual label index: {label.item()}")

if __name__ == "__main__":
    predict()
