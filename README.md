# 🎯 ArticulateAI – Real-Time Visual Speech Recognition (Lip Reading System)

ArticulateAI is a deep learning–based **Visual Speech Recognition (VSR)** system that predicts spoken words by analyzing **lip movements from silent video streams**, without using audio signals.

This project is designed as a **real-time, end-to-end lip reading system**, combining **computer vision + deep learning + sequence modeling**, and deployed with an interactive interface.

It demonstrates how AI can interpret speech in **noisy, silent, or privacy-sensitive environments**.

---

## 🔥 Key Features

 **Real-time lip reading using webcam**
 **Audio-free speech recognition**
 **CNN + BiLSTM + Attention architecture**
 **Live prediction with confidence scores**
 **Custom dataset pipeline (record → preprocess → train)**
 **Streamlit-based web interface**
 **Optimized for **small custom datasets**

---

## 🧠 Model Architecture

The system follows a **spatio-temporal pipeline**:

### 1. CNN Encoder
- Extracts spatial features from lip regions
- Input: `64×64` grayscale frames

### 2. BiLSTM (Bidirectional LSTM)
- Learns temporal dependencies in lip motion
- Captures forward & backward context

### 3. Attention Mechanism
- Focuses on important frames
- Improves prediction accuracy

### 4. Fully Connected Layer
- Outputs final predicted word

---

## 📁 Project Structure
lipreading/
│
├── datasets/
│ └── my_dataset/
│ ├── train/
│ └── val/
│
├── models/
│ └── model.py
│
├── checkpoints/
│ ├── best_lip_model.pth
│ └── checkpoint_lip_model.pth
│
├── realtime_inference.py # 🎥 Real-time webcam prediction
├── streamlit_app.py # 🌐 Web app
├── train.py
├── classes.txt
├── requirements.txt
└── README.md

---

## 📊 Dataset

Custom dataset recorded using webcam and organized as:
dataset/
├── train/
│ ├── blue
│ ├── clean
│ ├── go
│ ├── green
│ ├── he
│ ├── hello
│ ├── place
│ ├── she
│ ├── what
│ └── yellow
│
└── val/
├── blue
├── clean
├── go
├── green
├── he
├── hello
├── place
├── she
├── what
└── yellow

- Each folder represents one **word class**
- Preprocessing includes:
  - Face detection
  - Mouth region extraction
  - Frame resizing (64×64)
  - Sequence padding (75 frames)

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Streamlit**
- **MediaPipe (for face detection)**
- **CUDA (optional GPU acceleration)**

---

## 🧪 Training Details

- Input size: `64 × 64`
- Sequence length: `75 frames`
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Framework: PyTorch

---

## 🚀 Running the Project

### ▶️ 1. Train Model
```bash
python train.py
```
2. Real-Time Inference (Webcam)
```bash
python realtime_inference.py
```
- Live prediction on webcam feed
- Displays Top-3 predictions with confidence
3. Streamlit Web App
```bash
streamlit run streamlit_app.py
```
- Upload video
- Get instant prediction
📈 Results
- Achieved high training accuracy (~99%)
- Validation accuracy up to ~90%
Performs best on:
- Frontal face
- Good lighting
- Clear lip movement
⚠️ Limitations:
- Confusion between similar lip movements (visemes)
- Performance depends on dataset size
🧪 Example Output
Top3: hello:0.76 | blue:0.23 | yellow:0.01
>>> PREDICTED: hello
🔮 Future Improvements
- Sentence-level lip reading (continuous speech)
- Transformer-based models (ViT / Video Transformers)
- Larger datasets (GRID, LRS2, LRS3)
- Multilingual support
- Deployment using ONNX / TensorRT
💡 Applications
Assistive communication (hearing-impaired users)
- Silent speech interfaces
- Defense & surveillance
- Human-computer interaction
- AI research
  
👩‍💻 Author

Ragita Kumari
B.Tech – Information Technology
Deen Dayal Upadhyaya Gorakhpur University

📜 License

For academic, research, and portfolio use only

📌 Citation
@misc{articulateai2025,
  author       = {Ragita Kumari},
  title        = {ArticulateAI: Real-Time Visual Speech Recognition Using Lip Reading},
  year         = {2025},
  note         = {CNN–BiLSTM–Attention based lip reading system}
}
