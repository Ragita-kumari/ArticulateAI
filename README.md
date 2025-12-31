# ArticulateAI – Visual Speech Recognition (Lip Reading System)

ArticulateAI is a deep learning–based **visual speech recognition system** that predicts spoken words by analyzing **lip movements from silent video clips**, without relying on audio signals. The project is designed as a lightweight, end-to-end lip-reading solution trained on a **custom mouth-cropped dataset** and deployed using an interactive **Streamlit web interface**.

This system demonstrates how computer vision and deep learning can be used to interpret speech in environments where audio is unavailable, noisy, or intentionally muted.

---
##  Project Structure
```
lipreading/
│
├── datasets/
│   └── my_dataset/
│       ├── hello/
│       ├── go/
│       ├── what/
│       └── ...
│
├── models/
│   └── model.py
│
├── checkpoints/
│   ├── best_lip_model.pth
│   └── checkpoint_lip_model.pth
│
├── train.py
├── app.py
├── classes.txt
├── requirements.txt
└── README.md
```

---

## 🚀 Project Highlights

- **Audio-free speech recognition** using lip movements only  
- **CNN + BiLSTM + Attention** based deep learning architecture  
- **Custom dataset creation** using webcam recording and mouth-region cropping  
- **Real-time prediction** through a Streamlit web application  
- Lightweight and suitable for **small, custom datasets**

---

## 🧠 System Architecture

The model follows a **spatio-temporal learning pipeline**:

1. **CNN Encoder**  
   - Extracts spatial features such as lip contours and mouth shape from each video frame.

2. **BiLSTM Network**  
   - Captures forward and backward temporal dependencies in lip movements across frames.

3. **Attention Mechanism**  
   - Focuses on the most informative frames for accurate word prediction.

4. **Classifier**  
   - Predicts the final spoken word from a predefined vocabulary.

---

## 📁 Dataset Structure

A custom dataset is used, recorded via webcam and organized as:
```
D:\DESKTOP\LIPREADING\DATASET
├───train
│   ├───blue
│   ├───clean
│   ├───go
│   ├───green
│   ├───he
│   ├───hello
│   ├───him
│   ├───place
│   ├───she
│   └───what
└───val
    ├───blue
    ├───clean
    ├───go
    ├───green
    ├───he
    ├───hello
    ├───him
    ├───place
    ├───she
    └───what
```

Each folder represents one **spoken word class**, containing multiple silent video samples.

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch** – Deep learning framework
- **OpenCV** – Video processing
- **NumPy** – Numerical operations
- **Tkinter** – Dataset recording GUI
- **Streamlit** – Web-based deployment
- **CUDA** – GPU acceleration (optional)

---

## 🧪 Training Details

- Input size: `64 × 64` mouth-cropped frames  
- Sequence length: `75 frames`  
- Loss function: **Cross-Entropy Loss**  
- Optimizer: **Adam**  
- Evaluation: Train/Validation accuracy and loss curves  

---

## 🌐 Streamlit Demo Features

- Upload silent video clips (`.mp4`, `.avi`, `.mov`)
- Automatic preprocessing and frame extraction
- Real-time word prediction
- User-friendly and interactive UI

---

## 📊 Experimental Results

- Achieved **moderate accuracy** on a small custom dataset
- Performs best on **clear, frontal, well-lit videos**
- Common confusions occur between visually similar words (visemes)
- Demonstrates strong potential for improvement with more data and augmentation

---

## Future Enhancements

- Sentence-level lip reading instead of isolated words  
- Larger and more diverse datasets  
- Transformer-based visual speech models  
- Data augmentation for robustness  
- Support for regional and multilingual lip reading  

---

## Applications

- Assistive communication for hearing-impaired users  
- Silent speech interfaces  
- Privacy-preserving human–computer interaction  
- Surveillance and security analysis  
- Research and educational demonstrations  

---

##  Author

**Ragita Kumari**  
Information Technology | AI & Machine Learning  
Deen Dayal Upadhyaya Gorakhpur University  

---

## License

This project is intended for **academic and research and portfolio purposes**.  

---
## Citation
@misc{articulateai2025,
  author       = {Ragita Kumari},
  title        = {ArticulateAI: Visual Speech Recognition Using Lip Reading},
  year         = {2025},
  howpublished = {\url{https://github.com/your-username/ArticulateAI}},
  note         = {CNN–BiLSTM–Attention based lip reading system}
}


*If you find this project useful, please give it a star on GitHub!* 


