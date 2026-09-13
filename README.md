# ✏️ MNIST Digit Classifier

![Python](https://img.shields.io/badge/Python-3.13.3-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.94%25-brightgreen.svg)

A deep learning web application that recognizes handwritten digits in real time. Draw any digit (0–9) on the canvas and the CNN model predicts it instantly — no upload needed.

---

## 🖼️ Demo

![App Screenshot](images/screenshot.png) 

🔗 **Live Demo:** [digit-recognition.herokuapp.com](https://digit-detective-c0fd8e943fe2.herokuapp.com)
### 🎥 Video Demo

[▶ Watch the demo](./Demo_Video.mov)

---

## ⚙️ How It Works

1. User draws a digit on an HTML canvas
2. The canvas image is captured and sent to a Flask backend via a POST request
3. The image is preprocessed — cropped, centered, and resized to 28×28 pixels
4. The preprocessed image is passed through a trained CNN model
5. The predicted digit is returned and displayed instantly

---

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) trained on the MNIST dataset.

| Layer | Details |
|-------|---------|
| Conv1 | 32 filters, 3×3 kernel, ReLU + MaxPool |
| Conv2 | 64 filters, 3×3 kernel, ReLU + MaxPool |
| FC1 | 64×7×7 → 128 neurons, ReLU |
| FC2 | 128 → 10 neurons (output) |

---

## 📊 Results

| Model | Architecture | Test Accuracy |
|-------|-------------|---------------|
| Baseline | Fully Connected Network (FCN) | 96.75% |
| **Final** | **CNN** | **98.94%** |

The CNN achieves roughly 4x fewer mistakes compared to the baseline FCN.

---

## 🛠️ Tech Stack

- **Model:** PyTorch, torchvision
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript (Canvas API)
- **Deployment:** Railway.app
- **Dataset:** MNIST (60,000 training images, 10,000 test images)

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Flask app**
```bash
python app.py
```

**4. Open in browser**
