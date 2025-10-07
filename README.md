# Wojak Reactor

> Real-time **facial expression** and **hand-gesture** detector that reacts with custom **Wojak emotion images**.  
> Built purely for fun — runs locally on your webcam using **Python, OpenCV, and MediaPipe**.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.11 |
| **Computer Vision** | [OpenCV](https://opencv.org/) |
| **Machine Learning Landmarks** | [MediaPipe](https://developers.google.com/mediapipe) |
| **Math + Data Handling** | NumPy |
| **Environment** | macOS ARM (M3), Virtualenv |
| **Version Control** | Git + GitHub |
| **IDE** | VS Code |
| **Host** | Local runtime (not web-deployed) |

---

## 💫 Features

- Detects **facial expressions** in real time  
  - 😐 Neutral  
  - 😀 Happy / Smiling  
  - 😂 Laughing  
  - 😮 Shocked  
  - 😡 Angry  
  - 😢 Sad  
  - 😞 Disappointed  
  - 😔 Depressed  
  - 😴 Mouth-open / “mouth-breather”  

- Detects **hand gestures**  
  - ❤️ Heart shape or Korean “finger heart” → *Love / Support*  
  - 🤦 Hands on face or head → *Stressed*  

---

## 🧰 Requirements

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe numpy
