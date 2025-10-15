# 🧠 Wojak Reactor  
### Real-Time Facial Expression & Hand Gesture Recognition with OpenCV + MediaPipe

**Wojak Reactor** is a small project where I explored real-time **emotion** and **gesture** detection.  
It uses **MediaPipe FaceMesh** and **Hands** for facial and hand landmark tracking, paired with **OpenCV** overlays and simple geometric logic
No AI model, no fancy training — just math, curiosity, and a bit of chaos to learn how real-time computer vision actually works.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.11 |
| **Computer Vision** | [OpenCV](https://opencv.org/) — webcam input, live overlays, and alpha blending |
| **Landmark Detection** | [MediaPipe](https://developers.google.com/mediapipe) — facial & hand landmark tracking |
| **Math / Geometry** | NumPy + Python `math` — distance, angles, and feature ratios for expression logic |
| **Runtime** | Local webcam session (runs offline) |

---

## 💫 Features

### Facial Expression Detection  
Detects facial states in real time using geometric landmark analysis:
- 😐 **Neutral** 
- 🤦 **Stressed**  
- 😮 **Shocked**    
- 😢 **Sad** 
- 😡 **Angry** 

### ✋ Hand Gesture Detection  
Interprets simple hand gestures using MediaPipe landmarks:
- ❤️ Heart-shaped hands
- 🤦 Hands on face or above head 

---

## 🧠 How It Works
- **Facial landmarks:** Distances between lips, eyes, brows, and face height determine emotion.  
- **Hand landmarks:** Thumb–index proximity and hand–face distance classify gestures.  
- **Temporal smoothing:** Recent detections are stored and majority-voted to avoid flicker.  
- **OpenCV overlays:** Trigger images are alpha-blended on the live webcam feed.  

---

## 🧰 Setup & Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe numpy
python3 wojak_reactor.py


