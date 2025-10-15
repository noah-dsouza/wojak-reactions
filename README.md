# 🧠 Wojak Reactor  
### Real-Time Facial Expression & Hand Gesture Reactions

**Wojak Reactor** is a real-time emotion and gesture detection system that reacts with custom **Wojak images**.  
It uses **OpenCV** for webcam input and **MediaPipe** for facial and hand landmark detection.  
Built purely for fun, this project helped me explore how real-time computer vision actually works under the hood.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.11 |
| **Computer Vision** | [OpenCV](https://opencv.org/) — video capture, display, image overlays |
| **Landmark Detection** | [MediaPipe](https://developers.google.com/mediapipe) — facial & hand tracking |
| **Math & Data Handling** | NumPy — geometric calculations and array processing |
| **Runtime** | Local webcam session (no cloud or web app) |

---

## 💫 Features

### Facial Expression Detection
Detects live expressions and maps them to matching Wojak emotions:
- 😐 **Neutral**  
- 😡 **Angry**  
- 😢 **Sad**   
- 😮 **Shocked**  
- 😴 **Mouth-breather** — jaw dropped, relaxed expression  
- 🤦 **Stressed** — head in hands or squinting with tension  

### ✋ Hand Gesture Detection
Recognizes expressive gestures in real time:
- ❤️ Heart shape with both hands → *Love / Support*  
- 🤦 Hands covering face → *Stress / Frustration*  

---

## 🧠 What I’m Learning
- How **MediaPipe FaceMesh** and **Hands** models track landmarks in real time.  
- How to compute **angles, distances, and regions** to infer emotion states.  
- Integrating **OpenCV overlays** (bounding boxes, text, triggers) for live feedback.  
- Efficiently handling real-time frame updates at ~30 FPS.  
- How to make local ML pipelines feel interactive and reactive.  

---

## 🧰 Setup & Requirements

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe numpy
python3 wojak_reactor.py
