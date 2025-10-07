
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import os

# Mediapipe
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.6,
                             min_tracking_confidence=0.6)

hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# Image utils 
def load_img(path, size=(220, 220)):
    full_path = os.path.join("images", path)
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[WARN] Could not load {full_path}. Using red placeholder.")
        img = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        img[:, :, 2] = 255  
        img[:, :, 3] = 255  

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def overlay_with_alpha(frame_bgr, overlay_bgra, x, y):
    """Alpha-blend BGRA overlay onto BGR frame at (x, y)."""
    h, w = overlay_bgra.shape[:2]
    H, W = frame_bgr.shape[:2]
    if x >= H or y >= W:
        return frame_bgr
    x2, y2 = min(x + h, H), min(y + w, W)
    oh, ow = x2 - x, y2 - y
    if oh <= 0 or ow <= 0:
        return frame_bgr

    roi = frame_bgr[x:x2, y:y2]
    overlay_roi = overlay_bgra[:oh, :ow]
    alpha = overlay_roi[:, :, 3:4] / 255.0
    roi[:] = (alpha * overlay_roi[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    return frame_bgr

#  Load reaction images 
reactions = {
    "neutral":      load_img("neutral.jpg"),
    "happy":        load_img("happy.jpg"),
    "sad":          load_img("sad.png"),
    "angry":        load_img("angry.jpg"),
    "depressed":    load_img("depressed.jpg"),
    "shocked":      load_img("shocked.jpg"),
    "mouth":        load_img("mouth-open.jpg"),
    "stressed":     load_img("stressed.jpg"),
    "love":         load_img("heart.png"),
    "laughing":     load_img("laughing.webp"),
    "disappointed": load_img("Dissapointed.jpg"),
}

# Geometry 
def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def np_pt(lm, w, h): return np.array([lm.x * w, lm.y * h], dtype=np.float32)

# Face detection 
def detect_face_reaction(landmarks, w, h):
    lm = landmarks
    left_eye, right_eye = np_pt(lm[33], w, h), np_pt(lm[263], w, h)
    eye_width = dist(left_eye, right_eye)
    face_top, face_bottom = np_pt(lm[10], w, h), np_pt(lm[152], w, h)
    face_h = max(dist(face_top, face_bottom), 1.0)

    top_lip, bottom_lip = np_pt(lm[13], w, h), np_pt(lm[14], w, h)
    left_mouth, right_mouth = np_pt(lm[61], w, h), np_pt(lm[291], w, h)
    mouth_center = (left_mouth + right_mouth) / 2

    mouth_open = dist(top_lip, bottom_lip) / face_h
    mouth_width = dist(left_mouth, right_mouth) / eye_width

    upper_lid_L, lower_lid_L = np_pt(lm[159], w, h), np_pt(lm[145], w, h)
    upper_lid_R, lower_lid_R = np_pt(lm[386], w, h), np_pt(lm[374], w, h)
    eye_open = (dist(upper_lid_L, lower_lid_L) + dist(upper_lid_R, lower_lid_R)) / (2 * face_h)

    brow_L, brow_R = np_pt(lm[105], w, h), np_pt(lm[334], w, h)
    brow_gap = (dist(brow_L, left_eye) + dist(brow_R, right_eye)) / (2 * face_h)

    corners_drop = ((left_mouth[1] - mouth_center[1]) + (right_mouth[1] - mouth_center[1])) / (2 * face_h)
    nose, eyes_mid_y = np_pt(lm[1], w, h), (left_eye[1] + right_eye[1]) / 2
    head_down = (nose[1] - eyes_mid_y) / face_h

    # Expression rules 
    if mouth_open > 0.075 and eye_open < 0.028:
        return "laughing"
    if mouth_open > 0.09 or brow_gap > 0.125:
        return "shocked"
    if mouth_open > 0.065:
        return "mouth"
    if brow_gap < 0.095 and mouth_open < 0.035:
        return "angry"
    if mouth_width > 0.86 and mouth_open < 0.06:
        return "happy"
    if corners_drop > 0.012 and mouth_open < 0.055:
        return "sad"
    if head_down > 0.035 and corners_drop > 0.004 and mouth_open < 0.06:
        return "disappointed"
    if eye_open < 0.022 and corners_drop > 0.0:
        return "depressed"
    return "neutral"

# Hand gesture detection 
def palm_center(hand_pts):
    idx = [0,1,5,9,13,17]
    return np.mean(np.array([hand_pts[i] for i in idx]), axis=0)

def finger_heart_single(hand_pts, w):
    thumb, index = hand_pts[4], hand_pts[8]
    return dist(thumb, index)/w < 0.035

def two_hand_heart(h1, h2, w):
    return dist(h1[8], h2[8])/w < 0.15 and dist(h1[4], h2[4])/w < 0.18

def hand_near_face(hand_pts, nose, h):
    wrist, palm = hand_pts[0], palm_center(hand_pts)
    return dist(wrist, nose)/h < 0.25 or dist(palm, nose)/h < 0.22

def hand_on_head(hand_pts, eyes_y):
    return palm_center(hand_pts)[1] < eyes_y

def detect_hand_reaction(all_hands, face_lms, w, h):
    if not all_hands:
        return None
    if face_lms:
        nose = np_pt(face_lms[1], w, h)
        eyes_y = (np_pt(face_lms[33], w, h)[1] + np_pt(face_lms[263], w, h)[1])/2
    else:
        nose, eyes_y = None, None

    if len(all_hands) >= 2 and two_hand_heart(all_hands[0], all_hands[1], w):
        return "love"
    for pts in all_hands:
        if finger_heart_single(pts, w):
            return "love"
    if nose is not None:
        for pts in all_hands:
            if hand_near_face(pts, nose, h) or hand_on_head(pts, eyes_y):
                return "stressed"
    return None

# Smoothing 
recent = deque(maxlen=6)
def smooth_vote(val):
    recent.append(val)
    vals, counts = np.unique(np.array(recent), return_counts=True)
    return vals[np.argmax(counts)]

# Main loop 
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res, hands_res = face_mesh.process(rgb), hands.process(rgb)
    reaction = "neutral"

    face_lms = None
    if face_res.multi_face_landmarks:
        face_lms = face_res.multi_face_landmarks[0].landmark
        reaction = detect_face_reaction(face_lms, W, H)

    hand_pts_all = []
    if hands_res.multi_hand_landmarks:
        for hand in hands_res.multi_hand_landmarks:
            hand_pts_all.append([(lm.x * W, lm.y * H) for lm in hand.landmark])

    hand_result = detect_hand_reaction(hand_pts_all, face_lms, W, H)
    if hand_result:
        reaction = hand_result

    reaction = smooth_vote(reaction)
    frame = overlay_with_alpha(frame, reactions[reaction], 30, 30)

    cv2.putText(frame, f"Reaction: {reaction}", (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)

    cv2.imshow("Wojak Reactor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()