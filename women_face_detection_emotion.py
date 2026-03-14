import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp
from collections import deque

# =====================================================
# LOAD YOLOv8 FACE MODEL
# =====================================================
face_model = YOLO("yolov8n-face.pt")

# =====================================================
# LOAD FAIRFACE GENDER MODEL (WEIGHTS ONLY)
# =====================================================
def load_gender_model():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 18)
    state_dict = torch.load(
        "res34_fair_align_multi_7_20190809.pt",
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

gender_model = load_gender_model()
print("✅ Gender model loaded")

# =====================================================
# FAIRFACE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# EMOTION MODEL (ONNX – OpenCV)
# =====================================================
emotion_net = cv2.dnn.readNetFromONNX("emotion_model.onnx")
EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger", "fear", "disgust"]

# =====================================================
# MEDIAPIPE POSE
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# MOVEMENT HISTORY
# =====================================================
movement_history = deque(maxlen=15)
prev_landmarks = None
last_time = time.time()

# =====================================================
# GENDER CHECK
# =====================================================
def is_female(face_img):
    if face_img.shape[0] < 40 or face_img.shape[1] < 40:
        return False

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    input_tensor = transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = gender_model(input_tensor)
        gender_logits = outputs[:, 7:9]
        gender = torch.argmax(gender_logits, dim=1).item()
        return gender == 1  # Female

# =====================================================
# EMOTION DETECTION (NO TENSORFLOW)
# =====================================================
def detect_emotion(face_img):
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))

        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 255,
            size=(64, 64),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )

        emotion_net.setInput(blob)
        preds = emotion_net.forward()[0]

        idx = np.argmax(preds)
        return EMOTIONS[idx], float(preds[idx])

    except Exception:
        return None, 0.0


def is_abnormal_emotion(emotion, confidence):
    if confidence < 0.6:
        return False
    return emotion in ["fear", "anger", "sad"]

# =====================================================
# SUSPICIOUS MOVEMENT DETECTION
# =====================================================
def detect_suspicious_behavior(landmarks, prev_landmarks, dt):
    alerts = []

    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

    if lw.y < ls.y and rw.y < rs.y:
        alerts.append("Hands Raised")

    if nose.y > hip.y:
        alerts.append("Possible Fall")

    if prev_landmarks and dt > 0:
        prev_lw = np.array([
            prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
            prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        ])
        curr_lw = np.array([lw.x, lw.y])

        speed = np.linalg.norm(curr_lw - prev_lw) / dt
        movement_history.append(speed)

        if speed > 2.5:
            alerts.append("Sudden Movement")

        if len(movement_history) == movement_history.maxlen:
            if np.mean(movement_history) > 1.5:
                alerts.append("Panic / Restless")

    return alerts

# =====================================================
# VIDEO PROCESSING
# =====================================================
def process_video(source):
    global prev_landmarks, last_time

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        abnormal_emotion = False

        # ---------------- FACE DETECTION ----------------
        results = face_model(frame, conf=0.4)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                if is_female(face):
                    emotion, conf = detect_emotion(face)

                    label = "Woman"
                    color = (0, 255, 0)

                    if emotion:
                        label += f" | {emotion}"
                        if is_abnormal_emotion(emotion, conf):
                            abnormal_emotion = True
                            color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

        # ---------------- POSE DETECTION ----------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            alerts = detect_suspicious_behavior(
                pose_result.pose_landmarks.landmark,
                prev_landmarks,
                dt
            )

            prev_landmarks = pose_result.pose_landmarks.landmark

            y = 40
            for alert in alerts:
                cv2.putText(
                    frame,
                    f"⚠ {alert}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                y += 35
        else:
            prev_landmarks = None

        # ---------------- GLOBAL ALERT ----------------
        if abnormal_emotion:
            cv2.putText(
                frame,
                "🚨 ABNORMAL EMOTION DETECTED",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        cv2.imshow("Women Safety System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("1 - Webcam")
    print("2 - Video File")

    choice = input("Select option: ").strip()

    if choice == "1":
        process_video(0)
    elif choice == "2":
        path = input("Enter video path: ").strip()
        process_video(path)
    else:
        print("Invalid choice")
