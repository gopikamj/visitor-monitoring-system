import cv2
import pickle
import numpy as np
from django.utils import timezone
from django.core.files.base import ContentFile
from firebase_admin import db
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import torch
import torch.serialization
from .models import BlockedVisitor, todaysvisiter


print("RUNNING CAMERA(2).PY")

# Allow YOLO loading in PyTorch 2.6+
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass


def is_dark_frame(frame, threshold=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold


def enhance_low_light_if_needed(frame, dark_mode):
    if not dark_mode:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_frame


def start_live_monitoring():

    print("⏳ Initializing face recognition and emotion detection models...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")

    with open("labels.pickle", "rb") as f:
        label_map = pickle.load(f)

    id_to_name = {v: k for k, v in label_map.items()}

    emotion_model = load_model('models/emotion_model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    print("⏳ Initializing weapon detection model...")
    weapon_model = None

    try:
        weapon_model = YOLO(r"D:\bu Visitor Monitoring System\project\weapon_detection_test\best.pt")
        print("✅ Weapon Detection Model Loaded Successfully")
        print("Model classes:", weapon_model.names)
    except Exception as e:
        print(f"⚠️ Warning: Could not load weapon detection model: {e}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)

    print("🔥 Live Monitoring Started")

    CONFIDENCE_THRESHOLD = 46
    REQUIRED_MATCH_FRAMES = 3

    last_valid_name = None
    same_name_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if frame is dark
        dark_mode = is_dark_frame(frame, threshold=80)

        # Enhance only if needed
        enhanced_frame = enhance_low_light_if_needed(frame, dark_mode)

        # Use enhanced frame for further processing
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

        if dark_mode:
            cv2.putText(
                enhanced_frame,
                "Low Light Mode",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(70, 70)
        )

        # ===================================
        # WEAPON DETECTION
        # ===================================
        weapon_detected = False

        if weapon_model is not None:
            try:
                results = weapon_model(enhanced_frame, imgsz=640, conf=0.4)

                for r in results:
                    if r.boxes is None:
                        continue

                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = weapon_model.names[cls]

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        weapon_detected = True

                        print(f"⚠️ WEAPON ALERT: {label} ({conf:.2f})")

                        cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            enhanced_frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

            except Exception as e:
                print("⚠️ Weapon detection error:", e)

        if weapon_detected:
            cv2.putText(
                enhanced_frame,
                "WEAPON DETECTED - ALERT",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        # ===================================
        # FACE RECOGNITION + EMOTION
        # ===================================
        for (x, y, w, h) in faces:

            face_gray = gray[y:y+h, x:x+w]

            if face_gray.size == 0:
                continue

            face_gray = cv2.resize(face_gray, (200, 200))
            label_id, confidence = recognizer.predict(face_gray)
            predicted_name = id_to_name.get(label_id, "Unknown")

            if confidence <= CONFIDENCE_THRESHOLD:
                current_name = predicted_name

                if current_name == last_valid_name:
                    same_name_count += 1
                else:
                    last_valid_name = current_name
                    same_name_count = 1
            else:
                current_name = "Unknown"
                same_name_count = 0

            if current_name != "Unknown" and same_name_count >= REQUIRED_MATCH_FRAMES:
                name = current_name
                is_blocked = BlockedVisitor.objects.filter(name=name).exists()
                status = "blocked" if is_blocked else "allowed"
            else:
                name = "Unknown"
                status = "unknown"

            print(
                f"Predicted: {predicted_name}, Final: {name}, "
                f"Confidence: {confidence:.2f}, MatchCount: {same_name_count}, Status: {status}"
            )

            if status == "blocked":
                ref = db.reference("Device")
                ref.update({
                    "Device": "L0B0S1",
                    "Mode": "Man",
                    "Motion": "No Motion"
                })

            face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            preds = emotion_model.predict(face_roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]

            if name == "Unknown":
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(enhanced_frame, (x, y), (x+w, y+h), box_color, 2)

            cv2.putText(
                enhanced_frame,
                f"{name} ({status}) - {emotion} [{confidence:.1f}]",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Optional DB save
            """
            face_color = enhanced_frame[y:y+h, x:x+w]
            success, buffer = cv2.imencode('.jpg', face_color)

            if success:
                visitor = todaysvisiter(
                    visitername=name,
                    status=status,
                    emotion=emotion,
                    dateofvisit=timezone.now()
                )

                visitor.image.save(
                    f"{name}.jpg",
                    ContentFile(buffer.tobytes()),
                    save=True
                )
            """

        cv2.imshow("Live Monitoring", enhanced_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()