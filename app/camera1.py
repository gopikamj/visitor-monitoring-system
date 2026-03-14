import cv2
import pickle
import numpy as np
from django.utils import timezone
from django.core.files.base import ContentFile
from firebase_admin import db
from tensorflow.keras.models import load_model
from .models import BlockedVisitor, todaysvisiter


print("RUNNING APP CAMERA1.PY")

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

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")

    with open("labels.pickle", "rb") as f:
        label_map = pickle.load(f)

    id_to_name = {v: k for k, v in label_map.items()}

    emotion_model = load_model('models/emotion_model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(1)

    print("🔥 Live Monitoring Started")

    # safer recognition settings
    CONFIDENCE_THRESHOLD = 46
    REQUIRED_MATCH_FRAMES = 3

    last_valid_name = None
    same_name_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check whether frame is dark
        dark_mode = is_dark_frame(frame, threshold=80)

        # Enhance only if needed
        enhanced_frame = enhance_low_light_if_needed(frame, dark_mode)

        # Use enhanced frame for detection and recognition
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

        if dark_mode:
            cv2.putText(
                enhanced_frame,
                "Low Light Mode",
                (30, 40),
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

        for (x, y, w, h) in faces:

            # Face recognition
            face_gray = gray[y:y+h, x:x+w]

            if face_gray.size == 0:
                continue

            face_gray = cv2.resize(face_gray, (200, 200))
            label_id, confidence = recognizer.predict(face_gray)
            predicted_name = id_to_name.get(label_id, "Unknown")

            # strict recognition logic
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

            # Firebase update if blocked
            if status == "blocked":
                ref = db.reference("Device")
                ref.update({
                    "Device": "L0B0S1",
                    "Mode": "Man",
                    "Motion": "No Motion"
                })

            # Emotion detection
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            preds = emotion_model.predict(face_roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]

            # box color
            if name == "Unknown":
                box_color = (0, 0, 255)   # red
            else:
                box_color = (0, 255, 0)   # green

            # draw face box
            cv2.rectangle(enhanced_frame, (x, y), (x+w, y+h), box_color, 2)

            # show name, status, emotion, confidence
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