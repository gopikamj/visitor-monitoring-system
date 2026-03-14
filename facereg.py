import cv2
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("labels.pickle", "rb") as f:
    label_map = pickle.load(f)

id_to_name = {v: k for k, v in label_map.items()}
print("Loaded labels:", id_to_name)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def live_face_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Live face recognition started.")
    print("Press 'q' to quit.")

    CONFIDENCE_THRESHOLD = 49
    REQUIRED_MATCH_FRAMES = 2

    last_valid_name = None
    same_name_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)

            label_id, confidence = recognizer.predict(face_roi)
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
                final_name = current_name
            else:
                final_name = "Unknown"

            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)

            print(
                f"Predicted: {predicted_name}, "
                f"Final: {final_name}, "
                f"Confidence: {confidence:.2f}, "
                f"MatchCount: {same_name_count}"
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{final_name} ({confidence:.1f})"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Live Face Recognition - Press 'q' to exit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_face_recognition()