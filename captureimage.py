import cv2
import os

def capture_face_images(label_name, num_images=None):

    # Create folder for saving faces
    save_dir = os.path.join("faces", label_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    print(f"Capturing faces for '{label_name}'")
    print("Press 'c' to capture face, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error grabbing frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        # Draw rectangles (for preview only)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Face Capture (Press C to save face, Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press "c" to capture face
        if key == ord('c'):
            if len(faces) == 0:
                print("No face detected.")
                continue

            # Only crop the first detected face
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]  # crop only face (gray recommended)

            # Resize to fixed size for training
            face_img = cv2.resize(face_img, (200, 200))

            filename = f"{label_name}_{count:04d}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), face_img)
            print(f"Saved: {filename}")
            count += 1

            if num_images and count >= num_images:
                print("Reached target number of face images.")
                break

        # Press "q" to quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total saved face images: {count}")


if __name__ == "__main__":
    label = input("Enter person name: ").strip()
    capture_face_images(label)
