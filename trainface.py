import cv2
import os
import numpy as np
import pickle

DATA_DIR = "faces"             # base folder

def load_images_and_labels(data_dir):
    images = []
    labels = []
    label_map = {}             # name -> id
    current_id = 0

    for root, dirs, files in os.walk(data_dir):
        for dirname in dirs:
            person_name = dirname
            person_dir = os.path.join(root, dirname)

            # assign numeric id for this person
            if person_name not in label_map:
                label_map[person_name] = current_id
                current_id += 1

            person_id = label_map[person_name]

            # loop through images
            for filename in os.listdir(person_dir):
                if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
                    continue

                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("Could not read", img_path)
                    continue

                # resize to fixed size (same as capture)
                img = cv2.resize(img, (200, 200))

                images.append(img)
                labels.append(person_id)

    return images, labels, label_map


if __name__ == "__main__":
    print("Loading images...")
    images, labels, label_map = load_images_and_labels(DATA_DIR)

    if len(images) == 0:
        print("No images found in 'faces' folder.")
        exit()

    print("Number of images:", len(images))
    print("Persons and IDs:", label_map)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Create LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Training model...")
    recognizer.train(images, labels)
    print("Training finished.")

    # Save model
    recognizer.save("face_model.yml")

    # Save label map (so we can convert id -> name later)
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_map, f)

    print("Model saved as face_model.yml")
    print("Labels saved as labels.pickle")
