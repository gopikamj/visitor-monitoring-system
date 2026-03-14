import os

image_folder = r"D:\bu Visitor Monitoring System\project\wep_dataset\train\images"
label_folder = r"D:\bu Visitor Monitoring System\project\wep_dataset\train\labels"

images = [f.split('.')[0] for f in os.listdir(image_folder)]
labels = [f.split('.')[0] for f in os.listdir(label_folder)]

missing_labels = []

for img in images:
    if img not in labels:
        missing_labels.append(img)

print("Images without labels:", missing_labels)