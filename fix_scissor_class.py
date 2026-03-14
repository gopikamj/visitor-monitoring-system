import os

# Folder where your downloaded scissor labels are
folder = r"C:\Users\ASUS\Downloads\Scissors.v1i.yolov8\train\labels"  # change this path

for file in os.listdir(folder):
    if file.endswith(".txt"):
        path = os.path.join(folder, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            parts[0] = "2"   # change class id to scissors
            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("All scissor labels changed to class 2")