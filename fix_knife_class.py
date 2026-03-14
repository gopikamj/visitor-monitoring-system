import os

folder = r"C:\Users\ASUS\Downloads\knife.v1i.yolov8\valid\labels"

for file in os.listdir(folder):
    if file.endswith(".txt"):
        path = os.path.join(folder, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            parts[0] = "1"   # change class id to knife
            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("All knife labels changed to class 1")