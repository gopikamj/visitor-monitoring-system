import os

folder = r"D:\bu Visitor Monitoring System\project\wep_dataset\val\labels"

for file in os.listdir(folder):
    if file.endswith(".txt"):

        # skip gun labels like 1.txt 2.txt 3.txt
        if file.replace(".txt","").isdigit():
            continue

        path = os.path.join(folder, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            parts[0] = "1"   # knife class
            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("Knife labels updated successfully!")