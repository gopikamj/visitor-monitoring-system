import cv2
from ultralytics import YOLO

# =====================================================
# LOAD WEAPON DETECTION MODEL
# =====================================================
model = YOLO("weapon_yolov8.pt")

# =====================================================
# VIDEO PROCESSING
# =====================================================
# def detect_weapon(source):
#     cap = cv2.VideoCapture(source)

#     if not cap.isOpened():
#         print("❌ Cannot open video source")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame, conf=0.4)

#         for r in results:
#             if r.boxes is None:
#                 continue

#             for box, cls, conf in zip(
#                 r.boxes.xyxy.cpu().numpy(),
#                 r.boxes.cls.cpu().numpy(),
#                 r.boxes.conf.cpu().numpy()
#             ):
#                 x1, y1, x2, y2 = map(int, box)
#                 label = model.names[int(cls)]

#                 # RED ALERT BOX
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                 cv2.putText(
#                     frame,
#                     f"⚠ {label.upper()} {conf:.2f}",
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (0, 0, 255),
#                     2
#                 )

#         cv2.imshow("WEAPON DETECTION ALERT", frame)

#         if cv2.waitKey(1) & 0xFF == 27:  # ESC
#             break

#     cap.release()
#     cv2.destroyAllWindows()
def detect_weapon(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return "Cannot open video source"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)

        for r in results:
            if r.boxes is None:
                continue

            for box, cls, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]

                # 🔴 WEAPON FOUND
                message = f"Weapon detected: {label.upper()} (Confidence: {conf:.2f})"
                print(message)

                # Optional display
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"⚠ {label.upper()} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                cv2.imshow("WEAPON DETECTION ALERT", frame)
                cv2.waitKey(1)

                # 🔚 EXIT IMMEDIATELY
                cap.release()
                cv2.destroyAllWindows()
                return message

        cv2.imshow("WEAPON DETECTION ALERT", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 🟢 NO WEAPON FOUND
    return "No weapon detected"

# =====================================================
# MAIN
# =====================================================
# if __name__ == "__main__":
def weapon_main(choice,path=None):
    # print("1 - Live Webcam")
    # print("2 - Video File")

    # choice = input("Select option: ").strip()

    if choice == "1":
        return detect_weapon(0)
    elif choice == "2":
        # path = input("Enter video path: ").strip()
        return detect_weapon(path)
    

# import cv2
# from ultralytics import YOLO

# # =====================================================
# # LOAD WEAPON DETECTION MODEL
# # =====================================================
# model = YOLO("weapon_yolov8.pt")

# # 👉 Weapon classes only (model names അനുസരിച്ച്)
# WEAPON_CLASSES = ["gun", "knife", "pistol", "rifle"]

# # =====================================================
# # VIDEO PROCESSING
# # =====================================================
# def detect_weapon(source):
#     cap = cv2.VideoCapture(source)

#     if not cap.isOpened():
#         print("❌ Cannot open video source")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame, conf=0.4)

#         for r in results:
#             if r.boxes is None:
#                 continue

#             for box, cls, conf in zip(
#                 r.boxes.xyxy.cpu().numpy(),
#                 r.boxes.cls.cpu().numpy(),
#                 r.boxes.conf.cpu().numpy()
#             ):
#                 label = model.names[int(cls)].lower()

#                 # 🔴 WEAPON ONLY FILTER
#                 if label not in WEAPON_CLASSES:
#                     continue

#                 x1, y1, x2, y2 = map(int, box)

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                 cv2.putText(
#                     frame,
#                     f"⚠ WEAPON: {label.upper()} {conf:.2f}",
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (0, 0, 255),
#                     2
#                 )

#         cv2.imshow("WEAPON DETECTION ONLY", frame)

#         if cv2.waitKey(1) & 0xFF == 27:  # ESC
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # =====================================================
# # MAIN
# # =====================================================
# if __name__ == "__main__":
#     print("1 - Live Webcam")
#     print("2 - Video File")

#     choice = input("Select option: ").strip()

#     if choice == "1":
#         detect_weapon(0)
#     elif choice == "2":
#         path = input("Enter video path: ").strip()
#         detect_weapon(path)
#     else:
#         print("Invalid choice")

