from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Lakshmi\Downloads\Visitor Monitoring System\Visitor Monitoring System\project\wepontool\runs\detect\train\weights\best.ptt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    frame = results[0].plot()

    cv2.imshow("Weapon Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()