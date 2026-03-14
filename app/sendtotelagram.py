import cv2
from ultralytics import YOLO
import requests
import time
import os

# --------- TELEGRAM CONFIGURATION ---------
TOKEN = ""
CHAT_ID = ""

def send_alert():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": "🚨 ALERT! Firearm detected! Possible threat identified!"
    }
    data=requests.post(url, data=data)
    print(data)
    
    # Send snapshot
    # url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    # with open(snapshot_path, "rb") as photo:
    #     requests.post(url_photo, files={"photo": photo}, data={"chat_id": CHAT_ID})
send_alert()
