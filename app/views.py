import base64
import time
import cv2
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import Visitor, Guest, BlockedVisitor,todaysvisiter
from django.contrib import messages
import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from django.core.mail import EmailMessage
from django.conf import settings
from .camera import start_live_monitoring
# Path to your service account key
cred = credentials.Certificate("visitorcounter-25074-firebase-adminsdk-fbsvc-41f4336c6d.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://visitorcounter-25074-default-rtdb.firebaseio.com/'
})
# Create your views here.
DATA_DIR = "faces"
def send_test_email(message,to,image):
    email = EmailMessage(
        subject='Visitor Alert',
        body=message,
        from_email=settings.EMAIL_HOST_USER,
        to=[to],
    )

    # Image path
    mypath="media/visited photo/"+str(image)
    image_path = os.path.join(settings.BASE_DIR,mypath)

    # Attach image
    email.attach_file(image_path)

    email.send()
    return ("Email Sent Successfully!")

def index(request):
    if request.POST:
        start_live_monitoring()
        return redirect('/')
    return render(request, 'app/index.html')

def visitor_register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        phone = request.POST.get('phone')
        address = request.POST.get('address')

        if password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return redirect('register')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('register')
        
        # Create User
        user = User.objects.create_user(username=username, email=email, password=password)
        
        # Create Visitor Profile
        Visitor.objects.create(user=user, phone=phone, address=address, view_password=password)
        
        messages.success(request, 'Account created successfully! Please login.')
        return redirect('login')

    return render(request, 'app/register.html')

def visitor_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            messages.info(request, f'You are now logged in as {username}.')
            return redirect('index')
        else:
            messages.error(request, 'Invalid username or password.')
            
    return render(request, 'app/login.html')

def visitor_logout(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')
import os
import cv2


def capture_face_images(label_name, num_images=None):

    # Create folder for saving faces
    save_dir = os.path.join("faces", label_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)

    # If camera is already open somewhere, release it
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)  # IMPORTANT: give OS time to free camera

    # Open camera again cleanly
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


@login_required
def add_visitor(request):
    is_blocked_param = request.GET.get('blocked', 'false') == 'true'

    if request.method == 'POST':
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        purpose = request.POST.get('purpose')
        photo_data = request.POST.get('photo_data')
        is_blocked = request.POST.get('is_blocked') == 'true'

        if is_blocked:
             visitor = BlockedVisitor(
                 added_by=request.user,
                 name=name,
                 phone=phone,
                 email=email,
                 reason=purpose, # Purpose becomes Reason for blocked
             )
        else:
            visitor = Guest(
                added_by=request.user,
                name=name,
                phone=phone,
                email=email,
                purpose=purpose
            )

        if photo_data:
            # Handle base64 image
            for i in range(15):
                if photo_data.startswith(f'data:image/{i}'):
                    break
                format, imgstr = photo_data.split(';base64,') 
                ext = format.split('/')[-1] 
                data = ContentFile(base64.b64decode(imgstr), name=f'{name}_photo.{ext}')
                visitor.photo = data

        visitor.save()
        capture_face_images(name, num_images=10)
        if is_blocked:
             messages.success(request, 'Blacklisted visitor added successfully!')
             return redirect('blocked_visitor_list')
        
        messages.success(request, 'Visitor added successfully!')
        capture_face_images(name, num_images=10)
        return redirect('visitor_list')
    
    return render(request, 'app/add_visitor.html', {'is_blocked': is_blocked_param})

@login_required
def visitor_list(request):
    # Allow all logged-in users to see all guests
    guests = Guest.objects.all().order_by('-created_at')
    
    # Map phone number to BlockedVisitor ID (Global check)
    blocked_map = dict(BlockedVisitor.objects.all().values_list('phone', 'id'))
    
    for guest in guests:
        if guest.phone in blocked_map:
            guest.is_blocked_view = True
            guest.blocked_record_id = blocked_map[guest.phone]
        else:
            guest.is_blocked_view = False

    return render(request, 'app/visitor_list.html', {'guests': guests, 'page_title': 'All Visitors'})

@login_required
def blocked_visitor_list(request):
    # Allow all logged-in users to see all blocked visitors
    guests = BlockedVisitor.objects.all().order_by('-blocked_at')
    # Add attributes for template compatibility
    for guest in guests:
        guest.is_blocked_view = True 
        guest.created_at = guest.blocked_at
        guest.purpose = guest.reason 
        
    return render(request, 'app/visitor_list.html', {'guests': guests, 'page_title': 'Blocked Visitors'})

@login_required
def block_visitor(request, visitor_id):
    # Allow blocking any guest
    guest = get_object_or_404(Guest, id=visitor_id)
    
    # Check if already blocked (Global check)
    if not BlockedVisitor.objects.filter(phone=guest.phone).exists():
        BlockedVisitor.objects.create(
            added_by=request.user, # The user performing the block action
            name=guest.name,
            phone=guest.phone,
            email=guest.email,
            reason=guest.purpose,
            photo=guest.photo
        )
        messages.success(request, f'Visitor {guest.name} has been added to the Blocklist.')
    else:
        messages.info(request, f'Visitor {guest.name} is already in the Blocklist.')
        
    return redirect('visitor_list')

@login_required
def unblock_visitor(request, visitor_id):
    # Allow unblocking any guest
    guest = get_object_or_404(BlockedVisitor, id=visitor_id)
    guest.delete()
    messages.success(request, f'Visitor {guest.name} has been removed from the Blocklist.')
    return redirect('blocked_visitor_list')
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
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
@login_required
# def livemonitering(request):

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("face_model.yml")

#     with open("labels.pickle", "rb") as f:
#         label_map = pickle.load(f)

#     id_to_name = {v: k for k, v in label_map.items()}

#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         return HttpResponse("Cannot access camera")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         faces = face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(100, 100)
#         )

#         for (x, y, w, h) in faces:

#             face_roi = gray[y:y+h, x:x+w]
#             face_roi = cv2.resize(face_roi, (200, 200))

#             label_id, confidence = recognizer.predict(face_roi)

#             print("Confidence:", confidence)

#             # Default values
#             name = "Unknown"
#             status = "unknown"
#             message = "Unknown Person Detected!"

#             # 🔥 Recognition Logic
#             if confidence < 80:
#                 name = id_to_name.get(label_id, "Unknown")

#                 is_blocked = BlockedVisitor.objects.filter(name=name).exists()

#                 if is_blocked:
#                     status = "blocked"
#                     message = f"{name} is BLOCKED!"
#                 else:
#                     status = "allowed"
#                     message = f"{name} is ALLOWED"

#             # 🔥 Rectangle Color Based On Status
#             if status == "blocked":
#                 color = (0, 0, 255)  # Red
#             elif status == "allowed":
#                 color = (0, 255, 0)  # Green
#             else:
#                 color = (0, 255, 255)  # Yellow

#             # Draw rectangle
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#             cv2.putText(
#                 frame,
#                 name,
#                 (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2
#             )

#             # 🔥 Save Image to Database
#             face_color = frame[y:y+h, x:x+w]
#             success, buffer = cv2.imencode('.jpg', face_color)

#             if success:
#                 filename = f"{name}.jpg"
#                 visitor = todaysvisiter(
#                     visitername=name,
#                     status=status,
#                     dateofvisit=timezone.now()
#                 )

#                 visitor.image.save(
#                     filename,
#                     ContentFile(buffer.tobytes()),
#                     save=True
#                 )

#             cap.release()
#             cv2.destroyAllWindows()
#             data=todaysvisiter.objects.all()
#             print(data)
#             return render(request, "app/livemonitoring.html", {
#                 "message": message,
#                 "status": status,"data":data
#             })

#         cv2.imshow("Live Monitoring - Press q to exit", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     data=todaysvisiter.objects.all()
#     print(data)
#     return render(request, "app/livemonitoring.html",{"data":data})




def livemonitering(request):

    # 🔥 Load Face Recognition Model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")

    with open("labels.pickle", "rb") as f:
        label_map = pickle.load(f)

    id_to_name = {v: k for k, v in label_map.items()}

    # 🔥 Load Emotion Model
    emotion_model = load_model('models/emotion_model.h5')
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap1 = cv2.VideoCapture(0)

    if not cap1.isOpened():
        return HttpResponse("Cannot access camera")

    while True:
        ret, frame = cap1.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            # ----------------------------
            # 🔥 FACE RECOGNITION
            # ----------------------------
            face_gray = gray[y:y+h, x:x+w]
            face_gray = cv2.resize(face_gray, (200, 200))

            label_id, confidence = recognizer.predict(face_gray)

            name = "Unknown"
            status = "unknown"

            if confidence < 80:
                name = id_to_name.get(label_id, "Unknown")
                is_blocked = BlockedVisitor.objects.filter(name=name).exists()

                if is_blocked:
                    status = "blocked"
                else:
                    status = "allowed"
            else:
                status = "unknown"
            

            # ----------------------------
            # 🔥 EMOTION DETECTION
            # ----------------------------
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            preds = emotion_model.predict(face_roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]

            print("Detected:", name, status, emotion)

            # ----------------------------
            # 🔥 DRAW RECTANGLE
            # ----------------------------
            if status == "blocked":
                color = (0, 0, 255)
            elif status == "allowed":
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame,
                        f"{name} | {emotion}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

            # ----------------------------
            # 🔥 SAVE TO DATABASE
            # ----------------------------
            face_color = frame[y:y+h, x:x+w]
            success, buffer = cv2.imencode('.jpg', face_color)
            print("Face saved to DB:", name, status, emotion)
            if success:
                filename = f"{name}.jpg"

                visitor = todaysvisiter(
                    visitername=name,
                    status=status,
                    emotion=emotion,
                    dateofvisit=timezone.now()
                )

                visitor.image.save(
                    filename,
                    ContentFile(buffer.tobytes()),
                    save=True
                )

            cap1.release()
            # cv2.destroyAllWindows()
            user=request.user
            print("***************************************************8")
            print(user)
            dd=User.objects.get(username=user)
            print(dd.email)
            print(filename)
            data = todaysvisiter.objects.all()
            ##############################################################################
            message="A Visiter is their he is "+str(status)+" by you"
            send_test_email(message,dd.email,filename)
            if (status=="blocked") and ((emotion!="Happy")or (emotion!="Neutral") ):
                ref = db.reference("Device")

                # Update only Device field
                ref.update({
                "Device": "L0B1S1",
                "Mode": "Man",
                "Motion": "No Motion"
            })
            return render(request, "app/livemonitoring.html", {
                "name": name,
                "status": status,
                "emotion": emotion,
                "data": data
            })
        
        cv2.imshow("Live Monitoring - Press q to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()

    data = todaysvisiter.objects.all()
    return render(request, "app/livemonitoring.html", {"data": data})




def allowuser(request):
    

    # Reference to Device node
    ref = db.reference("Device")

    # Update only Device field
    ref.update({
    "Device": "L1B0S0",
    "Mode": "Man",
    "Motion": "No Motion"
})

    print("Device updated successfully!")
    return redirect('/livemonitering')
def Resetuser(request):
    

    # Reference to Device node
    ref = db.reference("Device")

    # Update only Device field
    ref.update({
    "Device": "L0B0S0",
    "Mode": "Man",
    "Motion": "No Motion"
})

    print("Device updated successfully!")
    return redirect('/visitors/')