from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import os
import pickle
import sqlite3
from PIL import Image
import io

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Load YOLO face detection model
model = YOLO("yolov8n-face.pt")

# Known faces directory
KNOWN_FACES_DIR = "C:\RCSS\SUTHERLAND"
CACHE_FILE = "face_cache.pkl"

# Load cached face encodings if available
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"Loaded {len(known_face_encodings)} faces from cache.")
else:
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

        if os.path.isdir(person_folder):
            print(f"Processing {person_name}...")

            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    normalized_encoding = encodings[0] / np.linalg.norm(encodings[0])
                    known_face_encodings.append(normalized_encoding)
                    known_face_names.append(person_name)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Encoded and cached {len(known_face_encodings)} faces.")

# Routes for pages
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/mode")
async def mode(request: Request):
    return templates.TemplateResponse("mode.html", {"request": request})

@app.get("/attendance")
async def attendance(request: Request):
    return templates.TemplateResponse("attendance.html", {"request": request})

# Process frame and mark attendance
@app.post("/process_frame/")  # Handling image upload for face recognition
async def process_frame(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces with YOLO
    results = model(image_cv)
    best_face = None
    highest_confidence = 0

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > highest_confidence:
                highest_confidence = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_face = (y1, x2, y2, x1)  # ymin, xmax, ymax, xmin

    # Recognize the face if detected
    name = "Unknown"
    if best_face:
        face_encodings = face_recognition.face_encodings(image_cv, [best_face])

        if face_encodings:
            face_encoding = face_encodings[0] / np.linalg.norm(face_encodings[0])
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
                mark_attendance(name)  # Store attendance

    return {"name": name, "status": "Attendance recorded" if name != "Unknown" else "Face not recognized"}
# Function to store attendance in the database
def mark_attendance(name):
    conn = sqlite3.connect("database/attendance.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

# Fetch attendance report
@app.get("/report")
async def get_attendance():
    conn = sqlite3.connect("database/attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    data = cursor.fetchall()
    conn.close()
    return JSONResponse({"attendance": data})

