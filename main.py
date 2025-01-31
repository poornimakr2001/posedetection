from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import os
import pickle
import io
from PIL import Image

app = FastAPI()

# Serve static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Load YOLO model
model = YOLO("yolov8n-face.pt")

# Directory containing subfolders of individuals
KNOWN_FACES_DIR = r"C:\\RCSS\\SUTHERLAND_PREPROCESSED"  # Each subfolder is an individual
CACHE_FILE = "face_cache.pkl"

# Load cached face encodings if available
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"Loaded {len(known_face_encodings)} faces from cache.")
else:
    known_face_encodings = []
    known_face_names = []

    # Loop through each individual's folder
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

        if os.path.isdir(person_folder):  # Ensure it's a directory
            print(f"Processing {person_name}...")

            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)

                # Load image and encode face
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)  # Store person's name

    # Cache encodings for faster future use
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"Encoded and cached {len(known_face_encodings)} faces.")

# Home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Frame processing endpoint
@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces using YOLO
    results = model(image_cv)
    best_face = None
    highest_confidence = 0

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > highest_confidence:
                highest_confidence = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_face = (y1, x2, y2, x1)

    # If a face is detected, perform recognition
    if best_face:
        face_encodings = face_recognition.face_encodings(image_cv, [best_face])
        name = "Unknown"

        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[best_match_index]

            # Draw bounding box and label
            cv2.rectangle(image_cv, (best_face[3], best_face[0]), (best_face[1], best_face[2]), (0, 255, 0), 2)
            cv2.putText(image_cv, name, (best_face[3], best_face[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save and return processed image
    output_path = "processed_frame.jpg"
    cv2.imwrite(output_path, image_cv)
    return FileResponse(output_path, media_type="image/jpeg")
