from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, Request

from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import os
import pickle
import io
from PIL import Image  # <-- Import Image for handling image files

app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates
templates = Jinja2Templates(directory="templates")

# Load YOLO model
model = YOLO("yolov8n-face.pt")

# Directory for known faces
KNOWN_FACES_DIR = r"C:\RCSS\input_images"
CACHE_FILE = "face_cache.pkl"

# Load cached face encodings
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"Loaded {len(known_face_encodings)} faces from cache.")
else:
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Encoded and cached {len(known_face_encodings)} faces.")

# Home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Video streaming generator function
def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using YOLO
        results = model(rgb_frame)
        face_locations = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Recognize faces
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[best_match_index]
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame to JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

# Video streaming endpoint
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# Frame processing endpoint
@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    # Process the uploaded file, detect faces, and return an image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # For demonstration, save the image as processed (you can add your processing logic here)
    output_path = "processed_frame.jpg"
    image.save(output_path)  # Save the image to disk (you can modify this for real processing)
    
    # Return the processed image
    return FileResponse(output_path, media_type="image/jpeg")

