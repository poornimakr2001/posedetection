import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import os
import pickle

# Load the YOLOv8 model for face detection
model = YOLO("yolov8n-face.pt")

# Directory containing known face images
KNOWN_FACES_DIR = r"C:\RCSS\input_images"
CACHE_FILE = "face_cache.pkl"

# Load known face encodings from cache if available
# Load known face encodings from cache if available
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
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])
    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Encoded and cached {len(known_face_encodings)} faces.")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using YOLO
    results = model(rgb_frame)
    face_locations = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer values
            face_locations.append((y1, x2, y2, x1))  # Convert format for face_recognition
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

    # Show output
    cv2.imshow("YOLO Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()