import os
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
from PIL import Image
import pickle

# Load YOLO model
model = YOLO("yolov8n-face.pt")

# Load cached known face encodings
with open("face_cache.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Path to test images
TEST_DIR = "C:\\Users\\krpoo\\OneDrive\\Desktop\\internship\\test_images"

# Function to process images
def test_accuracy():
    total_tests = 0
    correct_predictions = 0

    for image_name in os.listdir(TEST_DIR):
        image_path = os.path.join(TEST_DIR, image_name)

        # Read Image
        image = Image.open(image_path)
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

        name = "Unknown"
        if best_face:
            face_encodings = face_recognition.face_encodings(image_cv, [best_face])
            if face_encodings:
                face_encoding = face_encodings[0] / np.linalg.norm(face_encodings[0])
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < 0.5:
                    name = known_face_names[best_match_index]

        # Compare with expected output (modify as needed)
        expected_name = image_name.split(".")[0]  # Assuming filename is the person's name
        total_tests += 1

        if name.lower() == expected_name.lower():
            correct_predictions += 1
            print(f"✅ Correct: {image_name} -> {name}")
        else:
            print(f"❌ Incorrect: {image_name} -> {name} (Expected: {expected_name})")

    # Calculate Accuracy
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n🎯 Model Accuracy: {accuracy:.2f}%")

# Run accuracy test
test_accuracy()
