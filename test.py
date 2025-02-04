import os
import cv2
import numpy as np
import face_recognition
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import pickle
# Load YOLO model
model = YOLO("yolov8n-face.pt")

# Directory containing subfolders of individuals (test dataset)
TEST_FACES_DIR = r"C:\\RCSS\\SUTHERLAND_TEST"  # Replace with your test dataset path

# Load cached face encodings and names
CACHE_FILE = "face_cache.pkl"
with open(CACHE_FILE, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Lists to store ground truth and predicted labels
true_labels = []
predicted_labels = []

# Loop through each individual's folder in the test dataset
for person_name in os.listdir(TEST_FACES_DIR):
    person_folder = os.path.join(TEST_FACES_DIR, person_name)

    if os.path.isdir(person_folder):  # Ensure it's a directory
        print(f"Processing {person_name}...")

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            # Load image
            image = face_recognition.load_image_file(image_path)
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Only recognize if confidence is above 95%
                    if matches[best_match_index] and face_distances[best_match_index] < 0.4:  # Adjust threshold as needed
                        name = known_face_names[best_match_index]

                # Append ground truth and predicted labels
                true_labels.append(person_name)
                predicted_labels.append(name)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=known_face_names)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=known_face_names, yticklabels=known_face_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()