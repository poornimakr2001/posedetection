from fastapi import FastAPI, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import os
import pickle
import mysql.connector
from datetime import datetime, timedelta
import time
import secrets
import bcrypt
from typing import Optional
from PIL import Image
import io
import random

app = FastAPI()
security = HTTPBasic()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "attendancedb"
}

 # Load YOLO face detection model
model = YOLO("yolov8n-face.pt")

# Face recognition setup
KNOWN_FACES_DIR = r"C:\RCSS\SUTHERLAND"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
FACE_CACHE_FILE = "face_cache.pkl"

# Initialize face encodings cache
known_face_encodings = []
known_face_names = []

def setup_database():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Verify tables exist (schema should be created separately)
    cursor.execute("SHOW TABLES LIKE 'employee'")
    if not cursor.fetchone():
        raise Exception("Database tables not initialized. Please run the SQL schema first.")
    
    conn.close()

setup_database()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def generate_missing_employee_ids():
    """
    Generate employee IDs for existing employees who only have names (folders)
    but no emp_id in the database.
    """
    print("Generating missing employee IDs...")
    
    try:
        # Connect to database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Get all existing employee IDs to avoid duplicates
        cursor.execute("SELECT emp_id FROM employee")
        existing_ids = [row['emp_id'] for row in cursor.fetchall()]
        print(f"Found {len(existing_ids)} existing employee IDs")
        
        # Scan through directories in KNOWN_FACES_DIR
        for folder_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, folder_name)
            if not os.path.isdir(person_dir):
                continue
                
            # Check if this person exists in the database by name
            cursor.execute("SELECT emp_id FROM employee WHERE name = %s", (folder_name,))
            result = cursor.fetchone()
            
            if result:
                # Person exists but check if they have an emp_id
                emp_id = result['emp_id']
                if emp_id:
                    print(f"Employee '{folder_name}' already has ID: {emp_id}")
                    continue
            
            # Generate a new employee ID based on name
            # Format: First 2 letters of name + 4 random digits
            name_prefix = ''.join(c for c in folder_name[:2] if c.isalpha()).upper()
            if len(name_prefix) < 2:  # Ensure we have at least 2 characters
                name_prefix = (name_prefix + "XX")[:2]
                
            # Keep generating until we find an unused ID
            while True:
                random_digits = ''.join(random.choices('0123456789', k=4))
                new_emp_id = f"{name_prefix}{random_digits}"
                
                if new_emp_id not in existing_ids:
                    existing_ids.append(new_emp_id)  # Add to our tracking list
                    break
            
            print(f"Generated ID '{new_emp_id}' for employee '{folder_name}'")
            
            # If the employee exists, update their record
            if result:
                cursor.execute(
                    "UPDATE employee SET emp_id = %s WHERE name = %s",
                    (new_emp_id, folder_name)
                )
                print(f"Updated employee '{folder_name}' with new ID: {new_emp_id}")
            else:
                # If the employee doesn't exist yet, we need to create a database entry
                # First, try to get a face encoding from their directory
                face_encoding = None
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            image_path = os.path.join(person_dir, filename)
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            if encodings:
                                face_encoding = encodings[0]
                                break
                        except Exception as e:
                            print(f"Error processing image {filename} for {folder_name}: {e}")
                
                if face_encoding:
                    # Generate a temporary password (they can reset it later)
                    temp_password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
                    
                    # Insert new employee with the generated ID
                    cursor.execute("""
                    INSERT INTO employee (emp_id, name, password_hash, face_embedding)
                    VALUES (%s, %s, %s, %s)
                    """, (
                        new_emp_id,
                        folder_name,
                        hash_password(temp_password),
                        str(face_encoding.tolist())
                    ))
                    print(f"Created new employee entry for '{folder_name}' with ID: {new_emp_id} and temporary password")
                else:
                    print(f"Warning: Could not find valid face encoding for '{folder_name}', skipping database entry")
        
        # Commit all changes
        conn.commit()
        print("Employee ID generation completed successfully")
        
        # Update the face recognition cache
        load_known_faces()
        
    except Exception as e:
        print(f"Error generating employee IDs: {e}")
        if 'conn' in locals() and conn.is_connected():
            conn.rollback()
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def load_known_faces():
    global known_face_encodings, known_face_names
    
    known_face_encodings = []
    known_face_names = []
    
    print(f"Checking KNOWN_FACES_DIR: {KNOWN_FACES_DIR}")
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Creating directory: {KNOWN_FACES_DIR}")
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return
    
    # Build a map of names to employee IDs from the database
    name_to_emp_id = {}
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT emp_id, name FROM employee")
        for employee in cursor.fetchall():
            name_to_emp_id[employee['name']] = employee['emp_id']
        conn.close()
        print(f"Loaded {len(name_to_emp_id)} employee name mappings from database")
    except Exception as e:
        print(f"Error loading employee data: {e}")
    
    if os.path.exists(FACE_CACHE_FILE):
        try:
            with open(FACE_CACHE_FILE, "rb") as f:
                known_face_encodings, known_face_names = pickle.load(f)
            print(f"Loaded {len(known_face_names)} faces from cache")
            return
        except Exception as e:
            print(f"Error loading face cache: {e}")
    
    # Build cache from known_faces directory
    print("Building face cache from directory...")
    total_processed = 0
    
    for folder_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, folder_name)
        if os.path.isdir(person_dir):
            # Determine the employee ID based on the folder name
            emp_id = name_to_emp_id.get(folder_name, folder_name)
            
            print(f"Processing directory: {folder_name} (emp_id: {emp_id})")
            face_count = 0
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        print(f"  Processing image: {filename}")
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(emp_id)  # Store emp_id for recognition
                            face_count += 1
                        else:
                            print(f"  Warning: No face detected in {filename}")
                    except Exception as e:
                        print(f"  Error processing {image_path}: {e}")
            
            print(f"  Added {face_count} faces for {folder_name}")
            total_processed += face_count
    
    print(f"Total faces processed: {total_processed}")
    
    # Save cache if we found any faces
    if known_face_names:
        try:
            with open(FACE_CACHE_FILE, "wb") as f:
                pickle.dump((known_face_encodings, known_face_names), f)
            print(f"Cached {len(known_face_names)} faces")
        except Exception as e:
            print(f"Error saving face cache: {e}")
    else:
        print("No faces found to cache")

def mark_attendance(employee_id: int, is_login: bool = True):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    today = datetime.now().date()
    
    try:
        if is_login:
            # Check if already logged in today
            cursor.execute("""
            SELECT id FROM attendance 
            WHERE employee_id = %s AND date = %s AND logout_time IS NULL
            """, (employee_id, today))
            
            if cursor.fetchone():
                return False
            
            # Create new attendance record
            cursor.execute("""
            INSERT INTO attendance (employee_id, login_time, date)
            VALUES (%s, NOW(), %s)
            """, (employee_id, today))
        else:
            # Find latest login without logout
            cursor.execute("""
            SELECT id, login_time FROM attendance
            WHERE employee_id = %s AND logout_time IS NULL
            ORDER BY login_time DESC LIMIT 1
            """, (employee_id,))
            
            record = cursor.fetchone()
            if record:
                attendance_id, login_time = record
                hours_worked = (datetime.now() - login_time).total_seconds() / 3600
                
                cursor.execute("""
                UPDATE attendance 
                SET logout_time = NOW(), total_hours = %s
                WHERE id = %s
                """, (hours_worked, attendance_id))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT password_hash FROM admin_users WHERE username = %s", 
                  (credentials.username,))
    admin = cursor.fetchone()
    conn.close()
    
    if not admin or not bcrypt.checkpw(credentials.password.encode('utf-8'), 
                                      admin['password_hash'].encode('utf-8')):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Run on startup
@app.on_event("startup")
async def startup_event():
    """Run when the application starts"""
    load_known_faces()
    # Uncomment to auto-generate IDs on startup
    # generate_missing_employee_ids()

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=JSONResponse)
async def register_employee(
    emp_id: str = Form(...),
    name: str = Form(...),
    password: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Validate inputs
        if not emp_id or not name or not password:
            return JSONResponse(
                {"status": "error", "message": "All fields are required"},
                status_code=400
            )
        
        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Detect and encode face
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            return JSONResponse(
                {"status": "error", "message": "No face detected in image"},
                status_code=400
            )
        
        encodings = face_recognition.face_encodings(image_np, face_locations)
        if not encodings:
            return JSONResponse(
                {"status": "error", "message": "Could not encode face"},
                status_code=400
            )
        
        # Save to database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        try:
            # Check if employee exists
            cursor.execute("SELECT id FROM employee WHERE emp_id = %s", (emp_id,))
            if cursor.fetchone():
                return JSONResponse(
                    {"status": "error", "message": "Employee ID already exists"},
                    status_code=400
                )
            
            # Insert new employee
            cursor.execute("""
            INSERT INTO employee (emp_id, name, password_hash, face_embedding)
            VALUES (%s, %s, %s, %s)
            """, (
                emp_id, 
                name, 
                hash_password(password),
                str(encodings[0].tolist())
            ))
            
            # Save face image - use the name for the folder
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            image_path = os.path.join(person_dir, "profile.jpg")
            image.save(image_path)
            
            print(f"Saved face image for employee {emp_id} in folder {name} at {image_path}")
            
            # Update cache
            known_face_encodings.append(encodings[0])
            known_face_names.append(emp_id)  # Store the emp_id for identification
            
            try:
                with open(FACE_CACHE_FILE, "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
                print(f"Updated face cache with new employee {emp_id}")
            except Exception as e:
                print(f"Error updating face cache: {e}")
            
            conn.commit()
            
            # Reload known faces
            load_known_faces()
            
            return JSONResponse({
                "status": "success", 
                "message": f"Employee {name} registered successfully with ID {emp_id}"
            })
        except mysql.connector.Error as err:
            conn.rollback()
            return JSONResponse(
                {"status": "error", "message": f"Database error: {err}"},
                status_code=500
            )
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in registration: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/process_attendance", response_class=JSONResponse)
async def process_attendance(
    emp_id: str = Form(...),
    password: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Verify credentials
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
        SELECT id, name, password_hash FROM employee WHERE emp_id = %s
        """, (emp_id,))
        employee = cursor.fetchone()
        conn.close()
        
        if not employee or not verify_password(password, employee['password_hash']):
            return JSONResponse(
                {"status": "error", "message": "Invalid credentials"},
                status_code=401
            )
        
        # Process face recognition
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            return JSONResponse(
                {"status": "error", "message": "No face detected"},
                status_code=400
            )
        
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        if not face_encodings:
            return JSONResponse(
                {"status": "error", "message": "Could not encode face"},
                status_code=400
            )
        
        # Compare with registered faces
        matches = face_recognition.compare_faces(
            known_face_encodings, 
            face_encodings[0],
            tolerance=0.4  # Lower tolerance for fewer false positives
        )
        
        if not any(matches):
            return JSONResponse(
                {"status": "error", "message": "Face not recognized"},
                status_code=401
            )
        
        # Get the best match
        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encodings[0]
        )
        best_match_index = np.argmin(face_distances)
        
        if not matches[best_match_index]:
            return JSONResponse(
                {"status": "error", "message": "Face verification failed"},
                status_code=401
            )
        
        # Verify emp_id matches the recognized face
        recognized_emp_id = known_face_names[best_match_index]
        if recognized_emp_id != emp_id:
            return JSONResponse(
                {"status": "error", "message": "Face does not match employee ID"},
                status_code=401
            )
        
        # Mark attendance
        mark_attendance(employee['id'], is_login=True)
        
        return JSONResponse({
            "status": "success",
            "message": "Attendance marked successfully",
            "employee_id": emp_id,
            "name": employee['name']  # Return the name from the database
        })
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(verify_admin)):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    # Daily attendance
    cursor.execute("""
    SELECT e.emp_id, e.name, a.login_time, a.logout_time, a.total_hours 
    FROM attendance a
    JOIN employee e ON a.employee_id = e.id
    WHERE a.date = CURDATE()
    ORDER BY a.login_time DESC
    """)
    daily_attendance = cursor.fetchall()
    
    # Weekly posture reports
    cursor.execute("""
    SELECT e.emp_id, e.name, 
           COUNT(CASE WHEN pl.posture = 'Good Posture' THEN 1 END) as good_posture,
           COUNT(CASE WHEN pl.posture = 'Leaning Posture' THEN 1 END) as bad_posture,
           DATE(pl.timestamp) as date
    FROM posture_logs pl
    JOIN employee e ON pl.employee_id = e.id
    WHERE pl.timestamp >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    GROUP BY e.emp_id, e.name, DATE(pl.timestamp)
    ORDER BY date DESC
    """)
    posture_reports = cursor.fetchall()
    
    conn.close()
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "daily_attendance": daily_attendance,
        "posture_reports": posture_reports,
        "username": username
    })

@app.get("/video_feed")
async def video_feed():
    cap = cv2.VideoCapture(0)
    
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform posture detection (simplified)
            posture = "Good Posture" if random.random() > 0.3 else "Leaning Posture"
            
            # Add posture text to frame
            cv2.putText(frame, f"Posture: {posture}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/admin/generate_employee_ids")
async def admin_generate_ids(username: str = Depends(verify_admin)):
    """Admin endpoint to generate employee IDs for existing folders"""
    try:
        generate_missing_employee_ids()
        return JSONResponse({
            "status": "success",
            "message": "Employee ID generation completed"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/admin/debug_directories")
async def debug_directories(username: str = Depends(verify_admin)):
    """Debug route to display directory structure and mapping"""
    try:
        # Check directory structure
        structure = {"directories": []}
        
        # Get employee mapping from database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT emp_id, name FROM employee")
        employees = cursor.fetchall()
        conn.close()
        
        # Create lookup maps
        name_to_id = {emp['name']: emp['emp_id'] for emp in employees}
        id_to_name = {emp['emp_id']: emp['name'] for emp in employees}
        
        # Check directories
        for folder_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, folder_name)
            if os.path.isdir(person_dir):
                dir_info = {
                    "folder_name": folder_name,
                    "matched_emp_id": name_to_id.get(folder_name, "Unknown"),
                    "images": []
                }
                
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        dir_info["images"].append(filename)
                
                dir_info["image_count"] = len(dir_info["images"])
                structure["directories"].append(dir_info)
        
        # Check face cache if it exists
        cache_info = {"exists": False}
        if os.path.exists(FACE_CACHE_FILE):
            cache_info["exists"] = True
            try:
                with open(FACE_CACHE_FILE, "rb") as f:
                    encodings, names = pickle.load(f)
                cache_info["encoding_count"] = len(encodings)
                cache_info["name_count"] = len(names)
                
                # Sample of stored identifiers (first 5)
                cache_info["sample_identifiers"] = names[:5]
                
                # Check for mismatches
                mismatches = []
                for name in names:
                    if name not in id_to_name:
                        mismatches.append(name)
                cache_info["mismatches"] = mismatches[:5] if mismatches else []
            except Exception as e:
                cache_info["error"] = str(e)
        
        return JSONResponse({
            "employee_count": len(employees),
            "directory_structure": structure,
            "face_cache": cache_info
        })
    except Exception as e:
        print(f"Error debugging directories: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)