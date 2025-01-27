import cv2
import face_recognition
import os
import numpy as np
import dlib
import csv
from datetime import datetime

class AttendanceLogger:
    def __init__(self):
        current_time = datetime.now()
        # Generate filename based on current date and time
        self.csv_file = current_time.strftime("%Y-%m-%d_%H-%M-%S") + "_attendance.csv"
        self.logged_today = set()  # Track who has been logged today
        self.create_csv_if_not_exists()
    
    def create_csv_if_not_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Timestamp', 'Confidence'])
    
    def log_attendance(self, known_faces):
        current_time = datetime.now()
        today_str = current_time.strftime("%Y-%m-%d")
        
        # Filter faces with confidence >= 50%
        high_confidence_faces = [face for face in known_faces if face[1] >= 0.50]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Log each high-confidence face
            for name, confidence in high_confidence_faces:
                # Remove confidence percentage from name if it exists
                clean_name = name.split(' (')[0]
                attendance_key = f"{clean_name}_{today_str}"    
                
                # Only log once per day per person
                if attendance_key not in self.logged_today:
                    writer.writerow([
                        clean_name,
                        current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{confidence:.2%}"
                    ])
                    self.logged_today.add(attendance_key)
                    print(f"Logged attendance for {clean_name} with {confidence:.2%}")

def load_known_faces(faces_dir):
    """Load known faces from directory."""
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(faces_dir):
        print(f"Error: Directory '{faces_dir}' not found")
        return known_face_encodings, known_face_names
    
    face_detector = dlib.get_frontal_face_detector()
    
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            try:
                face_image = face_recognition.load_image_file(image_path)
                face_locations = face_detector(face_image, 1)
                
                if len(face_locations) > 0:
                    top = face_locations[0].top()
                    right = face_locations[0].right()
                    bottom = face_locations[0].bottom()
                    left = face_locations[0].left()
                    
                    face_encoding = face_recognition.face_encodings(
                        face_image, 
                        [(top, right, bottom, left)]
                    )
                    
                    if face_encoding:
                        known_face_encodings.append(face_encoding[0])
                        known_face_names.append(person_name)
                        print(f"Loaded face for: {person_name}")
                        break
                    
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                
    return known_face_encodings, known_face_names

def run_face_recognition():
    print("Loading known faces...")
    known_face_encodings, known_face_names = load_known_faces('lfw_funneled')
    print(f"Loaded {len(known_face_names)} known faces")
    
    print("Opening webcam...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return

    face_detector = dlib.get_frontal_face_detector()
    attendance_logger = AttendanceLogger()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        dlib_faces = face_detector(rgb_small_frame, 1)
        
        face_locations = []
        for face in dlib_faces:
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            left = face.left()
            face_locations.append((top, right, bottom, left))
        
        face_encodings = []
        for loc in face_locations:
            try:
                encodings = face_recognition.face_encodings(
                    rgb_small_frame,
                    [loc]
                )
                if encodings:
                    face_encodings.append(encodings[0])
            except Exception as e:
                print(f"Error encoding face: {e}")
        
        face_names = []
        recognized_faces = []
        
        for face_encoding in face_encodings:
            name = "Unknown"
            confidence = 0.0
            
            if known_face_encodings:
                try:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, 
                        face_encoding, 
                        tolerance=0.6
                    )
                    
                    if True in matches:
                        matched_indices = [i for i, match in enumerate(matches) if match]
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, 
                            face_encoding
                        )
                        
                        best_match_index = matched_indices[np.argmin(face_distances[matched_indices])]
                        confidence = 1 - min(face_distances)
                        
                        if confidence > 0.45:
                            name = known_face_names[best_match_index]
                            recognized_faces.append((name, confidence))
                except Exception as e:
                    print(f"Error matching face: {e}")
            
            name_with_confidence = f"{name} ({confidence:.2%})" if confidence > 0 else name
            face_names.append(name_with_confidence)
        
        # Log attendance for this frame
        attendance_logger.log_attendance(recognized_faces)
        
        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame, 
                name, 
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
        
        # Display status
        status_text = f"Faces: {len(face_locations)}"
        cv2.putText(
            frame, 
            status_text, 
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX, 
            0.8, 
            (0, 255, 0), 
            1
        )
        
        cv2.imshow('Face Recognition Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()
