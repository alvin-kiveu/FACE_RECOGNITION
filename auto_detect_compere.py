import cv2
import numpy as np
import mysql.connector

# Function to detect faces in an image
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image, faces

# Function to calculate the Euclidean distance between two feature vectors
def euclidean_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# Function to load employee features from the database
def load_employee_database():
    employee_database = {}
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="face"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT id, features FROM employees")
    result = mycursor.fetchall()
    for row in result:
        employee_id = row[0]
        features_str = row[1]
        features = np.fromstring(features_str, dtype=float, sep=',')
        employee_database[employee_id] = features
    mycursor.close()
    mydb.close()
    return employee_database

# Main function to capture images from the camera and verify employees
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return
    employee_database = load_employee_database()
    max_attempts = 10  # Maximum number of attempts to detect a face
    attempt_count = 0
    while attempt_count < max_attempts:
        ret, frame = cap.read()
        detected_frame, faces = detect_faces(frame)
        cv2.imshow('Face Recognition - Human Resources', detected_frame)
        if len(faces) == 1:
            captured_image_features = np.random.rand(128)  # Placeholder for actual feature extraction
            for employee_id, employee_features in employee_database.items():
                similarity_score = euclidean_distance(captured_image_features, employee_features)
                threshold = 0.5  # Example threshold
                if similarity_score < threshold:
                    print("Employee {} verified.".format(employee_id))
                    # Implement further actions (e.g., grant access, log verification)
                    break
            else:
                print("Employee not recognized.")
          
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        attempt_count += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
