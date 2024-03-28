import cv2
import numpy as np
import mysql.connector
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Function to detect faces in an image
def detect_faces(image):
    # Load the pre-trained Haar Cascade face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, faces

# Function to extract facial features from an image
def extract_features(image, face):
    # Extract the region of interest (face) from the image
    (x, y, w, h) = face
    face_roi = image[y:y+h, x:x+w]
    
    # Perform face recognition to extract facial features
    # For simplicity, let's assume we already have precomputed features for each employee
    # Here, we'll just return a random feature vector as an example
    feature_vector = np.random.rand(128)  # Example feature vector
    
    return feature_vector

# Function to store employee data in the database
def store_employee_data(employee_id, employee_name, features):
    # Connect to MySQL database
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="",
      database="face"
    )
    
    mycursor = mydb.cursor()

    # Insert employee data into the database
    sql = "INSERT INTO employees (name, features, userid) VALUES (%s, %s, %s)"
    val = (employee_name, ','.join(map(str, features)), employee_id)
    mycursor.execute(sql, val)

    mydb.commit()
    mycursor.close()
    mydb.close()

# Main function to capture images of employees and store their data in the database
def main():
    # Create Tkinter window
    root = tk.Tk()
    root.title("Employee Data Capture")

    # Create Tkinter label for instructions
    instructions_label = tk.Label(root, text="Enter employee name and press 'Capture Image'")
    instructions_label.pack()

    # Create Tkinter entry for employee name
    employee_name_entry = tk.Entry(root)
    employee_name_entry.pack()

    # Function to capture image and store employee data
    def capture_image():
        employee_name = employee_name_entry.get()
        
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        # Check if the camera is opened successfully
        if not cap.isOpened():
            print("Error: Unable to access camera")
            return
        
        # Loop to continuously capture images from the camera
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Detect faces in the captured frame
            _, faces = detect_faces(frame)
            
            # Display the frame with detected faces
            cv2.imshow('Capture Employee Image', frame)
            
            # Check if 'c' is pressed to capture the image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) == 1:
                # Generate a unique employee ID
                employee_id = np.random.randint(1000, 9999)  # Example: generate a 4-digit random number
                
                # Extract features from the captured image
                features = extract_features(frame, faces[0])
                
                # Store employee data in the database
                store_employee_data(employee_id, employee_name, features)
                print("Employee {} ({}) data stored in the database.".format(employee_id, employee_name))
                
                # Pause for a moment to allow user to see the message
                cv2.waitKey(2000)
                
                # Release the camera and close all OpenCV windows
                cap.release()
                cv2.destroyAllWindows()
                
                # Close Tkinter window
                root.destroy()
                
                # Break the loop to stop capturing images after one face is detected and processed
                break
            
            # Check for key press to exit the program
            if key == ord('q'):
                break
        
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    # Create Tkinter button to capture image
    capture_button = tk.Button(root, text="Capture Image", command=capture_image)
    capture_button.pack()

    # Run the Tkinter event loop
    root.mainloop()

# Entry point of the program
if __name__ == "__main__":
    main()
