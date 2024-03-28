import cv2
import time
import numpy as np
import hashlib

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
    
    # Convert the face ROI to grayscale
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Compute hash of the grayscale face region
    hash_value = hashlib.md5(gray_face).hexdigest()
    
    return hash_value

# Main function to capture images from the camera, detect faces, and verify user
def main():
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
        
        # Display the captured frame
        ##cv2.imshow('Camera', frame)
        
        # Detect faces in the captured frame
        detected_frame, faces = detect_faces(frame)
        
        # Display the frame with detected faces
        cv2.imshow('Face Recognition - Human Resources', detected_frame)
        
        # Check for key press to capture image and verify user
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) == 1:  
                # Save the captured image with a unique filename
                timestamp = int(time.time())
                image_path = "captured_image_{}.jpg".format(timestamp)
                cv2.imwrite(image_path, detected_frame)
                print("Captured image saved as:", image_path)
                
                # Verify that only one face is detected
                features = extract_features(frame, faces[0])
                print("Extracted features:", features)
                
                # Implement user verification logic here
                # For example, you can compare the hash value with a database
                
                # After verification, break the loop to stop capturing images
                break
            else:
                print("Error: More than one face detected. Please ensure only one face is visible.")
        
        # Check for key press to exit the program
        elif key == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Entry point of the program
if __name__ == "__main__":
    main()
