import cv2
import numpy as np
from deepface import DeepFace

# Load the face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

def detect_bounding_box(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces, frame

while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    
    if not result or video_frame is None:
        print("Error: Could not read frame.")
        break
    
    # Flip the frame horizontally for a mirror view
    video_frame = cv2.flip(video_frame, 1)
    
    faces, processed_frame = detect_bounding_box(video_frame)  # Detect faces
    
    text = "Face Not Detected"
    color = (0, 0, 255)
    
    if len(faces) > 0:
        text = "Face Detected"
        color = (0, 255, 0)
        
        # Extract face for emotion detection
        for (x, y, w, h) in faces:
            face_roi = video_frame[y:y+h, x:x+w]
            try:
                # Analyze emotions
                analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                text = f"Emotion: {emotion.capitalize()}"
            except:
                text = "Emotion: Unknown"
    
    cv2.putText(processed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show the processed frame in real-time
    cv2.imshow("Face & Emotion Detection", processed_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
