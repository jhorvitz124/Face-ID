import cv2
import dlib

def facial_recognition():
    # Load the pre-trained face detector
    detector = dlib.get_frontal_face_detector()
    
    # Load the pre-trained face recognition model
    face_recognizer = dlib.face_recognition_model_v1("shape_predictor_68_face_landmarks.dat")
    
    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = detector(gray)
        
        # Iterate over each detected face
        for face in faces:
            # Get the facial landmarks
            landmarks = face_recognizer(frame, face)
            
            # Draw a rectangle around the face
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw circles around the facial landmarks
            for landmark in landmarks.parts():
                cv2.circle(frame, (landmark.x, landmark.y), 2, (0, 0, 255), -1)
        
        # Display the frame
        cv2.imshow('Facial Recognition', frame)
        
        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

def main():
    facial_recognition()

if __name__ == "__main__":
    main()
