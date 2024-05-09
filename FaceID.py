import cv2
import face_recognition

# Initialize variables
known_face_encodings = []
known_face_names = []

def load_database():
    # Load the database from a file
    global known_face_encodings, known_face_names
    try:
        with open("database.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                name, encoding_str = line.strip().split(":")
                encoding = [float(x) for x in encoding_str.split(",")]
                known_face_names.append(name)
                known_face_encodings.append(encoding)
    except FileNotFoundError:
        print("Database file not found.")

def save_database():
    # Save the database to a file
    with open("database.txt", "w") as file:
        for name, encoding in zip(known_face_names, known_face_encodings):
            encoding_str = ",".join(str(x) for x in encoding)
            file.write(f"{name}:{encoding_str}\n")

def facial_recognition():
    # Load the database
    load_database()
    
    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Iterate over each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the current face encoding with the known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Find the best match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw the name of the person
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
        
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
