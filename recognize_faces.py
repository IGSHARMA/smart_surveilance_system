import cv2
import face_recognition
import pickle

# Load the known faces and encodings
with open('models/face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

face_locations = []
face_encodings = []
face_names = []

# Start the video capture from your webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if ret:
        cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame[:, :, ::-1]

    # Find all the face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    print(f"Detected face locations: {face_locations}")

    # Only process face encodings if faces are detected
    face_encodings = []
    if face_locations:
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"Generated face locations: {face_locations}")
        except Exception as e:
            print(f"Error in encoding faces: {e}")
            continue
    else:
        print("No faces were found.")
        continue  # Skip the rest of the loop and go to the next frame

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(f"Match found: {name} with encoding {known_face_encodings[best_match_index]}")  # Print matching name and encoding

        face_names.append(name)

        if name == "Unknown":
            print("Alert! Unknown face detected!")
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

video_capture.release()
cv2.destroyAllWindows()
