import cv2
import face_recognition
import pickle 

with open('models/face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

face_locations = []
face_encodings = []
face_names = []

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = []
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    else:
        print("No faces were found.")
    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distances(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        face_names.append(name)

        if name == "Unkownn":
            print("Alert! Uknown face detected!")

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 225), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (225, 225, 225), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()