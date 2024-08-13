import os
import face_recognition
import pickle

known_face_encodings = []
known_face_names = [] 

def encode_face(directory):
    for name in os.listdir(directory):
        person_dir = os.path.join(directory, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                img_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(name)

encode_face('datasets')
# Save the encodings and names
with open('models/face_encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Face encodings saved successfully.")