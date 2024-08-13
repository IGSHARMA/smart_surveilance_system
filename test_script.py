import cv2
import face_recognition

# video_capture = cv2.VideoCapture(0)
# ret, frame = video_capture.read()

# rgb_frame = frame[:, :, ::-1]
# face_locations = face_recognition.face_locations(rgb_frame)
# face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

# print(face_encodings)
import face_recognition

# Load a sample picture and learn how to recognize it.
image = face_recognition.load_image_file("datasets/pratinav/profilepic.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

print(face_encodings)
