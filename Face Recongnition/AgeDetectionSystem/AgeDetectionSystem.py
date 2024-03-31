import cv2
from keras.models import load_model
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
age_model = load_model("/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/AgeDetectionSystem/age_detection_model.h5")

AGE = ["6-20","25-30","42-48","60-98"]

def detect_emotion(face_image):
    face_image = cv2.resize(face_image, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1],1])
    predicted_class = np.argmax(age_model.predict(face_image))
    return AGE[predicted_class]

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        face_image = frame[y:y+h, x:x+w]
        emotion = detect_emotion(face_image)
        cv2.putText(frame, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

