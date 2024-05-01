
import cv2

from random import randrange
#load some pre-trained data frontals from opencv(hear cascade algorithme)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#to capture video from webcam.
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:
    #read the current frame
    successful_frame_read, frame = webcam.read()

    #must convert to gryscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangles around the face 
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Clever programmer Face Detector', frame)
    key = cv2.waitKey(1)

    #stop if Q key is pressed 
    if key==81 or key==113:
        break

#relase the videocapture object
webcam.relase()

print("code completed")


