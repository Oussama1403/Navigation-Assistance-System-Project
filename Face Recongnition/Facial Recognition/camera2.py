import cv2

# Load pre-trained cascade classifier for face detection
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with face rectangles
    cv2.imshow('Face Detection', frame)
    
    # Wait for key press; stop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break

# Release the VideoCapture object
webcam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Code completed")
