import cv2
import face_recognition
import pandas as pd
import os

def load_data_from_folder(folder, metadata_file):
    images = []
    labels = []
    metadata = []

    metadata_df = pd.read_excel(metadata_file)
    metadata_dict = metadata_df.set_index("Unique Identifiers ").T.to_dict("list")
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = face_recognition.load_image_file(os.path.join(folder,filename))
            encoding = face_recognition.face_encodings(img)[0]
            images.append(encoding)
            labels.append(filename.split(".")[0])
            identifier = filename.split(".")[0]
            if identifier in metadata_dict:
                metadata.append(metadata_dict[identifier])
            else:
                metadata.append(["Unknown","Unknown","Unknown"])
    return images, labels, metadata

training_data_folder = "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/Facial Recognition/TrainingDataset"
metadata_file = "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/Facial Recognition/Training.xlsx"

known_face_encodings, known_face_names, metadata = load_data_from_folder(training_data_folder, metadata_file)

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = "Unknown"
        full_name = "Unknown"
        age = "Unknown"
        gender = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            full_name, age, gender = metadata[first_match_index]
            cv2.putText(frame, "Identity verified, access granted", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,2555,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Identity not verified, access denied", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,2555,0), 2, cv2.LINE_AA)


        top, right, bottom, left = face_recognition.face_locations(frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)

        cv2.putText(frame, f"Name: {full_name}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255),1)
        cv2.putText(frame, f"Age: {age}", (left, bottom + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255),1)
        cv2.putText(frame, f"Gender: {gender}", (left, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255),1)
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
