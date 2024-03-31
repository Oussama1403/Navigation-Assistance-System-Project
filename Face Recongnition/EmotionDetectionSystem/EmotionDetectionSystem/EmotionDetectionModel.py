import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import os

emotion_folders = [
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/angry",
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/disgust",
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/fear",
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/happy",
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/neutral",
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/sad",
    "/home/top-g/Documents/Navigation-Assistance-System-Project/Face Recongnition/EmotionDetectionSystem/surprise"
]

def load_data(emotion_folders):
    images = []
    labels = []
    for i, folder in enumerate(emotion_folders):
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48,48))
            images.append(img)
            labels.append(i)
    return np.array(images), np.array(labels)

images, labels = load_data(emotion_folders)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
model.save("emotion_detection_model.h5")
