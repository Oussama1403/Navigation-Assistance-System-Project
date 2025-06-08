### Text Recognition, Object Detection, Face & Currency Recognition Documentation

This document provides an overview of the Python script `yolov8_camera_local.py` for real-time object detection, text recognition, face recognition, currency classification, and distance estimation using OpenCV, PyTesseract, pyttsx3, YOLOv8 (You Only Look Once), face_recognition, and a fine-tuned MobileNet model.

### Overview

The script captures frames from a video source, performs:
- **Object detection** using YOLOv8 ONNX model.
- **Text recognition** using PyTesseract (OCR) with real-time audio feedback.
- **Face recognition** using the `face_recognition` library and a directory of known faces.
- **Currency recognition** using a fine-tuned MobileNet model for Tunisian currency.
- **Distance estimation** for detected objects using the pinhole camera model.
- **Dominant color detection** for objects.
- **Audio feedback** for recognized objects, text, and faces using pyttsx3.

The script uses multi-threading for concurrent processing of object detection, face recognition, currency detection, and text recognition, improving performance and responsiveness.

### Libraries Used

- **OpenCV (cv2)**: Image processing, video capture, drawing, and display.
- **PyTesseract**: Tesseract OCR engine for text recognition.
- **NumPy (np)**: Numerical operations and array manipulation.
- **pyttsx3**: Text-to-speech for audio feedback.
- **onnxruntime**: Running YOLOv8 ONNX model for object detection.
- **face_recognition**: Face detection and recognition.
- **TensorFlow / Keras**: For loading and running the currency classification model.
- **threading, queue**: Multi-threading and inter-thread communication.

### Main Features & Functions

1. **Object Detection (YOLOv8)**
   - Loads YOLOv8 ONNX model and COCO class names.
   - Detects objects, draws bounding boxes, labels, and estimates distance.
   - Detects dominant color of each object.
   - Provides audio feedback for detected objects.

2. **Text Recognition**
   - Uses PyTesseract for OCR on video frames.
   - Recognizes text in both English and French.
   - Displays recognized text and provides audio feedback.

3. **Face Recognition**
   - Loads known faces from a directory.
   - Detects and recognizes faces in video frames.
   - Draws bounding boxes and labels for recognized faces.

4. **Currency Recognition**
   - Loads a fine-tuned MobileNet model for Tunisian currency.
   - Pre-filters regions likely to contain currency using edge detection.
   - Classifies detected currency notes and draws bounding boxes.

5. **Distance Estimation**
   - Estimates real-world distance to detected objects using the pinhole camera model and known object widths.

6. **Dominant Color Detection**
   - Detects the dominant color in the region of detected objects.

7. **Audio Feedback**
   - Uses pyttsx3 to provide spoken feedback for recognized objects, text, and faces.

8. **Multi-threaded Architecture**
   - Uses separate threads for object detection, face recognition, currency detection, and text recognition.
   - Main thread handles video capture and display.
   - Thread-safe sharing of detection results.

### Usage

1. Ensure all required libraries are installed: OpenCV, PyTesseract, pyttsx3, onnxruntime, face_recognition, tensorflow, numpy.
2. Place YOLOv8 ONNX model and COCO class names file in the specified paths.
3. Place known face images in the `known_faces` directory (filenames as names).
4. Place the currency classification model in the specified path.
5. Run the script. The video stream will display with annotated objects, recognized text, faces, and currency.
6. Audio feedback will be provided for detected items.
7. Press `q` to exit the script.

### Configuration

- **Camera**: Uses the local PC camera (`cv2.VideoCapture(0)`).
- **Persistence**: Detection results persist for a configurable number of frames.
- **Intervals**: Recognition intervals for text, face, and currency can be adjusted.
- **Confidence thresholds**: Configurable for object and currency detection.
- **Tesseract path**: Update `pytesseract.pytesseract.tesseract_cmd` if needed.

### Conclusion

This script provides a robust, multi-modal solution for real-time visual analysis, including object detection, text and face recognition, currency classification, distance estimation, and audio feedback. The modular, multi-threaded design allows for efficient and extensible integration into assistive or navigation systems.