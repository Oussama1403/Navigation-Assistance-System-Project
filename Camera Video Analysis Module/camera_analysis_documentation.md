### Text Recognition and Object Detection Documentation

This document provides an overview of the Python script `camera_analysis.py`  for real-time text recognition and object detection using OpenCV, PyTesseract, pyttsx3 and YOLO (You Only Look Once) neural network.

### Overview

The script captures frames from a video source, performs object detection to identify objects in the frames, and utilizes PyTesseract for real-time text recognition and it speaks the text using pyttsx3 librarie, Detected objects are annotated with bounding boxes and labels, and recognized text is displayed on the video stream. The script employs multiprocessing to handle frame processing and rendering simultaneously for improved performance.

### Libraries Used

- **OpenCV (cv2)**: Used for image processing, object detection, and displaying frames.
- **PyTesseract**: Provides an interface to Tesseract OCR (Optical Character Recognition) engine for text recognition.
- **NumPy (np)**: Utilized for numerical operations and array manipulation.
- **pyttsx3**: Convert text to speech.

### Functions

1. **load_yolo_model(weights_file, config_file)**:
   - Loads the YOLO object detection model from the specified weights and configuration files.

2. **detect_objects(net, frame, classes)**:
   - Performs object detection on the input frame using the provided YOLO neural network.
   - Identifies objects in the frame based on specified confidence and non-maximum suppression thresholds.
   - Returns the class IDs, confidences, and bounding boxes of the detected objects.

3. **draw_boxes(frame, class_ids, confidences, boxes, classes)**:
   - Draws bounding boxes and labels around the detected objects on the frame.
   - Labels the objects with their corresponding class names and confidence scores.

4. **display_frame(frame)**:
   - Displays the annotated frame with object detection results and recognized text.

5. **live_text_recognition(frame)**:
   - Performs real-time text recognition using PyTesseract on the input frame.
   - Converts the frame to grayscale and applies Gaussian blur to reduce noise.
   - Utilizes PyTesseract to extract text from the preprocessed frame.
   - Displays recognized text on the frame based on confidence scores.

6. **text_to_audio(text)**:
   - Function that uses pyttsx3 to convert text to speech

7. **process_frames(frame_queue, result_queue, yolo_model, classes)**:
   - Worker function executed by a separate process to handle frame processing.
   - Retrieves frames from the input queue, performs object detection and text recognition, and puts annotated frames into the result queue.

8. **main()**:
   - Main function responsible for initializing the script, loading YOLO model and class names, and setting up multiprocessing.
   - Captures frames from the video source and enqueues them for processing.
   - Retrieves annotated frames from the result queue and displays them in real-time.
   - Handles keyboard interrupts and releases resources upon script termination.

### Usage

1. Ensure that the required libraries (OpenCV, PyTesseract, pyttsx3) are installed.
2. Configure the script with the appropriate paths to YOLO weights, configuration files, and Tesseract data directory.
3. Run the script, and it will display the video stream with annotated objects and recognized text in real-time.
4. Press `ctrl+c` to exit the script.

### Conclusion

This script provides a flexible and efficient solution for real-time text recognition and object detection in video streams, with the ability to configure confidence thresholds and support multiple languages for text recognition. It can be customized and integrated into various applications requiring automated visual analysis and interpretation.