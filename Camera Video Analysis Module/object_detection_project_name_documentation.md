# Object Detection with YOLO Documentation

object_detection.py performs object detection using the YOLO (You Only Look Once) deep learning algorithm. The script loads a pre-trained YOLO model and uses it to detect objects in real-time from a live video stream or recorded video file. Detected objects are annotated with bounding boxes and labels.

## Dependencies
- OpenCV (cv2): OpenCV is used for image processing and video capture.
- NumPy: NumPy is used for numerical computations and array manipulation.

## Usage
1. Ensure that the necessary dependencies are installed.
2. Run the script with Python (`python object_detection.py`).
3. The script will open a live video stream from the default camera (index 0). If using a different camera or video file, modify the `cap = cv2.VideoCapture(0)` line accordingly.
4. Detected objects will be annotated with bounding boxes and labels in real-time.

## Functions
1. **load_yolo_model(weights_file, config_file):**
   - Loads the pre-trained YOLO model from the specified weights and configuration files.

2. **detect_objects(net, frame, classes):**
   - Performs object detection on the input frame using the loaded YOLO model.
   - Returns the class IDs, confidences, and bounding boxes of detected objects.

3. **draw_boxes(frame, class_ids, confidences, boxes, classes):**
   - Draws bounding boxes and labels on the input frame based on detected objects.
   - Returns the annotated frame.

4. **display_frame(frame):**
   - Displays the annotated frame using OpenCV.

5. **main():**
   - Orchestrates the object detection process by calling the necessary functions.
   - Loads the YOLO model, captures video frames, detects objects, draws bounding boxes, and displays the annotated frames in real-time.

## Configuration
- **CONFIDENCE_THRESHOLD:** Adjusts the confidence threshold for object detection.
- **NMS_THRESHOLD:** Adjusts the non-maximum suppression (NMS) threshold for filtering overlapping bounding boxes.
- **INPUT_FRAME_RESOLUTION:** Adjusts the resolution of input frames for object detection. Higher resolutions may impact performance.

