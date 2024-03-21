import cv2
import numpy as np
import os
import multiprocessing



# Constants and Configurations
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.6
INPUT_FRAME_RESOLUTION = (640, 480)

# Load YOLO objecGIT t detection algorithm
def load_yolo_model(weights_file, config_file):
    return cv2.dnn.readNet(weights_file, config_file)

# Function to perform object detection with adjusted non-maximum suppression (NMS) threshold
def detect_objects(net, frame, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression with adjusted thresholds
    if boxes:
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)
        boxes = np.array(boxes)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = boxes[indices].tolist()
            class_ids = class_ids[indices].tolist()
            confidences = confidences[indices].tolist()
        else:
            filtered_boxes = []
            class_ids = []
            confidences = []
    else:
        filtered_boxes = []

    return class_ids, confidences, filtered_boxes

# Function to draw bounding boxes and labels on the frame
def draw_boxes(frame, class_ids, confidences, boxes, classes):
    frame_copy = frame.copy()
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_copy, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame_copy

# Function to display the annotated frame using OpenCV
def display_frame(frame):
    cv2.imshow("Object Detection", frame)

def process_frames(frame_queue, result_queue, yolo_model, classes):
    while True:
        frame = frame_queue.get()
        class_ids, confidences, boxes = detect_objects(yolo_model, frame, classes)
        annotated_frame = draw_boxes(frame, class_ids, confidences, boxes, classes)
        result_queue.put(annotated_frame)

# Main function to run object detection
def main():
    current_directory = os.getcwd()

    # Load YOLO model
    yolo_neural_network = load_yolo_model(f"{current_directory}/Camera Video Analysis Module/yolo/yolov3.weights", f"{current_directory}/Camera Video Analysis Module/yolo/cfg/yolov3.cfg")

    # Load class names
    classes = []
    with open(f"{current_directory}/Camera Video Analysis Module/yolo/data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    frame_queue = multiprocessing.Queue(maxsize=2)  # Adjust maxsize for buffering
    result_queue = multiprocessing.Queue(maxsize=1)

    process = multiprocessing.Process(target=process_frames, args=(frame_queue, result_queue, yolo_neural_network, classes))
    process.start()


    try:
        # Open camera for video capture
        cap = cv2.VideoCapture(0)  # Change 0 to your camera index or video file path
        while True:
            ret, frame = cap.read()
            frame_resized = cv2.resize(frame, INPUT_FRAME_RESOLUTION)
            frame_queue.put(frame_resized)
            try:
                annotated_frame = result_queue.get(timeout=1)  # Handle potential queue timeout
                display_frame(annotated_frame)
            except:
                print("empty queue")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        # Release the camera and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        frame_queue.put(None)  # Signal process to end
        process.join()  # Wait for processing to finish

if __name__ == "__main__":
    main()
