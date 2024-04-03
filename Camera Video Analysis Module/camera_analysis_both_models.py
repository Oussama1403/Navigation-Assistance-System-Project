import cv2
import numpy as np

# Function to load YOLO model
def load_yolo_model(weights_file, config_file):
    return cv2.dnn.readNet(weights_file, config_file)

# Function to detect objects using YOLO model
def detect_objects(net, frame, classes, confidence_threshold=0.2, nms_threshold=0.4):
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
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        filtered_boxes = boxes[indices].tolist()
        class_ids = [class_ids[i] for i in indices]
        confidences = [confidences[i] for i in indices]
    else:
        filtered_boxes = []

    return class_ids, confidences, filtered_boxes

# Function to draw bounding boxes and labels on the frame
def draw_boxes(frame, class_ids, confidences, boxes, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Main function
def main():
    # Load YOLO models
    current_directory = os.getcwd()

    yolo_weights_file = "yolov3.weights"
    yolo_config_file = "yolov3.cfg"
    openimages_weights_file = "yolov3-openimages.weights"
    openimages_config_file = "yolov3-openimages.cfg"
    
    yolo_model = load_yolo_model(yolo_weights_file, yolo_config_file)
    openimages_model = load_yolo_model(openimages_weights_file, openimages_config_file)

    # Load class names
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Open camera for video capture
    cap = cv2.VideoCapture(0)  # Change 0 to your camera index or video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (416, 416))

        # Detect objects using YOLOv3
        yolo_class_ids, yolo_confidences, yolo_boxes = detect_objects(yolo_model, frame_resized, classes)

        # Detect objects using YOLOv3 trained on Open Images
        openimages_class_ids, openimages_confidences, openimages_boxes = detect_objects(openimages_model, frame_resized, classes)

        # Draw bounding boxes and labels for YOLOv3
        draw_boxes(frame, yolo_class_ids, yolo_confidences, yolo_boxes, classes)

        # Draw bounding boxes and labels for YOLOv3 trained on Open Images
        draw_boxes(frame, openimages_class_ids, openimages_confidences, openimages_boxes, classes)

        # Display the annotated frame
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
