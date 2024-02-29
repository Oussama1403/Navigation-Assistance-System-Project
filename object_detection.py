import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/cfg/yolov3.cfg")
classes = []
with open("yolo/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Function to perform object detection with adjusted non-maximum suppression (NMS) threshold
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  # Adjust confidence threshold as needed
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
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=0.2, nms_threshold=0.6)  # Adjust NMS threshold
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

# Main loop for video capture
cap = cv2.VideoCapture(0)  # Change 0 to your camera index or video file path

try:
    while True:
        ret, frame = cap.read()
        # Resize input frames to a smaller resolution
        frame_resized = cv2.resize(frame, (640, 480))  # Adjust the resolution as needed

        # Perform object detection on the resized frame
        class_ids, confidences, boxes = detect_objects(frame_resized)

        # Create a copy of the frame to draw on
        frame_copy = frame_resized.copy()

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_copy, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the annotated frame using OpenCV
        cv2.imshow("Object Detection", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
