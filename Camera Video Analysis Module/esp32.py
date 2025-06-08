import cv2
import numpy as np
import os
import pyttsx3
import pytesseract
import onnxruntime as ort
import threading
import queue
import time
import face_recognition
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from copy import deepcopy


# Constants and Configurations
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.6
INPUT_FRAME_RESOLUTION = (640, 480)
TEXT_RECOGNITION_INTERVAL = 10
FACE_RECOGNITION_INTERVAL = 3
CURRENCY_RECOGNITION_INTERVAL = 5
FACE_PERSISTENCE_FRAMES = 30
TEXT_PERSISTENCE_FRAMES = 30
OBJECT_PERSISTENCE_FRAMES = 30
CURRENCY_PERSISTENCE_FRAMES = 30
AUDIO_COOLDOWN_FRAMES = 30
CURRENCY_CONFIDENCE_THRESHOLD = 0.5
CURRENCY_NMS_THRESHOLD = 0.4
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.TESSDATA_PREFIX = r'C:\Program Files\Tesseract-OCR'

# Camera parameters for distance estimation (refined based on 1.26m actual distance)
FOCAL_LENGTH_MM = 3.05  # Refined focal length based on calibration to 1.26m
SENSOR_WIDTH_MM = 3.6   # Approximate sensor width of OV2640 (1/4" sensor) in mm
IMAGE_WIDTH_PX = INPUT_FRAME_RESOLUTION[0]  # Image width in pixels (640)

# Update known object widths
KNOWN_OBJECT_WIDTHS = {
    'person': 0.55,     # Adjusted shoulder width to 0.55m for better alignment
    'car': 1.8,        # Average car width
    'bicycle': 0.6,    # Average bicycle width
    'motorcycle': 0.8,
    'bus': 2.5,
    'truck': 2.5,
    'dog': 0.4,
    'cat': 0.3,
    'chair': 0.5,
    'table': 1.0,
    'bottle': 0.1,     # Average bottle width
    'cup': 0.08,       # Average cup width
    'laptop': 0.35,    # Average laptop width
    'tvmonitor': 0.6,  # Average TV/monitor width
    'sofa': 1.5,       # Average sofa width
    # Add more COCO classes as needed
}

# Global variables to store detection results with thread-safe access
last_face_results = []
last_face_update_frame = 0
last_text_results = []
last_text_update_frame = 0
last_object_results = []
last_object_update_frame = 0
last_currency_results = []
last_currency_update_frame = 0
last_audio_announcements = {}
results_lock = threading.Lock()

# Queues for inter-thread communication
frame_queue_object = queue.Queue(maxsize=1)
frame_queue_face = queue.Queue(maxsize=1)
frame_queue_currency = queue.Queue(maxsize=1)
frame_queue_text = queue.Queue(maxsize=1)
results_queue = queue.Queue()

# Function to estimate distance to an object
def estimate_distance(pixel_width, object_label, confidence=None):
    """
    Estimate distance to an object using the pinhole camera model.
    Formula: Distance (Z) = (Focal Length * Real-World Width) / Pixel Width
    Returns distance in meters or None if unknown object width or invalid pixel width.
    """
    if pixel_width <= 0:
        print(f"Invalid pixel width ({pixel_width}) for {object_label}")
        return None
    
    # Get the real-world width of the object
    real_width = KNOWN_OBJECT_WIDTHS.get(object_label.lower())
    if real_width is None:
        print(f"No known width for object: {object_label}")
        return None
    
    # Convert focal length to pixels (calibrated value)
    focal_length_px = (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX) / SENSOR_WIDTH_MM
    print(f"Debug - pixel_width: {pixel_width}, real_width: {real_width}, focal_length_px: {focal_length_px}")
    
    # Calculate distance in meters
    distance = (focal_length_px * real_width) / pixel_width
    
    # Adjust distance based on confidence (optional)
    if confidence is not None:
        distance *= (1 + (1 - confidence))  # Penalize lower confidence
    
    # Warn if bounding box is too small
    if pixel_width < 10:
        print(f"Warning: Bounding box for {object_label} is too small, distance may be inaccurate")
    
    print(f"Debug - Calculated distance for {object_label}: {distance:.2f} m")
    return distance

# Function to detect the dominant color in a region of interest (ROI)
def detect_dominant_color(roi, k=3):
    small_roi = cv2.resize(roi, (32, 32))
    hsv_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
    _, v_threshold = cv2.threshold(hsv_roi[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (hsv_roi[:,:,2] > v_threshold * 0.5) & (hsv_roi[:,:,2] < v_threshold * 1.5)
    pixels = hsv_roi.reshape(-1, 3)[mask.reshape(-1)]
    if len(pixels) < 10:
        return "Unknown"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    unique, counts = np.unique(labels, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]
    dominant_color_hsv = centers[dominant_cluster]
    dominant_color_bgr = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    h, s, v = dominant_color_hsv
    if s < 40 or v < 40:
        return "Gray" if v > 100 else "Black"
    elif v > 200 and s < 80:
        return "White"
    if 0 <= h < 15 or 165 <= h <= 180:
        return "Red" if s > 100 and v > 100 else "Pink"
    elif 15 <= h < 45:
        return "Orange" if s > 100 else "Brown"
    elif 45 <= h < 75:
        return "Yellow"
    elif 75 <= h < 150:
        return "Green" if s > 80 else "Olive"
    elif 150 <= h < 165:
        return "Cyan"
    elif 165 <= h < 255:
        return "Blue" if s > 100 and v > 100 else "Purple"
    return "Unknown"

# Load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)
                print(f"Loaded known face: {name}")
    return known_face_encodings, known_face_names

# Load YOLOv8 ONNX model (CPU only)
def load_yolo_model(onnx_file):
    try:
        session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        print("Using CPUExecutionProvider for YOLOv8")
        return session
    except Exception as e:
        print(f"Failed to load YOLOv8 model on CPU: {e}")
        return None

# Load currency classification model
def load_currency_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded currency model from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load currency model: {e}")
        return None

# Preprocess frame for YOLOv8
def preprocess_frame(frame, input_size=(640, 640)):
    img = cv2.resize(frame, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

# Perform object detection with YOLOv8
def detect_objects(session, frame, classes):
    if session is None:
        print("YOLOv8 session is None, cannot perform detection")
        return [], [], []
    
    input_size = (640, 640)
    img = preprocess_frame(frame, input_size)
    try:
        outputs = session.run(None, {session.get_inputs()[0].name: img})[0]
    except Exception as e:
        print(f"Error in YOLOv8 inference: {e}")
        return [], [], []
    
    class_ids = []
    confidences = []
    boxes = []
    height, width = frame.shape[:2]
    
    for detection in outputs[0].transpose(1, 0):
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > CONFIDENCE_THRESHOLD:
            cx, cy, w, h = detection[:4]
            x = int((cx - w / 2) * width / input_size[0])
            y = int((cy - h / 2) * height / input_size[1])
            w = int(w * width / input_size[0])
            h = int(h * height / input_size[1])
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
    
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_class_ids = [class_ids[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
        else:
            filtered_boxes = []
            filtered_class_ids = []
            filtered_confidences = []
    else:
        filtered_boxes = []
        filtered_class_ids = []
        filtered_confidences = []
    
    print(f"YOLOv8 detected {len(filtered_boxes)} objects: {[(classes[cid], conf) for cid, conf in zip(filtered_class_ids, filtered_confidences)]}")
    return filtered_class_ids, filtered_confidences, filtered_boxes

# Draw bounding boxes, labels, and distance annotations for objects
def draw_boxes(frame, class_ids, confidences, boxes, classes, audio_queue, current_frame):
    global last_object_results, last_object_update_frame, last_audio_announcements
    
    object_results = []
    current_labels = set()
    
    print(f"Processing {len(boxes)} detected objects in draw_boxes")
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        
        # Extract ROI for color detection
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            color = "Unknown"
            distance = None
            print(f"Invalid bounding box for {label}: w={w}, h={h}")
        else:
            roi = frame[y:y+h, x:x+w]
            color = detect_dominant_color(roi)
            # Estimate distance using bounding box width
            distance = estimate_distance(w, label)
            if distance is not None:
                print(f"Distance calculated for {label}: {distance:.2f} m")
        
        # Store results including distance
        object_results.append((label, confidence, (x, y, w, h), color, distance))
        current_labels.add(label)
        
        with results_lock:
            last_announced = last_audio_announcements.get(label, -AUDIO_COOLDOWN_FRAMES)
            if current_frame - last_announced >= AUDIO_COOLDOWN_FRAMES:
                audio_queue.put(f"{label}")
                last_audio_announcements[label] = current_frame
    
    with results_lock:
        last_object_results = object_results
        last_object_update_frame = current_frame
        print(f"Updated last_object_results with {len(last_object_results)} objects: {[(label, conf) for label, conf, _, _, _ in last_object_results]}")
    
    return frame

# Draw persisted object bounding boxes with color and distance
def draw_persisted_objects(frame, object_results):
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    print(f"Drawing {len(object_results)} persisted objects")
    
    for label, confidence, (x, y, w, h), color, distance in object_results:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw label with color and confidence above the box
        label_text = f"{label} ({color}) {confidence:.2f}"
        label_y = y - 10 if y - 10 > 0 else y + 20
        cv2.putText(frame, label_text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw distance line and annotation if distance is available
        if distance is not None:
            # Blue line from bottom center of box downward, clipped to frame
            line_start = (x + w // 2, y + h)
            line_end = (x + w // 2, min(y + h + 30, frame_height - 1))
            cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            
            # Distance text positioned above the bounding box
            distance_text = f"{distance:.2f} m"
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + w // 2 - text_size[0] // 2
            text_y = y - 20  # Default position above the box
            
            # Adjust text_y to stay within frame boundaries
            if text_y - text_size[1] < 10:  # If too close to top
                text_y = 10 + text_size[1]
            elif text_y > frame_height - 10:  # If too close to bottom
                text_y = frame_height - 10
            
            # Draw a black background rectangle for readability
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), 
                         (0, 0, 0), -1)
            
            # Draw distance text in blue
            cv2.putText(frame, distance_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            print(f"Drawing distance annotation for {label}: {distance:.2f} m at ({text_x}, {text_y})")
    
    return frame

# Draw persisted face bounding boxes
def draw_persisted_faces(frame, face_results):
    for (top, right, bottom, left), name in face_results:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 250), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame

# Detect and recognize faces
def recognize_faces(frame, known_face_encodings, known_face_names, audio_queue, current_frame):
    global last_face_results, last_face_update_frame
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        if face_encoding.size > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
        face_results.append(((top, right, bottom, left), name))
    
    with results_lock:
        last_face_results = face_results
        last_face_update_frame = current_frame
    
    return frame

# Draw persisted text annotations
def draw_persisted_text(frame, text_results):
    for text, (x, y) in text_results:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Draw persisted currency bounding boxes
def draw_persisted_currency(frame, currency_results):
    for denomination, confidence, (x, y, w, h) in currency_results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan box for currency
        cv2.putText(frame, f"{denomination} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return frame

# Pre-filter regions likely to contain currency using edge detection
def prefilter_currency_regions(frame, window_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_regions = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter contours by size (currency notes are roughly rectangular)
        if (w > window_size[0] * 0.5 and h > window_size[1] * 0.5) and (w < frame.shape[1] * 0.8 and h < frame.shape[0] * 0.8):
            # Expand the region slightly to ensure the entire note is captured
            x = max(0, x - 20)
            y = max(0, y - 20)
            w = min(frame.shape[1] - x, w + 40)
            h = min(frame.shape[0] - y, h + 40)
            candidate_regions.append((x, y, w, h))
    
    return candidate_regions

# Detect currency using the fine-tuned model
def detect_currency(frame, currency_model, class_labels, current_frame):
    global last_currency_results, last_currency_update_frame
    
    currency_results = []
    window_size = (224, 224)  # Match MobileNetV2 input size
    
    # Pre-filter regions likely to contain currency
    candidate_regions = prefilter_currency_regions(frame, window_size)
    
    for (x, y, w, h) in candidate_regions:
        # Adjust window to fit the model's input size
        window = frame[y:y + h, x:x + w]
        if window.shape[0] < window_size[1] or window.shape[1] < window_size[0]:
            continue
        
        # Preprocess window for the model
        window_resized = cv2.resize(window, window_size)
        window_array = img_to_array(window_resized) / 255.0
        window_array = np.expand_dims(window_array, axis=0)
        
        # Predict
        predictions = currency_model.predict(window_array, verbose=0)
        confidence = np.max(predictions)
        if confidence > CURRENCY_CONFIDENCE_THRESHOLD:
            class_idx = np.argmax(predictions)
            denomination = class_labels[class_idx]
            currency_results.append((denomination, confidence, (x, y, w, h)))
    
    # Apply non-maximum suppression to reduce overlapping boxes
    if currency_results:
        boxes = [(x, y, w, h) for _, _, (x, y, w, h) in currency_results]
        confidences = [confidence for _, confidence, _ in currency_results]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CURRENCY_CONFIDENCE_THRESHOLD, CURRENCY_NMS_THRESHOLD)
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_results = [currency_results[i] for i in indices]
        else:
            filtered_results = []
    else:
        filtered_results = []
    
    with results_lock:
        last_currency_results = filtered_results
        last_currency_update_frame = current_frame
    
    return frame

# Optimized text recognition
def live_text_recognition(frame, audio_queue, current_frame):
    global last_text_results, last_text_update_frame
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        list_words = []
        text_results = []
        confidence_score = pytesseract.image_to_data(blur, lang='fra+eng', output_type=pytesseract.Output.DICT)
        confidence_threshold = 60
        n_boxes = len(confidence_score['text'])
        for i in range(n_boxes):
            if int(confidence_score['conf'][i]) > confidence_threshold:
                text = confidence_score['text'][i].strip()
                if text:
                    x, y = confidence_score['left'][i], confidence_score['top'][i]
                    text_results.append((text, (x, y)))
                    list_words.append(text)
        if list_words:
            audio_queue.put(" ".join(list_words))
        
        with results_lock:
            last_text_results = text_results
            last_text_update_frame = current_frame
        
        return frame
    except Exception as e:
        print(f"Text recognition error: {e}")
        return frame

# Worker threads for parallel processing
def object_detection_worker(yolo_session, classes, audio_queue):
    while True:
        try:
            frame, current_frame = frame_queue_object.get(timeout=1)
            if frame is None:  # Signal to exit
                break
            frame_copy = deepcopy(frame)
            class_ids, confidences, boxes = detect_objects(yolo_session, frame_copy, classes)
            draw_boxes(frame_copy, class_ids, confidences, boxes, classes, audio_queue, current_frame)
            results_queue.put(("object", current_frame))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Object detection error: {e}")

def face_recognition_worker(known_face_encodings, known_face_names, audio_queue):
    while True:
        try:
            frame, current_frame = frame_queue_face.get(timeout=1)
            if frame is None:  # Signal to exit
                break
            if current_frame % FACE_RECOGNITION_INTERVAL == 0:
                frame_copy = deepcopy(frame)
                recognize_faces(frame_copy, known_face_encodings, known_face_names, audio_queue, current_frame)
                results_queue.put(("face", current_frame))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Face recognition error: {e}")

def currency_detection_worker(currency_model, class_labels):
    while True:
        try:
            frame, current_frame = frame_queue_currency.get(timeout=1)
            if frame is None:  # Signal to exit
                break
            if current_frame % CURRENCY_RECOGNITION_INTERVAL == 0:
                frame_copy = deepcopy(frame)
                detect_currency(frame_copy, currency_model, class_labels, current_frame)
                results_queue.put(("currency", current_frame))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Currency detection error: {e}")

def text_recognition_worker(audio_queue):
    while True:
        try:
            frame, current_frame = frame_queue_text.get(timeout=1)
            if frame is None:  # Signal to exit
                break
            if current_frame % TEXT_RECOGNITION_INTERVAL == 0:
                frame_copy = deepcopy(frame)
                live_text_recognition(frame_copy, audio_queue, current_frame)
                results_queue.put(("text", current_frame))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Text recognition error: {e}")

# Audio worker thread
def audio_worker(audio_queue, engine):
    while True:
        text = audio_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Audio error: {e}")

# Display frame
def display_frame(frame):
    try:
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(10)  # Increased delay to force window refresh
    except Exception as e:
        print(f"Display error: {e}")

# Main function
def main():
    global last_face_results, last_face_update_frame
    global last_text_results, last_text_update_frame
    global last_object_results, last_object_update_frame
    global last_currency_results, last_currency_update_frame
    global last_audio_announcements
    
    current_directory = os.getcwd()
    yolo_onnx_file = os.path.join(current_directory, "Camera Video Analysis Module", "yolo", "yolov8n.onnx")
    classes = []
    coco_names_path = os.path.join(current_directory, "Camera Video Analysis Module", "yolo", "data", "coco.names")
    try:
        with open(coco_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Could not find {coco_names_path}")
        return

    known_faces_dir = os.path.join(current_directory, "Camera Video Analysis Module", "known_faces")
    if not os.path.exists(known_faces_dir):
        print(f"Error: Known faces directory {known_faces_dir} not found. Please create it and add images.")
        return
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    if not known_face_encodings:
        print("Warning: No known faces loaded. Face recognition will not work.")

    # Load currency model
    currency_model_path = os.path.join(current_directory, "Camera Video Analysis Module", "tunisian_currency_classification", "currency_model.h5")
    currency_model = load_currency_model(currency_model_path)
    if currency_model is None:
        print("Exiting due to currency model loading failure")
        return
    class_labels = ['5dt', '10dt', '20Dt', '50dt']

    yolo_session = load_yolo_model(yolo_onnx_file)
    if yolo_session is None:
        print("Exiting due to YOLOv8 model loading failure")
        return

    engine = pyttsx3.init()
    audio_queue = queue.Queue()
    audio_thread = threading.Thread(target=audio_worker, args=(audio_queue, engine))
    audio_thread.start()

    # Start worker threads
    object_thread = threading.Thread(target=object_detection_worker, args=(yolo_session, classes, audio_queue))
    face_thread = threading.Thread(target=face_recognition_worker, args=(known_face_encodings, known_face_names, audio_queue))
    currency_thread = threading.Thread(target=currency_detection_worker, args=(currency_model, class_labels))
    text_thread = threading.Thread(target=text_recognition_worker, args=(audio_queue,))
    
    object_thread.start()
    face_thread.start()
    currency_thread.start()
    text_thread.start()

    # Connect to ESP32-CAM video stream
    stream_url = "http://192.168.254.198:81/stream"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {stream_url}")
        audio_queue.put(None)
        frame_queue_object.put((None, None))
        frame_queue_face.put((None, None))

        frame_queue_currency.put((None, None))
        frame_queue_text.put((None, None))
        object_thread.join()
        face_thread.join()
        currency_thread.join()
        text_thread.join()
        audio_thread.join()
        return

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            frame_resized = cv2.resize(frame, INPUT_FRAME_RESOLUTION)
            print(f"Processing frame {frame_count}")

            # Clear persisted results if timeout is exceeded
            with results_lock:
                if frame_count - last_face_update_frame > FACE_PERSISTENCE_FRAMES:
                    last_face_results = []
                if frame_count - last_text_update_frame > TEXT_PERSISTENCE_FRAMES:
                    last_text_results = []
                if frame_count - last_object_update_frame > OBJECT_PERSISTENCE_FRAMES:
                    last_object_results = []
                if frame_count - last_currency_update_frame > CURRENCY_PERSISTENCE_FRAMES:
                    last_currency_results = []

            # Distribute frame to worker threads
            frame_copy = deepcopy(frame_resized)
            try:
                if not frame_queue_object.full():
                    frame_queue_object.put((frame_copy, frame_count))
                if not frame_queue_face.full():
                    frame_queue_face.put((frame_copy, frame_count))
                if not frame_queue_currency.full():
                    frame_queue_currency.put((frame_copy, frame_count))
                if not frame_queue_text.full():
                    frame_queue_text.put((frame_copy, frame_count))
            except Exception as e:
                print(f"Error distributing frames to queues: {e}")

            # Small delay to allow worker threads to process
            time.sleep(0.02)

            # Draw persisted results
            with results_lock:
                print(f"Main loop - last_object_results: {[(label, conf) for label, conf, _, _, _ in last_object_results]}")
                annotated_frame = frame_resized.copy()
                if last_face_results:
                    annotated_frame = draw_persisted_faces(annotated_frame, last_face_results)
                if last_text_results:
                    annotated_frame = draw_persisted_text(annotated_frame, last_text_results)
                if last_currency_results:
                    annotated_frame = draw_persisted_currency(annotated_frame, last_currency_results)
                if last_object_results:
                    annotated_frame = draw_persisted_objects(annotated_frame, last_object_results)

            # Display and FPS
            display_frame(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Main error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Signal worker threads to exit
        frame_queue_object.put((None, None))
        frame_queue_face.put((None, None))
        frame_queue_currency.put((None, None))
        frame_queue_text.put((None, None))
        audio_queue.put(None)
        # Wait for threads to finish
        object_thread.join()
        face_thread.join()
        currency_thread.join()
        text_thread.join()
        audio_thread.join()

if __name__ == "__main__":
    main()