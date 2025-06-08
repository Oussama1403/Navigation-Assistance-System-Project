# Installation & Usage Guide for YOLOv8 Camera Analysis Module

This guide explains how to set up and use the upgraded camera analysis module with YOLOv8, face recognition, currency detection, and text/audio feedback.

---

## 1. Python Environment Setup

1. **Create and activate a Python virtual environment** (recommended):

   ```bash
   python -m venv camera_env
   ```

   - **Windows:**
     ```bash
     camera_env\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source camera_env/bin/activate
     ```

2. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

---

## 2. Install Required Python Packages

Install all dependencies:

```bash
pip install opencv-python numpy pyttsx3 pytesseract onnxruntime face_recognition tensorflow
```

- For `face_recognition`, you may need to install CMake and dlib system dependencies:
  - **Windows:** Download and install CMake from [cmake.org](https://cmake.org/download/).
  - **Linux:** `sudo apt-get install build-essential cmake`
  - **Mac:** `brew install cmake`

---

## 3. Install Tesseract OCR

- **Windows:** Download and install from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
- **Linux:**  
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-fra
  ```
- **Mac:**  
  ```bash
  brew install tesseract
  brew install tesseract-lang
  ```

---

## 4. Download Models & Data

1. **YOLOv8 ONNX Model:**
   - Download a YOLOv8 ONNX model (e.g., `yolov8n.onnx`) using the [Ultralytics export script](https://docs.ultralytics.com/models/yolov8/#export).
   - Place it in:  
     `Camera Video Analysis Module/yolo/yolov8n.onnx`

2. **COCO Class Names:**
   - Download `coco.names` and place in:  
     `Camera Video Analysis Module/yolo/data/coco.names`

3. **Known Faces Directory:**
   - Create a directory:  
     `Camera Video Analysis Module/known_faces`
   - Add images of known people (filenames as names).

4. **Currency Classification Model:**
   - Place your fine-tuned MobileNet model (e.g., `currency_model.h5`) in:  
     `Camera Video Analysis Module/tunisian_currency_classification/currency_model.h5`

---

## 5. Configure Tesseract Path (Windows Only)

If Tesseract is not in your PATH, update the following in your script:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.TESSDATA_PREFIX = r'C:\Program Files\Tesseract-OCR'
```

---

## 6. Run the Camera Analysis Script

1. Make sure your camera is connected.
2. Run the main script:
   ```bash
   python yolov8_camera_local.py
   ```
3. The script will display a window with real-time object, face, currency, and text recognition, plus audio feedback.

- **Press `q` to exit.**

---

## 7. Deactivate the Virtual Environment (Optional)

```bash
deactivate
```

---

## Notes

- For best results, ensure all model/data paths are correct.
- If you encounter errors with `face_recognition` or `onnxruntime`, check your Python version and system dependencies.
- You can export YOLOv8 ONNX models using:
  ```bash
  pip install ultralytics
  python export_yolov8.py
  ```

---















