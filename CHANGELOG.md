# Changelog

## June 8, 2025
- Upgraded Camera module to use YOLOv8 for more accurate and faster object detection.
- Integrated Tunisian currency classification using a fine-tuned MobileNet model.
- Implemented real-time distance estimation for detected objects using the pinhole camera model.
- Added face recognition capability to the Camera module for identifying known individuals.
- Enhanced Camera module performance and responsiveness by introducing multi-threading for concurrent processing.
- Updated `camera_analysis_documentation.md` and `camera_analysis_installation.md` to reflect recent changes and new features.

## May 1, 2024
- Added a C code in `Microcontroller/camera_server.ino` for esp32-cam that is responsible for capturing the video from camera and sending it to the pc to the **Camera Video Analysis Module**.

## Apr 4, 2024
- Enhance Camera Analysis Module with openimages dataset for more recognition of diverse data
## Apr 1, 2024
- Added Face Recognition module.

## March 28, 2024
- Added `text_to_audio()` To output the recognized text as audio that the user can hear.
- Updated documentation and installation files.

## March 27, 2024
- Implemented `live_text_recognition()`;Incorporated optical character recognition (OCR) to detect and read text in the environment.

## March 18, 2024
- Started working on the `Navigation module`.
- Successfully pushed `main.py` file, which geocodes any destination and generates routes between any origin position and destination position using the `HERE Geocode API` and `HERE Routing API`.
- Refer to the Navigation module documentation file for more details about APIs used and terminology explanation.

## March 13, 2024
- Updated `audio_to_text.py`.
- Add new functionality to filter and extract destinations based on matches between the recognized text and the entries in destination list avoiding the complexity of NER.
- Added `destinations.json`.
- Updated `README.md`- more simple and clean readme file.
- Cleaner and more Robust code.

## March 5, 2024
- Updated TODO.md.
- Organize project's files.


## March 4, 2024
- Pushed speech recognition module.
- Included `audio_to_text.py` Python script for speech-to-text conversion.
- Created this `CHANGELOG.md`.
- Pushed `requirements.txt` file for Python environment setup.

## March 1, 2024
- Pushed camera video analysis module.
- Included object detection Python script with documentation.
- Created `STEPS.md` file stating steps of installation.
- Added `TODO.md` file outlining future tasks.

