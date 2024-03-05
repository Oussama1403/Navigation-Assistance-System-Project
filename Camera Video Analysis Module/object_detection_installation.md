a recap of the necessary steps i've taken to install and use YOLO (You Only Look Once) for object detection, along with OpenCV for image processing. understand and follow along with the installation and usage process:

### Installing YOLO and OpenCV:

official webiste: [YOLO](https://pjreddie.com/darknet/yolo/)

1. **Clone the YOLO repository**:
   ```bash
   git clone https://github.com/pjreddie/darknet
   ```

<i>NB: darknet repo renamed to yolo in the project</i>

2. **Build YOLO**:
   ```bash
   cd darknet
   make
   ```

3. **Download YOLOv3 configuration and weights files**:
    Download the pre-trained weight file here (237 MB)
    ```bash
    wget https://pjreddie.com/media/files/yolov3.weights
    ```
    [More info in "how it works section"](https://pjreddie.com/darknet/yolo/#:~:text=weights-,How%20It%20Works,-Prior%20detection%20systems)
   - Place these files in the `darknet(yolo)` directory.

5. **Installing and configuring Python virtual environment**
   In yolo(darknet) dir, create and activate python virtual env where python packages will be installed:
   
   ```bash
   python -m venv python_local_env
   ```
   Activate the python virtual env:

   Windows:
   ```bash
   python_local_env\Scripts\activate
   ```
   MacOS and Linux:
   ```bash
   source python_local_env/bin/activate
   ```

6. **Install OpenCV with GUI support**:
   After activating the local virtual env, Install Required Packages:
   ```bash
   pip install opencv-python
   ```

### Using YOLO and OpenCV:

1. **Perform object detection**:
   - The Python script (`object_detection.py`) uses OpenCV to load the YOLO model and perform object detection.
   [object_detection.py](yolo/object_detection.py)


2. **Run the script**:
    First, make sure you have the following file structure:
    object_detection.py (the Python script provided)
    yolo/yolov3.weights (pre-trained weights file)
    yolo/cfg/yolov3.cfg (YOLOv3 configuration file found at cfg/)
    yolo/data/coco.names (file containing names of the objects found at data/)
   - Run your Python script to perform real-time object detection using the YOLO model.

     ```bash
     python object_detection.py
     ```
    The script will start capturing video from your camera and perform real-time object detection using the YOLO model. Detected objects will be annotated with bounding boxes and labels displayed on the video feed.

    Deactivate the Virtual Environment (Optional)
    Once you're done working with the YOLO project, you can deactivate the virtual environment by running:
    ```bash
    deactivate
    ```















