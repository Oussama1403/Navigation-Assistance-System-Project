# Indoor and Outdoor Navigation System Project Todo List


## Camera Video Analysis Module
- [x] Complete initial steps in object detection using yolo model.
- [ ] Collect more diverse data from the environment (e.g., trees, cars, pedestrians).
- [ ] Train the object detection model using the collected and labeled data.
- [ ] Label the collected data with appropriate annotations for training.
- [ ] Generate text data about object detected.
- [ ] Implement Threading module for more paralell and faster detection.
- [ ] Evaluate the trained model's performance on test datasets.
- [ ] Test the code thoroughly with various input scenarios and validate the accuracy and efficiency of object detection.

## Speech Recognition Module
- [x] Research and select a suitable speech recognition library or API.
- [x] Implement functionality to filter and extract destinations.
- [ ] Add more places to `destinations.json`
- [ ] Integrate the selected speech recognition solution into the navigation system.
- [ ] Test the speech recognition functionality in different environments and conditions.
- [ ] Implement error handling and feedback mechanisms for speech recognition failures.

## Navigation Module
- [ ] Implement google maps API for generating routes to destinations. (ON HOLD - GMAPS IS NOT FREE)
- [x] Implement alternative for google maps API (OSM API + HERE Geocode API)
- [ ] Integrate Both APIs for full navigation system.
- [ ] Implement natural language generation techniques for generating audio-based navigation instructions based on Speech Recognition results.
- [ ] Test the navigation instructions generation system with simulated and real-world scenarios.

## Voice Output Module
- [ ] Convert text instructions to audio instructions.

## Microcontroller
- [ ] Develop a python script that is responsible for capturing the video from camera and sending it to the pc to the **Camera Video Analysis Module**
- [ ] Develop a python script that is responsible for capturing the audio from micro and sending it to the pc to the **Speech Recognition Module**
- [ ] Develop a python file to receive upcoming audio instructions from pc (objects and navigation audio instructions) and delivering them to the Voice Output Device.