Architecture for the **indoor and outdoor Navigation Assistance System Project.**

Introduction **:**

The **Indoor and outdoor Navigation Assistance System** aims to provide blind or visually impaired individuals with comprehensive navigation support in indoor and outdoor environments. This document outlines the proposed architecture for the system, leveraging a combination of hardware components, software modules, integration mechanisms, and privacy and security considerations.

Team Members:

BEN SASSI Oussama.

FATMI Ghanem.

BEN ABDELJELIL mohamed yassine.

1\. Hardware Components:

1. **Camera:**

A wearable device equipped with a camera to capture live video of the user's surroundings and streams it to the PC for analysis.

2. **Microphone:**

Listens to the user's voice commands regarding their desired destination.

3. **Voice Output Device:**

Headphones or bone conduction headsets for delivering real-time audio instructions.

4. **PC:**

Processes video input from the camera and audio input from the microphone, performs AI analysis to recognize objects and speech through speech recognition, and interacts with external services such as Google Maps API for navigation instructions. Additionally, it sends audio instructions to the audio output device for user guidance.

2\. Software Components:

1. **Camera Video Analysis Module:**

Receives live video from the camera and processes it using AI algorithms (computer vision and machine learning techniques) to recognize objects and obstacles in the environment.

2. **Speech Recognition Module:**

Listens to voice commands from the microphone and converts them into text using speech recognition algorithms.

3. **Navigation Logic:**
1. Determines the user's current location based on the analysis of the camera video or (PC’s GPS) and determines the desired destination based on the recognized speech.
1. Calculates the optimal route to the destination using Google Maps API and generates navigation instructions.
1. Receives information from the video analysis, such as obstacles or other environmental objects, and generates audio instructions about the environment.

   4. **Voice Output Module:**

Converts navigation instructions into audio format and delivers them to the user through the voice output device.

3\. Integration and Communication:

1. **Camera-PC Integration:**

Establishes communication between the camera and the PC for streaming live video feed.

2. **Microphone-PC Integration:**

Sends voice commands from the microphone to the PC for speech recognition.

3. **PC-Audio Output Device Integration:**

Sends audio instructions from the PC to the audio output device, where the user will listen to the navigation guidance.

4. **PC-External Service Integration:**

Interacts with external services such as Google Maps API for obtaining navigation instructions.

4\. illustrating the Integration and Communication Section:

1. The camera captures the live video feed of the user's surroundings.
2. The Camera Feed Processing Module preprocesses the video feed to enhance quality and reduce noise. (optional)
2. The preprocessed video feed is then passed to the AI Algorithms (Camera Video Analysis Module).
2. The AI Algorithms analyze the video feed to extract relevant information about the user's environment, such as obstacles, landmarks, and text.
2. The microphone records voice commands from the user.
2. Voice commands are processed by the Speech Recognition Module on the PC.
2. The Navigation Logic module on the PC processes the analyzed data from both **Camera Video Analysis Module** and **Speech Recognition Module** and interacts with external services, such as the **Google Maps API** to determine the user's location and generate real-time navigation instructions.
2. These navigation instructions are converted into audio format by the Voice Output Module on the PC.
2. The audio instructions are delivered to the user through the Voice Output Device.

**This diagram illustrates how information flows through the various components of the system:**

![diag](diag.png)

5. Privacy and Security:
- **Data Encryption:**

Encrypts sensitive user data, including live video feeds and location information, to protect user privacy.

6. Testing and Feedback Mechanisms:
- **User Testing:**

Conducts rigorous testing with visually impaired users to evaluate effectiveness, usability, and accessibility.

- **Feedback Mechanisms:**

Implements mechanisms for users to provide feedback on their experience with the system, facilitating iterative improvements.

7. Documentation and Support:
- **User Manuals:**

Provides comprehensive documentation and user manuals to guide users on how to use the system effectively.
