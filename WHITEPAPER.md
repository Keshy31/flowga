# Flowga: The Local AI Yoga Instructor

**Version 1.0**

**Date: July 26, 2025**

## Abstract

Flowga is a decentralized, privacy-first yoga instruction application that leverages local-only Vision Language Models (VLMs) and real-time computer vision to provide personalized feedback on yoga poses. Operating entirely on a user's machine without any cloud dependency, Flowga ensures that all personal data remains private. This whitepaper details the Flowga system architecture, a multi-layered stack that combines real-time pose estimation, quantitative joint analysis, and AI-driven qualitative feedback. At its core, Flowga uses the MediaPipe framework for high-fidelity body landmark detection and a local VLM, accessed via Ollama, to interpret pose data and generate actionable, corrective advice. This document provides a technical specification of the data pipeline, analysis engine, user interface, and the privacy-preserving principles that define the Flowga experience.

## 1. Introduction

In an age where digital wellness applications often require data to be sent to the cloud, Flowga presents a resilient, offline-first alternative. It is designed for individuals seeking to improve their yoga practice in the privacy of their own homes, without the need for an internet connection or expensive subscriptions. The application provides the benefits of a personal yoga instructor by analyzing a user's form through a standard webcam and offering immediate, actionable feedback.

The design goals of the Flowga Protocol are:

- **Privacy-by-Design:** All processing, from video capture to AI analysis, must occur locally on the user's device. No data should ever be transmitted to external servers.
    
- **Real-Time Feedback:** The system must provide corrective guidance with minimal latency (<2 seconds) to allow users to adjust their poses mid-form.
    
- **High-Accuracy Pose Estimation:** The protocol must accurately detect and track key body landmarks to form a reliable basis for analysis.
    
- **Actionable, Qualitative Analysis:** The feedback should go beyond simple metrics, offering qualitative, easy-to-understand advice for improving alignment and form.
    
- **Accessibility:** The system should run on consumer-grade hardware and provide feedback through multiple modalities (visual and auditory) to cater to different user needs.
    
- **Resilience:** The application must be fully functional without an active internet connection after the initial setup.
    

This paper specifies the technical architecture and data flows designed to achieve these objectives.

## 2. System Architecture

The Flowga system is designed as a modular, multi-layered pipeline that processes video data from capture to user feedback.

- **Video Input Layer:** Captures the raw video feed from the user's webcam using OpenCV. This layer is responsible for frame acquisition and pre-processing.
    
- **Pose Estimation Layer:** Utilizes Google's MediaPipe framework to detect 33 distinct 3D body landmarks from each video frame. It also performs quantitative analysis, such as calculating joint angles.
    
- **Analysis & Feedback Layer:** This is the core of the system. It packages the landmark data, joint angles, and the video frame into a prompt for a local Vision Language Model (VLM). The VLM identifies the pose, provides a score, and generates corrective feedback. This layer also manages session state and logging.
    
- **User Interface Layer:** Presents the feedback to the user through multiple channels: a rich Terminal User Interface (TUI) built with `rich`, an optional mirrored video feed with data overlays, and Text-to-Speech (TTS) audio cues using `pyttsx3`.
    

## 3. Pose Detection and Landmark Analysis

The foundation of Flowga's feedback system is its ability to accurately perceive and quantify the user's body position. This is handled by the Pose Estimation Layer.

### 3.1. Landmark Extraction with MediaPipe

Flowga uses the `mp.solutions.pose` model from the MediaPipe library. For each frame received from the video input layer, the model identifies the coordinates of 33 key body landmarks.

|   |   |
|---|---|
|**Landmark Type**|**Examples**|
|**Face**|`nose`, `left_eye`, `right_eye`, `mouth_left`|
|**Arms**|`left_shoulder`, `right_elbow`, `left_wrist`|
|**Torso**|`left_hip`, `right_hip`, `left_shoulder`|
|**Legs**|`left_knee`, `right_ankle`, `left_foot_index`|

These landmarks provide a comprehensive skeletal representation of the user, which is essential for detailed analysis. The visibility of each landmark is also recorded, allowing the system to handle partial occlusions.

### 3.2. Quantitative Analysis: Joint Angles

To provide the VLM with concrete data for its analysis, Flowga calculates the angles of major body joints in real-time. The angle between three points (e.g., shoulder, elbow, wrist) is calculated using the dot product of the vectors formed by the points.

`Angle = arccos(((P1 - P2) · (P3 - P2)) / (|P1 - P2| * |P3 - P2|))`

Where `P2` is the joint (mid-point), and `P1` and `P3` are the connected points. This quantitative data is crucial for the AI to make objective assessments of alignment, such as determining if an arm is straight or a knee is bent at the correct angle for a specific pose.

## 4. AI-Powered Feedback Generation

The most innovative aspect of Flowga is its use of a Vision Language Model (VLM) to interpret complex pose data and generate human-like feedback. This entire process runs locally via the Ollama framework.

### 4.1. Model and Prompting

The system is optimized for smaller, efficient VLMs (e.g., `llava`, `qwen2-vl:7b`). The analysis is triggered periodically when a stable pose is detected. A multi-modal prompt is constructed and sent to the VLM, containing:

1. **The Base64-encoded video frame:** This provides the visual context.
    
2. **A textual list of all 33 landmarks and their coordinates:** This provides precise positional data.
    
3. **A list of calculated joint angles:** This provides quantitative metrics.
    
4. **A system instruction:** This guides the model's analysis.
    

A snippet of the system instruction is as follows:

```
"You are a yoga instructor. Based on the image and the provided landmark data, identify the yoga pose.
Provide a score from 0.0 to 1.0 for alignment.
Offer one concise, actionable tip for improvement.
Respond ONLY in the following JSON format:
{\"pose\": \"<pose_name>\", \"feedback\": \"<your_tip>\", \"score\": <score_value>}"
```

### 4.2. Local Inference with Ollama

The prompt is sent to the locally running Ollama instance. The VLM processes the image and text to produce a structured JSON response. This local-first approach guarantees user privacy, as the images and pose data never leave the user's machine. The JSON structure ensures that the model's output can be reliably parsed and integrated into the UI.

### 4.3. Feedback Loop

The AI analysis runs in a separate, non-blocking thread to ensure the UI remains responsive. Once the JSON feedback is received, it is pushed to a queue, from which the main UI thread reads and displays the information. This architecture ensures a smooth, real-time experience for the user.

## 5. User Interface and Experience

Flowga is designed with a "retro-terminal" aesthetic, but its functionality is thoroughly modern. It provides feedback through several coordinated channels.

### 5.1. The Terminal User Interface (TUI)

The primary interface is a dashboard within the user's terminal, built using the Python `rich` library. The TUI is organized into panels that display:

- **Live AI Feedback:** The identified pose, score, and corrective tip from the VLM.
    
- **Session Tracking:** A table showing all poses performed during the session and the cumulative time spent in each.
    
- **Landmark Data:** A real-time table of all 33 detected landmarks and their visibility scores, useful for debugging and advanced users.
    

### 5.2. Optional Video Window

For users who prefer a visual reference, Flowga can launch a separate OpenCV window that displays the live, mirrored webcam feed. The AI's feedback (pose name, score, and tip) is overlaid directly onto this video feed, allowing users to see themselves and the feedback simultaneously.

### 5.3. Text-to-Speech (TTS) Feedback

To allow users to focus on their practice without looking at a screen, Flowga incorporates a Text-to-Speech engine (`pyttsx3`). The corrective feedback generated by the AI is spoken aloud in a non-blocking thread. The system is designed to interrupt previous feedback if a new tip is generated, ensuring the user always receives the most current advice.

## 6. Session Tracking and Data Logging

To help users monitor their progress over time, Flowga includes a session management and logging layer.

### 6.1. Pose History and Summary

The application maintains an in-memory dictionary that tracks the cumulative time spent in each identified pose. When the user ends a session, a final summary table is displayed in the TUI, showing the total duration for each pose and the overall session length.

### 6.2. Data Logging for Analysis

Flowga includes an optional logging feature. When enabled, it saves key data from the session to a CSV file for later analysis. Each row in the CSV corresponds to a single frame where feedback was generated and includes:

|   |   |   |
|---|---|---|
|**Field**|**Type**|**Description**|
|`timestamp`|Float|The Unix timestamp of the frame.|
|`pose`|String|The pose name identified by the AI.|
|`feedback`|String|The corrective feedback from the AI.|
|`score`|Float|The alignment score from the AI (0.0-1.0).|
|`image_path`|String|The path to the saved frame image with landmarks drawn.|

This structured logging provides a rich dataset for users who wish to track their improvement or for developers looking to fine-tune the AI models.

## 7. Security and Privacy Considerations

- **No Data Transmission:** The core security principle of Flowga is that no user data is ever transmitted off the device. All video processing and AI inference are 100% local.
    
- **Local Model Storage:** The VLM is managed by Ollama and stored on the user's local disk, ensuring the model itself is also under the user's control.
    
- **Ephemeral Data:** Video frames and landmark data are processed in-memory and are only persisted to disk if the user explicitly enables the session logging feature.
    
- **No User Accounts:** The application does not require any form of registration or user account, eliminating the risk of centralized data breaches.
    

## 8. Conclusion

The Flowga protocol provides a robust, private, and effective framework for at-home yoga instruction. By combining state-of-the-art computer vision with the analytical power of local Vision Language Models, it delivers personalized, real-time feedback without compromising user privacy. Its modular architecture and multi-modal user interface make it an accessible and powerful tool for personal wellness. Future work may include expanding the library of recognized poses, fine-tuning the VLM on custom datasets for improved accuracy, and exploring deployment on low-power edge devices.