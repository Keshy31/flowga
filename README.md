# <img src="assets/flowga_logo.png" alt="Flowga Logo" width="300"/>

Your Personal AI Yoga Instructor, right in your terminal.

Flowga is a privacy-first, local-only application that uses your webcam and the power of Vision Language Models (VLMs) to provide real-time feedback on your yoga poses. It combines a rich terminal interface with an optional live video feed to guide your practice, track your sessions, and help you improve your alignment.

> [!NOTE]
> 
> Flowga is a prototype project. While fully functional, it has not been tested on a wide range of hardware. Feedback and contributions are welcome!

## Features

- **Real-Time Pose Detection**: Uses MediaPipe to accurately detect 33 different body landmarks in real-time.
    
- **AI-Powered Feedback**: Leverages a local Vision Language Model (via Ollama) to identify poses, provide constructive feedback, and score your alignment.
    
- **Privacy First**: 100% local. No data ever leaves your machine. No internet connection required after setup.
    
- **Rich Terminal UI**: A beautiful and functional dashboard built with Rich shows your pose history, session timers, and AI feedback.
    
- **Optional Video Window**: Enable a live, mirrored webcam feed with data overlays showing your pose, score, and feedback.
    
- **Text-to-Speech (TTS)**: Get instant audio feedback on your alignment without needing to look at the screen.
    
- **Session Tracking & Logging**: Automatically tracks the duration you hold each pose and can save a detailed log of your session, including images, for later review.
    
- **Customizable Theme**: A consistent, Dracula-inspired color palette is used across the entire application, and can be easily customized in the config.
    

## System Architecture

Flowga is built on a modular pipeline that processes video data from capture to user feedback, all locally on your machine.

## How It Works

The application runs in a main loop that orchestrates several components in real-time:

1. **Input (OpenCV)**: Captures frames from your webcam.
    
2. **Detection (MediaPipe)**: Each frame is passed to Google's MediaPipe `pose` model, which extracts the 3D coordinates of 33 body landmarks. It also calculates the angles of major joints (e.g., elbows, knees) to provide quantitative data.
    
3. **Analysis (Ollama VLM)**: Periodically (e.g., every 5 seconds), the current video frame, landmark coordinates, and joint angles are sent to a local Vision Language Model running via Ollama. This analysis runs in a separate thread to keep the UI responsive. The VLM returns a structured JSON object containing the identified pose, a concise feedback tip, and an alignment score.
    
4. **Output (Rich, OpenCV, pyttsx3)**: The feedback is delivered to you through multiple channels simultaneously:
    
    - The **Terminal UI** is updated with the latest pose, score, feedback, and session timers.
        
    - The **feedback is spoken aloud** using a Text-to-Speech engine in a non-blocking thread, allowing you to focus on your pose.
        
    - If the **video window** is enabled, the feedback is drawn directly onto the video feed as a text overlay.
        
5. **Tracking & Logging**: A session manager tracks the time you spend in each pose. If logging is enabled, it saves the feedback, score, and a snapshot image for every analysis cycle to a CSV file for post-session review.
    

## Setup

### Prerequisites

- Python 3.10+
    
- [Ollama](https://ollama.com/ "null") installed and running.
    

### Installation

1. **Clone the repository:**
    
    ```
    git clone https://github.com/Keshy31/flowga
    cd flowga
    ```
    
2. **Create and activate a virtual environment:**
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    
3. **Install the required packages from `requirements.txt`:**
    
    ```
    pip install -r requirements.txt
    ```
    
4. **Pull the Ollama model:** Flowga is optimized for smaller, efficient VLMs. We recommend `qwen2.5vl:7b, llava:7b` or a similar model.
    
    ```
    ollama pull qwen2.5vl:7b
    ```
    
    _You can use other models, but ensure they support vision and are specified with the `--model` flag when running._
    

## Usage

Run the application from your terminal:

```
python yoga_tui.py
```

Press `q` in the video window or `Ctrl+C` in the terminal to end the session and see your summary.

### Command-Line Options

|   |   |   |   |
|---|---|---|---|
|**Flag**|**Argument**|**Default**|**Description**|
|`--video-window`|(none)|`False`|Display the webcam feed in a separate GUI window.|
|`--model`|`<model_name>`|`llava`|Specify which Ollama VLM to use.|
|`--analysis_interval`|`<seconds>`|`5`|Set the time between analysis calls to the VLM.|
|`--log`|(none)|`False`|Enable session logging to a CSV file and image folder.|

**Example with video window and logging:**

```
python yoga_tui.py --video-window --log
```

## Session Logging

When you run Flowga with the `--log` flag, it creates a new folder in `logs/` for your session. Inside this folder, you will find:

- A `session_{timestamp}.csv` file with detailed feedback for each analysis point.
    
- An `images/` subfolder containing the corresponding video frames with landmarks drawn on them.
    

This allows you to track your progress over time by reviewing your scores and alignment in past sessions.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.