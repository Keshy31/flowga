---
tags:
  - ai
  - programming
---
## Project Overview
This project develops a local, AI-powered yoga assistant application that provides real-time feedback on pose alignment and tracks session statistics (e.g., poses performed and durations). It runs primarily as a terminal-based Text User Interface (TUI) with a retro, command-line aesthetic inspired by tools like Claude Code (e.g., ASCII art headers, colored text, code-like formatting). For usability, it includes an optional GUI window for live webcam video with overlays, while keeping stats and feedback in the terminal.

**Goals**:
- Build a minimum viable product (MVP) in one day (~6-8 hours) for prototyping.
- Leverage open-source LLMs via Ollama for privacy and local execution.
- Utilize the team's hardware (OMEN HP Gaming Laptop with RTX 4080 GPU, 32GB RAM) for GPU-accelerated performance.
- Enable easy demo videos for sharing (e.g., via screen recording).

**Scope**:
- Core Features: Webcam input, pose detection, AI analysis/feedback, tracking, TTS output, terminal dashboard.
- Out of Scope (for MVP): Multi-user support, mobile port, advanced AR, cloud integration.
- Assumptions: Team has Python experience; Ollama installed and model pulled; access to webcam.

**Team Roles**:
- Engineer: Implement code per build steps.
- Tester: Run incremental tests, validate on hardware.
- Designer/Reviewer: Ensure aesthetic matches (e.g., terminal styling).

**Timeline**: One-day sprint; extend if needed for polish.
- Morning: Setup + Core Modules (Steps 1-4).
- Afternoon: Add-Ons + Integration (Steps 5-7).
- Evening: Full Testing + Demo Prep.

**Risks**:
- Latency on large models: Mitigate with quantization/smaller variants.
- Webcam compatibility: Test early.
- TTS blocking: Use threading if issues arise.

## Tech Stack
- **Language**: Python 3.12+.
- **Libraries** (install via pip in venv):
  - `opencv-python`: For webcam capture and video processing.
  - `mediapipe`: For real-time pose landmark detection.
  - `ollama`: Python client for local LLM inference.
  - `pyttsx3`: Offline text-to-speech.
  - `rich`: For styled terminal UI (colors, tables, live updates).
  - `argparse`: Built-in for CLI flags.
  - Optional: `numpy` for landmark comparisons (if not already included).
- **External Tools**:
  - Ollama binary: For serving VLMs (e.g., pull `qwen2-vl:7b` quantized).
  - Demo Recording: Asciinema (for terminal) or OBS Studio (for full screen).
- **Hardware Requirements**: GPU-enabled machine (RTX 4080 for acceleration); webcam.

## System Design Recap
- **Architecture**: Modular, event-driven loop in a single script (`yoga_tui.py`).
- **Data Flow**:
  1. Input (OpenCV) → Frame capture.
  2. Detection (MediaPipe) → Landmarks extraction.
  3. Analysis (Ollama VLM) → Feedback generation (throttled queries).
  4. Tracking (Custom) → Pose duration logging.
  5. Output (pyttsx3 + Rich/OpenCV) → TTS + Terminal stats + Optional video window.
- **Aesthetic**: Terminal as dashboard (ASCII headers, colored feedback like Claude Code); video in separate window for mirror view with overlays.
- **CLI Usage**: `python yoga_tui.py --video-window --model qwen2-vl:7b`.

## Build Plan
Follow this sequenced plan for incremental development. Each step includes subtasks with checkboxes. Test after each step before proceeding. Code in `yoga_tui.py`; commit to Git for version control.

### Step 1: Setup and Environment (30-45 mins)
- [x] Create project directory and virtual environment: `python -m venv yoga_tui_env`.
- [x] Activate venv and install dependencies: `pip install opencv-python mediapipe ollama pyttsx3 rich`.
- [x] Add script skeleton with imports:
  ```python
  import cv2
  import mediapipe as mp
  import ollama
  import pyttsx3
  from rich.console import Console
  from rich.live import Live
  from rich.panel import Panel
  from rich.text import Text
  from rich.table import Table
  import time
  import os
  import base64
  from collections import defaultdict
  import argparse
  import threading  # For potential non-blocking TTS
  console = Console()
  ```
- [x] Add argparse for flags (e.g., `--video-window`, `--model` default 'qwen2-vl:7b').
- [x] Print welcome message in styled ASCII (use Rich Text/Panel for blocky header like "YOGA TUI").
- **Test**: Run script; ensure no errors, see styled welcome in terminal.

### Step 2: Input Module - OpenCV for Frame Ingestion (45 mins)
- [x] Implement webcam capture: `cap = cv2.VideoCapture(0)`; add mirror flip `frame = cv2.flip(frame, 1)`.
- [x] Create main loop: `while True: success, frame = cap.read(); if not success: break;`.
- [x] If `--video-window`: Initialize `cv2.namedWindow('Yoga Feed')`.
- [x] For terminal: Print frame metadata (e.g., "Frame: 720p @ FPS" in Rich Text).
- [x] Add exit handling: `if cv2.waitKey(1) & 0xFF == ord('q'): break` (if window enabled).
- **Test**: Run with/without `--video-window`. Without: Terminal logs frames. With: Window shows live mirror feed. Wave to confirm.

### Step 3: Detection Module - MediaPipe for Landmarks (45 mins)
- [ ] Initialize MediaPipe: `mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)`.
- [ ] In loop: `results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))`.
- [ ] Extract landmarks: If `results.pose_landmarks`, get list of dicts with joint names and coords.
- [ ] For terminal: Build Rich Table for landmarks (columns: Joint, X, Y, Visibility); add simple ASCII stick figure (custom function mapping coords to char grid).
- [ ] For video window (if enabled): Draw landmarks using `mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks)`.
- [ ] Update display: Terminal prints table; window shows overlaid frame via `cv2.imshow('Yoga Feed', frame)`.
- **Test**: Perform basic poses (e.g., T-pose). Terminal shows table/ASCII; window (if on) shows skeleton. Ensure real-time (<50ms lag).

### Step 4: Analysis Module - Ollama VLM for Inference (1 hour)
- [ ] Create analyze function: Encode frame to base64; build prompt with landmarks (e.g., "Analyze yoga pose image with keypoints: [list]. Output JSON: {'pose': 'name', 'feedback': 'tips', 'score': 1-10}").
- [ ] Query: `response = ollama.chat(model=args.model, messages=[{'role': 'user', 'content': prompt, 'images': [base64_image]}])`.
- [ ] Parse JSON from response; throttle calls (e.g., every 5s using time check).
- [ ] For terminal: Print parsed feedback in Rich Syntax (JSON highlighted like code).
- **Test**: Use static test image first (load via cv2.imread); then integrate to loop. Query on a yoga photo—verify sensible JSON in terminal.

### Step 5: Tracking Module - Custom Logic for Session State (45 mins)
- [ ] Initialize state: `pose_history = defaultdict(list)`; `current_pose = None; start_time = None`.
- [ ] Detect changes: Compare current landmarks to previous (e.g., using numpy.linalg.norm on coord diffs; threshold 0.1).
- [ ] On change or new analysis: If new pose, log duration for old (`pose_history[old_pose].append(time.time() - start_time)`); reset current.
- [ ] For terminal: Use Rich Progress for current timer; Table for running summary.
- **Test**: Mock landmarks; simulate pose changes—see timer/progress update in terminal. Integrate; do real poses and check logs.

### Step 6: Output Module - pyttsx3 + Feedback (45 mins)
- [ ] Initialize TTS: `engine = pyttsx3.init()`.
- [ ] On feedback: `engine.say(feedback['feedback']); engine.runAndWait()` (use threading if blocking: `threading.Thread(target=engine.say, args=(text,)).start()`).
- [ ] For terminal: Use Rich Live for dashboard (panels: Current Pose, Feedback [colored by score], Tracking Table).
- [ ] For video: Overlay text on frame (e.g., `cv2.putText(frame, f"Pose: {pose} - {feedback}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)`).
- **Test**: Generate sample feedback; hear TTS + see terminal/video updates. Ensure no loop blocking.

### Step 7: Orchestrator - Integration and Polish (30 mins)
- [ ] Tie in main loop: Input → Detection → (If stable) Analysis → Tracking → Output.
- [ ] Add session start/end: Welcome screen; on exit, print full summary Table.
- [ ] Handle errors (e.g., no detection: "No pose found" in red).
- [ ] Aesthetic Polish: ASCII header on start; colored elements (green good, red corrections).
- [ ] Cleanup: `cap.release(); cv2.destroyAllWindows()`.
- **Test**: Full end-to-end run. With `--video-window`: Terminal dashboard + video overlays. Without: Pure terminal. Record demo video.

## Testing Plan
- **Unit/Incremental**: As per each step; use sample images/poses.
- **Integration**: Test latency (<2s feedback); accuracy on 5-10 yoga poses (e.g., Warrior II, Downward Dog).
- **Edge Cases**: Poor lighting (adjust OpenCV brightness?); no person; fast movements.
- **Performance**: Monitor GPU with `nvidia-smi`; aim for 20-30 FPS.
- **Tools**: Pytest for scripts if time; manual for UI.

## Deployment and Demo
- **Run**: `python yoga_tui.py --video-window` from terminal.
- **Demo Videos**: Use OBS to capture screen (terminal + window); asciinema for terminal-only (`asciinema rec demo.cast`).
- **Sharing**: Upload videos to X/YouTube; include script in Git repo.
- **Extensions (Post-MVP)**:
  - Add more poses via prompt engineering.
  - Mobile export (e.g., via Kivy).
  - Fine-tune VLM if needed.

This plan provides self-contained direction—assign steps to team members and review progress at each test point. If clarifications needed, reference conversation history.