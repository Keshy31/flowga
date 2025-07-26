import cv2
import mediapipe as mp
import ollama
import pyttsx3
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
import time
import base64
from collections import defaultdict
import argparse
import threading
import json
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Initialize Rich Console
console = Console()

# --- Constants ---
POSE_DIFFERENCE_THRESHOLD = 0.05  # Tweak this value for sensitivity

# --- Global State & Locks ---
feedback_lock = threading.Lock()
latest_feedback = {
    "pose": "Initializing",
    "feedback": "Waiting for analysis...",
    "score": 0
}
last_analyzed_landmarks = None

# --- UI Rendering Functions ---
def create_pose_table(results):
    """Create a Rich table with pose landmark data."""
    table = Table(title="Pose Landmarks",
                  show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Landmark", width=20)
    table.add_column("X", width=10)
    table.add_column("Y", width=10)
    table.add_column("Z", width=10)
    table.add_column("Visibility", width=12)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmark_name = mp.solutions.pose.PoseLandmark(i).name
            table.add_row(
                str(i),
                landmark_name,
                f"{lm.x:.2f}",
                f"{lm.y:.2f}",
                f"{lm.z:.2f}",
                f"{lm.visibility:.2f}"
            )
    else:
        table.add_row("-", "No landmarks detected", "-", "-", "-", "-")

    return table

def create_tracking_table(pose_history, current_pose, current_pose_start_time):
    """Create a Rich table for pose history and current duration."""
    table = Table(title="Pose Session Summary",
                  show_header=True, header_style="bold cyan")
    table.add_column("Pose", style="dim", width=20)
    table.add_column("Total Duration (s)", width=20)

    if not pose_history and current_pose == "N/A":
        table.add_row("No poses tracked yet", "-")
        return table

    for pose, duration in pose_history.items():
        table.add_row(pose, f"{duration:.1f}")

    if current_pose != "N/A":
        current_duration = time.time() - current_pose_start_time
        table.add_row(f"[bold green]>> {current_pose}[/bold green]",
                      f"{current_duration:.1f} (current)")

    return table

def analyze_pose(model_name, base64_image, landmarks_str):
    """Send image and landmarks to Ollama for analysis and return JSON."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": ("Analyze the yoga pose in this image. "
                                f"These are the detected landmarks: {landmarks_str}. "
                                "Respond with only a JSON object containing 'pose' "
                                "(string), 'feedback' (string), and 'score' "
                                "(int 0-10)."),
                    "images": [base64_image],
                }
            ],
            options={"temperature": 0.2},
        )
        content = response['message']['content'].strip()
        feedback_json = json.loads(content)

        if not all(k in feedback_json for k in ['pose', 'feedback', 'score']):
            raise ValueError("Incomplete JSON response from model")

        return feedback_json

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        console.print(f"[bold red]Error parsing LLM response: {e}[/bold red]")
        return {"pose": "Error",
                "feedback": "Failed to analyze pose.", "score": 0}

def are_poses_different(new_landmarks, old_landmarks, threshold):
    """Compare two sets of landmarks to see if they are different."""
    if old_landmarks is None:
        return True  # Always analyze the first pose

    # Convert to NumPy arrays for vectorized calculation
    new_lm_array = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in new_landmarks.landmark])
    old_lm_array = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in old_landmarks.landmark])

    # Calculate Euclidean distance
    distance = np.linalg.norm(new_lm_array - old_lm_array)
    return distance > threshold

def tts_worker(q, rate, volume):
    """A worker thread that processes text from a queue to speak."""
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    while True:
        try:
            text = q.get()
            if text is None:  # Sentinel value to exit
                break
            engine.say(text)
            engine.runAndWait()
        except queue.Empty:
            continue
        except Exception as e:
            console.print(f"[bold red]TTS worker error: {e}[/bold red]")
    engine.stop()

def analysis_task(model_name, base64_image, landmarks):
    """Worker function to run pose analysis in a separate thread."""
    global latest_feedback, last_analyzed_landmarks

    # Convert landmarks to a string for the prompt
    landmarks_str = ", ".join([f"({lm.x:.2f}, {lm.y:.2f})"
                               for lm in landmarks.landmark])

    feedback = analyze_pose(model_name, base64_image, landmarks_str)
    with feedback_lock:
        latest_feedback = feedback
        last_analyzed_landmarks = landmarks  # Update last analyzed landmarks

def process_frame(cap, pose_detector):
    """Read a frame, process it, and run pose detection."""
    ret, frame = cap.read()
    if not ret:
        return None, None

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_frame)
    return frame, results

def handle_analysis_request(args, frame, results, last_analysis_time, executor):
    """Check if analysis is due and, if so, start it in a new thread."""
    global last_analyzed_landmarks
    current_time = time.time()

    if current_time - last_analysis_time > args.delay and results.pose_landmarks:
        if are_poses_different(results.pose_landmarks, last_analyzed_landmarks,
                               POSE_DIFFERENCE_THRESHOLD):
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # Start analysis in a background thread using the executor
            executor.submit(analysis_task, args.model, base64_image,
                            results.pose_landmarks)

            return current_time
    return last_analysis_time

def update_ui_and_state(layout, live, args, frame, results, pose_history,
                        current_pose, current_pose_start_time, tts_queue,
                        mp_drawing):
    """Update application state and refresh all UI components."""
    global latest_feedback
    with feedback_lock:
        current_feedback = latest_feedback.copy()

    # Update pose tracking
    new_pose = current_feedback.get("pose", "N/A")
    if new_pose != current_pose and new_pose != "Error":
        if current_pose != "N/A":
            duration = time.time() - current_pose_start_time
            pose_history[current_pose] += duration
            tts_queue.put(f"Great {current_pose}, now hold.")

        current_pose = new_pose
        current_pose_start_time = time.time()

    # Build UI components
    pose_table = create_pose_table(results)
    tracking_table = create_tracking_table(pose_history, current_pose,
                                           current_pose_start_time)

    feedback_panel = Panel(
        Text(f"[bold cyan]Pose:[/bold cyan] {current_feedback['pose']}\n"
             f"[bold green]Feedback:[/bold green] {current_feedback['feedback']}\n"
             f"[bold yellow]Score:[/bold yellow] {current_feedback['score']}/10"),
        title="[bold]AI Feedback[/bold]",
        border_style="blue"
    )

    # Update layout
    layout["left_panel"]["feedback"].update(feedback_panel)
    layout["left_panel"]["tracking"].update(tracking_table)
    layout["right_panel"].update(pose_table)

    # Refresh the live display
    live.update(layout)
    live.refresh()

    # Display video frame if enabled
    if args.video_window and frame is not None:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS)
        cv2.imshow('Yoga Feed', frame)

    return current_pose, current_pose_start_time

# --- Main Application ---
def main():
    """Main function to run the Yoga TUI application."""
    parser = argparse.ArgumentParser(
        description="AI Powered Yoga Pose Feedback and Tracker")
    parser.add_argument(
        "--video-window", action="store_true",
        help="Display the webcam feed in a separate GUI window.")
    parser.add_argument(
        "--model", type=str, default="qwen2.5vl:7b",
        help="The Ollama VLM model to use for analysis.")
    parser.add_argument(
        "--delay", type=int, default=5,
        help="The interval in seconds between pose analyses.")
    parser.add_argument(
        "--tts-rate", type=int, default=150,
        help="The speaking rate for TTS feedback (words per minute).")
    parser.add_argument(
        "--tts-volume", type=float, default=1.0,
        help="The volume for TTS feedback (0.0 to 1.0).")
    args = parser.parse_args()

    # --- UI and State Variables ---
    # Create a layout for the dashboard
    layout = Layout()
    layout.split_row(
        Layout(name="left_panel", ratio=2),
        Layout(name="right_panel", ratio=1)  # For landmark table
    )
    layout["left_panel"].split_column(
        Layout(name="feedback", size=10),
        Layout(name="tracking")
    )

    # --- Tracking State ---
    pose_history = defaultdict(float)
    current_pose = "N/A"
    current_pose_start_time = time.time()
    last_analysis_time = 0

    # Display Welcome Message
    ascii_art = r"""
[bold deep_sky_blue1]
███████╗██╗      ██████╗ ██╗    ██╗ ██████╗  █████╗
██╔════╝██║     ██╔═══██╗██║    ██║██╔════╝ ██╔══██╗
█████╗  ██║     ██║   ██║██║ █╗ ██║██║  ███╗███████║
██╔══╝  ██║     ██║   ██║██║███╗██║██║   ██║██╔══██║
██║     ███████╗╚██████╔╝╚███╔███╔╝╚██████╔╝██║  ██║
╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝[/]
"""
    console.print(Text(ascii_art, justify="center"))
    console.print(Text("Your Personal AI Yoga Instructor",
                       justify="center", style="italic bright_blue"))
    console.print(f"\nUsing model: [bold cyan]{args.model}[/bold cyan]")
    if args.video_window:
        console.print("Video window display is [bold green]enabled[/bold green].")
    else:
        console.print("Video window display is [bold red]disabled[/bold red].")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[bold red]Error: Could not open webcam.[/bold red]")
        return

    if args.video_window:
        cv2.namedWindow('Yoga Feed', cv2.WINDOW_NORMAL)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- TTS Setup ---
    tts_queue = queue.Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, args.tts_rate, args.tts_volume))
    tts_thread.daemon = True
    tts_thread.start()

    # --- Thread Pool for Analysis ---
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        with Live(layout, console=console, screen=True,
                  auto_refresh=False) as live:
            while True:
                frame, results = process_frame(cap, pose_detector)
                if frame is None:
                    print("Failed to grab frame. Exiting.")
                    break

                last_analysis_time = handle_analysis_request(
                    args, frame, results, last_analysis_time, executor)

                current_pose, current_pose_start_time = update_ui_and_state(
                    layout, live, args, frame, results, pose_history,
                    current_pose, current_pose_start_time, tts_queue,
                    mp_drawing
                )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # --- Signal threads to exit and cleanup ---
        executor.shutdown(wait=True)
        tts_queue.put(None)
        tts_thread.join(timeout=2)  # Wait for the TTS thread to finish

        # Cleanup
        cap.release()
        if args.video_window:
            cv2.destroyAllWindows()

        # --- Display Final Session Summary ---
        console.print("\n[bold magenta]Session Complete![/bold magenta]")
        final_summary_table = create_tracking_table(pose_history, "N/A", 0)
        summary_panel = Panel(final_summary_table,
                              title="Final Session Summary",
                              border_style="green")
        console.print(summary_panel)
        console.print("\n[bold green]Session ended. Goodbye![/bold green]")


if __name__ == "__main__":
    main()
