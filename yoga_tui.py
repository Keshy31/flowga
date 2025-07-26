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
import os
import base64
from collections import defaultdict
import argparse
import threading
import json

# Initialize Rich Console
console = Console()

# Canonical landmark names from MediaPipe
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

def create_pose_table(results):
    """Creates a Rich table to display pose landmarks."""
    table = Table(title="Pose Landmarks", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=3)
    table.add_column("Name", width=18)
    table.add_column("X", width=8)
    table.add_column("Y", width=8)
    table.add_column("Z", width=8)
    table.add_column("Visibility", width=10)

    if results.pose_landmarks:
        for i, (name, lm) in enumerate(zip(landmark_names, results.pose_landmarks.landmark)):
            table.add_row(
                str(i),
                name,
                f"{lm.x:.2f}",
                f"{lm.y:.2f}",
                f"{lm.z:.2f}",
                f"{lm.visibility:.2f}"
            )
    else:
        for i, name in enumerate(landmark_names):
            table.add_row(str(i), name, "-", "-", "-", "-")
    
    return table

def create_tracking_table(pose_history, current_pose, current_pose_start_time):
    """Creates a Rich table for pose history and current duration."""
    table = Table(title="Pose Session Summary", show_header=True, header_style="bold cyan")
    table.add_column("Pose", style="dim", width=20)
    table.add_column("Total Duration (s)", width=20)

    for pose, duration in pose_history.items():
        table.add_row(pose, f"{duration:.1f}")

    if current_pose != "N/A":
        current_duration = time.time() - current_pose_start_time
        # Add a row for the currently held pose
        table.add_row(f"[bold green]{current_pose} (current)[/bold green]", f"{current_duration:.1f}")

    return table

def analyze_pose(model_name, base64_image, landmarks_str):
    """Analyzes the pose using Ollama VLM and returns structured feedback."""
    try:
        prompt = (
            "You are a yoga instructor AI. Analyze the provided image and the keypoints of the person's pose. "
            "Identify the yoga pose. Provide specific, constructive feedback on their alignment. "
            "Finally, give a score from 1 to 10 on the accuracy of the pose. "
            f"Here are the landmarks: {landmarks_str}. "
            "Respond ONLY with a single JSON object in the format: "
            '{"pose": "<pose_name>", "feedback": "<your_feedback>", "score": <score_number>}'
        )

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [base64_image]
                }
            ]
        )
        
        # Extract and parse the JSON content
        content = response['message']['content']
        feedback_json = json.loads(content)
        return feedback_json

    except Exception as e:
        console.print(f"[bold red]Error during analysis: {e}[/bold red]")
        return None

def main():
    """Main function to run the Yoga TUI application."""
    parser = argparse.ArgumentParser(description="AI Powered Yoga Pose Feedback and Tracker")
    parser.add_argument("--video-window", action="store_true", help="Display the webcam feed in a separate GUI window.")
    parser.add_argument("--model", type=str, default="qwen2.5vl:7b", help="The Ollama VLM model to use for analysis.")
    args = parser.parse_args()

    # --- UI and State Variables ---
    # Create a layout for the dashboard
    layout = Layout()
    layout.split_row(
        Layout(name="left_panel", ratio=2),
        Layout(name="right_panel", ratio=1) # For landmark table
    )
    layout["left_panel"].split_column(
        Layout(name="feedback", size=10),
        Layout(name="tracking")
    )

    # Shared state for feedback from the analysis thread
    latest_feedback = {"pose": "N/A", "feedback": "Waiting for analysis...", "score": 0}
    analysis_thread = None

    # --- Tracking State ---
    pose_history = defaultdict(float)
    current_pose = "N/A"
    current_pose_start_time = time.time()

    # Display Welcome Message
    welcome_message = Text("YOGA TUI", justify="center", style="bold magenta")
    welcome_panel = Panel(welcome_message, title="Welcome", border_style="green")
    console.print(welcome_panel)
    console.print(f"Using model: [bold cyan]{args.model}[/bold cyan]")
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
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Analysis variables
    last_analysis_time = 0
    analysis_interval = 5  # seconds

    try:
        with Live(layout, console=console, screen=True, auto_refresh=False) as live:
            while True:
                success, frame = cap.read()
                if not success:
                    console.print("[bold yellow]Warning: Could not read frame from webcam.[/bold yellow]")
                    break

                # Mirror the frame for a more intuitive display
                frame = cv2.flip(frame, 1)

                # Process frame with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                # Draw landmarks and run analysis
                if results.pose_landmarks:
                    if args.video_window:
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    current_time = time.time()
                    # Check if it's time for a new analysis and if the previous one is done
                    if current_time - last_analysis_time > analysis_interval and (analysis_thread is None or not analysis_thread.is_alive()):
                        last_analysis_time = current_time

                        # Encode frame and prepare data for the thread
                        _, buffer = cv2.imencode('.jpg', frame)
                        base64_image = base64.b64encode(buffer).decode('utf-8')
                        landmark_list = [f"({lm.x:.2f}, {lm.y:.2f})" for lm in results.pose_landmarks.landmark]
                        landmarks_str = ", ".join(landmark_list)

                        # Define the analysis task for the thread
                        def analysis_task():
                            feedback = analyze_pose(args.model, base64_image, landmarks_str)
                            if feedback:
                                latest_feedback.update(feedback)

                        # Start the analysis in a new thread
                        analysis_thread = threading.Thread(target=analysis_task)
                        analysis_thread.start()

                # --- Check for pose change and update history ---
                if latest_feedback["pose"] != "N/A" and latest_feedback["pose"] != current_pose:
                    # Log duration for the previous pose
                    if current_pose != "N/A":
                        duration = time.time() - current_pose_start_time
                        pose_history[current_pose] += duration
                    
                    # Update to the new pose
                    current_pose = latest_feedback["pose"]
                    current_pose_start_time = time.time()

                # --- Update Dashboard ---
                # Update the landmark table
                pose_table = create_pose_table(results)
                layout["right_panel"].update(pose_table)

                # Update the feedback panel with dynamic colors
                score = latest_feedback.get('score', 0)
                if score >= 8:
                    border_style = "green"
                elif score >= 5:
                    border_style = "yellow"
                else:
                    border_style = "red"

                feedback_panel = Panel(
                    Text(f"Pose: {latest_feedback['pose']}\nFeedback: {latest_feedback['feedback']}\nScore: {latest_feedback['score']}/10", style="white"),
                    title="AI Feedback",
                    border_style=border_style
                )
                layout["feedback"].update(feedback_panel)

                # Update the tracking table
                tracking_table = create_tracking_table(pose_history, current_pose, current_pose_start_time)
                layout["tracking"].update(tracking_table)

                # Refresh the live display
                live.refresh()

                if args.video_window:
                    cv2.imshow('Yoga Feed', frame)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # Cleanup
        cap.release()
        if args.video_window:
            cv2.destroyAllWindows()
        
        # --- Display Final Session Summary ---
        console.print("\n[bold magenta]Session Complete![/bold magenta]")
        final_summary_table = create_tracking_table(pose_history, "N/A", 0) # Don't show current pose
        summary_panel = Panel(final_summary_table, title="Final Session Summary", border_style="green")
        console.print(summary_panel)
        console.print("\n[bold green]Session ended. Goodbye![/bold green]")

if __name__ == "__main__":
    main()
