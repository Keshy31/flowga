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
import queue

# Initialize Rich Console
console = Console()

# Shared TTS engine and a lock for thread-safe operations.
# A single engine is used to allow for speech interruption.
engine = pyttsx3.init()
engine_lock = threading.Lock()

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
    table.add_column("Landmark", width=20)
    table.add_column("Visibility", justify="right")

    if not results or not results.pose_landmarks:
        table.add_row("-", "No pose detected", "-")
        return table

    for i, landmark in enumerate(results.pose_landmarks.landmark):
        table.add_row(str(i), landmark_names[i], f"{landmark.visibility:.2f}")

    return table

def create_tracking_table(pose_history, current_pose=None, current_pose_start_time=0, is_final_summary=False):
    """Creates a Rich table to display pose history and session tracking."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Pose", style="white")
    table.add_column("Duration (s)", justify="right", style="green")

    temp_history = pose_history.copy()
    if current_pose and current_pose != "N/A":
        duration = time.time() - current_pose_start_time
        temp_history[current_pose] += duration

    sorted_poses = sorted(temp_history.items(), key=lambda item: item[1], reverse=True)

    if is_final_summary:
        table.title = "Final Session Summary"
        table.add_column("Time (%)", justify="right", style="yellow")
        poses_to_show = sorted_poses
        total_duration = sum(duration for _, duration in poses_to_show)
        table.show_footer = True
        table.footer = f"[bold]Total Poses: {len(poses_to_show)} | Total Duration: {total_duration:.1f}s[/bold]"
    else:
        table.title = "Pose Session Tracking"
        # Show top 7 poses in live view
        poses_to_show = sorted_poses[:7]
        total_duration = sum(duration for _, duration in poses_to_show)

    for pose, duration in poses_to_show:
        pose_name = pose
        if not is_final_summary and pose == current_pose:
            pose_name = f"[bold cyan]{pose} (current)[/bold cyan]"
        
        row = [pose_name, f"{duration:.1f}"]
        if is_final_summary and total_duration > 0:
            percentage = (duration / total_duration) * 100
            row.append(f"{percentage:.1f}%")
        
        table.add_row(*row)

    return table

def analyze_pose(model_name, base64_image, landmarks_str):
    """Analyzes the pose using Ollama VLM and returns structured feedback."""
    try:
        prompt = (
            "You are a yoga instructor AI. Analyze the provided image and the keypoints of the person's pose. "
            "Identify the yoga pose. Provide specific, constructive feedback on their alignment in **one single, concise sentence** suitable for audio playback. "
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

def stop_current_speech():
    """Stops any currently playing speech. Thread-safe."""
    try:
        with engine_lock:
            # This call interrupts the engine's current speech.
            engine.stop()
    except Exception as e:
        console.print(f"[bold red]Error stopping speech: {e}[/bold red]")

def main():
    """Main function to run the Yoga TUI application."""
    parser = argparse.ArgumentParser(description="Flowga - AI Yoga Pose Assistant")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama model to use for pose analysis")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--video_window", action="store_true", help="Show the OpenCV video window")
    parser.add_argument("--analysis_interval", type=float, default=5.0, help="Seconds between pose analyses")
    args = parser.parse_args()

    ascii_art = r"""
[bold deep_sky_blue1]
███████╗██╗      ██████╗ ██╗    ██╗ ██████╗  █████╗ 
██╔════╝██║     ██╔═══██╗██║    ██║██╔════╝ ██╔══██╗
█████╗  ██║     ██║   ██║██║ █╗ ██║██║  ███╗███████║
██╔══╝  ██║     ██║   ██║██║███╗██║██║   ██║██╔══██║
██║     ███████╗╚██████╔╝╚███╔███╔╝╚██████╔╝██║  ██║
╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝[/bold deep_sky_blue1]
    """
    console.print(ascii_art)
    console.print(Text("Your Personal AI Yoga Instructor", justify="center", style="italic bright_blue"))
    console.print(f"\nUsing model: [bold cyan]{args.model}[/bold cyan]")
    if args.video_window:
        console.print("Video window display is [bold green]enabled[/bold green].")
    else:
        console.print("Video window display is [bold red]disabled[/bold red].")

    # --- TTS Event Handler ---
    def on_finish(name, completed):
        # This callback runs in the main thread after speech finishes.
        # `completed` is True if the speech finished, False if interrupted.
        if completed:
            console.print(f"[dim]Finished speaking.[/dim]")

    engine.connect('finished-utterance', on_finish)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils # Restore drawing utility

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        console.print("[bold red]Error: Could not open camera.[/bold red]")
        return

    # --- Initialize TTS queue ---
    # The queue has a maxsize of 1 to ensure only the latest feedback is spoken.
    # It will now be processed directly in the main loop for thread safety.
    tts_queue = queue.Queue(maxsize=1)

    # --- Shared state for analysis results ---
    latest_feedback = {
        "pose": "N/A",
        "feedback": "Position yourself in frame to start.",
        "score": 0
    }
    feedback_lock = threading.Lock()

    # --- Pose tracking variables ---
    pose_history = defaultdict(float)
    current_pose = "N/A"
    current_pose_start_time = time.time()
    last_analysis_time = 0
    analysis_thread = None

    # --- Rich Layout Setup ---
    layout = Layout()
    layout.split_row(
        Layout(name="left_panel", ratio=2),
        Layout(name="right_panel", ratio=1)
    )
    layout["left_panel"].split_column(
        Layout(name="feedback", size=8),
        Layout(name="tracking")
    )

    try:
        # Start the TTS engine's non-blocking event loop
        engine.startLoop(False)
        with Live(layout, refresh_per_second=60, screen=True) as live:
            while True:
                # Use a non-blocking call to the TTS engine's event loop
                engine.iterate()

                # --- Process TTS Queue in Main Thread ---
                if not tts_queue.empty():
                    try:
                        text = tts_queue.get_nowait()
                        with engine_lock:
                            engine.say(text)
                    except queue.Empty:
                        pass # Ignore if queue was emptied by another part of the loop

                ret, frame = cap.read()
                if not ret:
                    console.print("[bold red]Error: Could not read frame.[/bold red]")
                    break

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                # --- Draw landmarks on the frame ---
                if results.pose_landmarks and args.video_window:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )

                # --- Pose Analysis Trigger ---
                current_time = time.time()
                if results.pose_landmarks:
                    if current_time - last_analysis_time > args.analysis_interval and (analysis_thread is None or not analysis_thread.is_alive()):
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
                                with feedback_lock:
                                    latest_feedback.update(feedback)
                                
                                # --- Update TTS Queue with Latest Feedback ---
                                # If the queue is full (i.e., contains old feedback), clear it.
                                if tts_queue.full():
                                    try:
                                        tts_queue.get_nowait() # Non-blocking get
                                    except queue.Empty:
                                        pass # Ignore if already empty
                                # Put the new, most relevant feedback into the queue.
                                tts_queue.put(feedback['feedback'])

                        # Start the analysis in a new thread
                        analysis_thread = threading.Thread(target=analysis_task)
                        analysis_thread.start()

                # --- Check for pose change and update history ---
                with feedback_lock:
                    if latest_feedback["pose"] != "N/A" and latest_feedback["pose"] != current_pose:
                        # Log duration for the previous pose
                        if current_pose != "N/A":
                            duration = time.time() - current_pose_start_time
                            pose_history[current_pose] += duration
                        
                        # Update to the new pose and interrupt any ongoing speech.
                        current_pose = latest_feedback["pose"]
                        current_pose_start_time = time.time()
                        stop_current_speech()

                # --- Update Dashboard ---
                # Update the landmark table
                pose_table = create_pose_table(results)
                layout["right_panel"].update(pose_table)

                # Update the feedback panel with dynamic colors
                with feedback_lock:
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
        # --- Signal TTS worker to exit and cleanup ---
        # Cleanup
        cap.release()
        if args.video_window:
            cv2.destroyAllWindows()
        
        # Stop any lingering TTS engine processes and end the loop on exit.
        try:
            engine.endLoop()
        except Exception:
            pass
        
        # --- Display Final Session Summary ---
        console.print("\n[bold magenta]Session Complete![/bold magenta]")
        final_summary_table = create_tracking_table(pose_history, is_final_summary=True)
        console.print(Panel(final_summary_table, title="Pose Session Summary", border_style="magenta"))
        console.print("\n[bold]Session ended. Goodbye![/bold]")

if __name__ == "__main__":
    main()
