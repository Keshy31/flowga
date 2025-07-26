import cv2
import mediapipe as mp
import ollama
import pyttsx3
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich import box
import time
import os
import base64
from collections import defaultdict
import argparse
import threading
import json
import queue
from pathlib import Path
import csv
import re
import numpy as np
from config import dracula_theme, LANDMARK_NAMES
from tui_components import create_pose_table, create_tracking_table
from analysis import analysis_worker, log_session_worker

# Initialize Rich Console with the Dracula theme
console = Console(theme=dracula_theme)

# --- Suppress TensorFlow/Mediapipe logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Shared TTS engine and a lock for thread-safe operations.
# A single engine is used to allow for speech interruption.
engine = pyttsx3.init()
engine_lock = threading.Lock()

def stop_current_speech():
    """Stops any currently playing speech. Thread-safe."""
    try:
        with engine_lock:
            # This call interrupts the engine's current speech.
            engine.stop()
    except Exception as e:
        console.print(f"[bold error]Error stopping speech: {e}[/bold error]")

def main():
    """Main function to run the Yoga TUI application."""
    global args, console, engine, tts_queue, latest_feedback, feedback_lock, log_queue
    parser = argparse.ArgumentParser(description="Flowga - AI Yoga Pose Assistant")
    parser.add_argument("--model", type=str, default="qwen2.5vl:7b", help="Ollama model to use for analysis")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--video_window", action="store_true", help="Show the OpenCV video window")
    parser.add_argument("--analysis_interval", type=float, default=5.0, help="Seconds between pose analyses")
    parser.add_argument("--log", action="store_true", help="Enable analytics logging to CSV and images")
    args = parser.parse_args()

    ascii_art = r"""
[bold][method]
 ███████╗██╗      ██████╗ ██╗    ██╗ ██████╗  █████╗ 
 ██╔════╝██║     ██╔═══██╗██║    ██║██╔════╝ ██╔══██╗
 █████╗  ██║     ██║   ██║██║ █╗ ██║██║  ███╗███████║
 ██╔══╝  ██║     ██║   ██║██║███╗██║██║   ██║██╔══██║
 ██║     ███████╗╚██████╔╝╚███╔███╔╝╚██████╔╝██║  ██║
 ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝
[/method][/bold]
    """
    console.print(ascii_art)
    console.print(Text("Your Personal AI Yoga Instructor", justify="center", style="italic info"))
    console.print(f"\nUsing model: [bold info]{args.model}[/bold info]")
    if args.video_window:
        console.print("Video window display is [bold success]enabled[/bold success].")
    else:
        console.print("Video window display is [bold error]disabled[/bold error].")
    
    if args.log:
        console.print("Analytics logging is [bold success]enabled[/bold success].")

    # --- TTS Event Handler ---
    def on_finish(name, completed):
        # This callback runs in the main thread after speech finishes.
        # `completed` is True if the speech finished, False if interrupted.
        if completed:
            console.print(f"[comment]Finished speaking.[/comment]")

    engine.connect('finished-utterance', on_finish)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils # Restore drawing utility

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        console.print("[bold error]Error: Could not open camera.[/bold error]")
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
    log_queue = None

    # --- Setup Logging if enabled ---
    if args.log:
        log_queue = queue.Queue()
        log_thread = threading.Thread(
            target=log_session_worker, 
            args=(log_queue, console),
            daemon=True
        )
        log_thread.start()

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
        with Live(layout, refresh_per_second=60, screen=True, console=console) as live:
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
                    console.print("[bold error]Error: Could not read frame.[/bold error]")
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
                    if current_time - last_analysis_time >= args.analysis_interval:
                        last_analysis_time = current_time

                        # This is a non-blocking call that passes copies to the thread
                        analysis_thread = threading.Thread(
                            target=analysis_worker, 
                            args=(frame.copy(), results.pose_landmarks, args, console, tts_queue, log_queue, latest_feedback), 
                            daemon=True
                        )
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
                        border_style = "#50fa7b"
                    elif score >= 5:
                        border_style = "#f1fa8c"
                    else:
                        border_style = "#ff5555"

                    feedback_text = latest_feedback.get('feedback', 'Waiting for analysis...')
                    pose_name = latest_feedback.get('pose', 'N/A')
                    feedback_panel = Panel(
                        Text(f"Pose: {pose_name}\nFeedback: {feedback_text}\nScore: {latest_feedback.get('score', 0)}/10", style="white"),
                        title="AI Feedback",
                        border_style=border_style
                    )
                    layout["feedback"].update(feedback_panel)

                    # Update the tracking table
                    tracking_table = create_tracking_table(console, pose_history, current_pose, current_pose_start_time)
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

        # Stop logging thread
        if args.log and log_queue:
            log_queue.put((None, None, None, None))  # Sentinel to stop logger
            log_thread.join() # Wait for logger to finish

        if args.video_window:
            cv2.destroyAllWindows()

        # Stop any lingering TTS engine processes and end the loop on exit.
        try:
            engine.endLoop()
        except Exception:
            pass
        
        # --- Log final pose duration before creating the summary ---
        if current_pose != "N/A":
            duration = time.time() - current_pose_start_time
            pose_history[current_pose] += duration

        # --- Display Final Session Summary ---
        console.print("\n[bold][keyword]Session Complete![/keyword][/bold]")
        final_summary_table = create_tracking_table(console, pose_history, is_final_summary=True)
        console.print(Panel(final_summary_table, title="Pose Session Summary", border_style="keyword"))
        console.print("\n[bold]Session ended. Goodbye![/bold]")

if __name__ == "__main__":
    main()