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
import threading
import json

# Initialize Rich Console
console = Console()

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
        while True:
            success, frame = cap.read()
            if not success:
                console.print("[bold yellow]Warning: Could not read frame from webcam.[/bold yellow]")
                break

            # Mirror the frame for a more intuitive display
            frame = cv2.flip(frame, 1)

            # Process frame with MediaPipe
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw landmarks on the frame
            if results.pose_landmarks:
                if args.video_window:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                current_time = time.time()
                if current_time - last_analysis_time > analysis_interval:
                    last_analysis_time = current_time

                    # Encode frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_image = base64.b64encode(buffer).decode('utf-8')

                    # Get landmarks as a string
                    landmark_list = []
                    for landmark in results.pose_landmarks.landmark:
                        landmark_list.append(f"({landmark.x:.2f}, {landmark.y:.2f})")
                    landmarks_str = ", ".join(landmark_list)

                    # Get AI feedback
                    feedback = analyze_pose(args.model, base64_image, landmarks_str)
                    if feedback:
                        feedback_panel = Panel(
                            Text(json.dumps(feedback, indent=2), style="white"),
                            title="AI Feedback",
                            border_style="cyan"
                        )
                        console.print(feedback_panel)

                console.print("Pose detected!", end='\r')
            else:
                console.print("No pose detected", end='\r')

            # Display frame metadata in terminal (example)
            h, w, _ = frame.shape
            # console.print(f"Frame: {w}x{h}", end='\r') # Replaced by pose status

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
        console.print("\n[bold green]Session ended. Goodbye![/bold green]")

if __name__ == "__main__":
    main()
