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

# Initialize Rich Console
console = Console()

def main():
    """Main function to run the Yoga TUI application."""
    parser = argparse.ArgumentParser(description="AI Powered Yoga Pose Feedback and Tracker")
    parser.add_argument("--video-window", action="store_true", help="Display the webcam feed in a separate GUI window.")
    parser.add_argument("--model", type=str, default="qwen2-vl:7b", help="The Ollama VLM model to use for analysis.")
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

    try:
        while True:
            success, frame = cap.read()
            if not success:
                console.print("[bold yellow]Warning: Could not read frame from webcam.[/bold yellow]")
                break

            # Mirror the frame for a more intuitive display
            frame = cv2.flip(frame, 1)

            # Display frame metadata in terminal (example)
            h, w, _ = frame.shape
            console.print(f"Frame: {w}x{h}", end='\r')

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
