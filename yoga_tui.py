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

if __name__ == "__main__":
    main()
