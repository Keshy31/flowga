import numpy as np
import ollama
import json
import re
import base64
import cv2
import threading
import queue
import os
import csv
from pathlib import Path
import mediapipe as mp
import time

# --- Helper Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., for a joint) using the dot product."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    # Calculate angle in radians and convert to degrees
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_pose(model_name, base64_image, landmarks_str, joint_angles, console):
    """Sends pose data to the AI model and returns JSON feedback."""
    try:
        prompt_text = (
            "Analyze the yoga pose in the image. "
            "The user is performing a yoga pose, and your task is to identify the pose, provide corrective feedback, and score it. "
            "Use the provided joint angles and landmark data to enhance your analysis. For example, in Warrior II, the front knee should be near 90 degrees. "
            f"Here are the calculated joint angles: {json.dumps(joint_angles) if joint_angles else 'Not available'}. "
            "Return a single JSON object with three keys: 'pose' (string), 'feedback' (string), and 'score' (float from 0.0 to 10.0)."
        )

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [base64_image]
                }
            ],
            options={"temperature": 0.3}
        )
        content = response['message']['content']

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            feedback_json = json.loads(json_str)
            return feedback_json
        else:
            console.print(f"[bold error]Could not find JSON in model response: {content}[/bold error]")
            return None

    except json.JSONDecodeError:
        console.print(f"[bold error]Failed to decode JSON from response: {json_str}[/bold error]")
        return None
    except Exception as e:
        console.print(f"[bold error]An unexpected error occurred during analysis: {e}[/bold error]")
        return None

# --- Worker Threads ---

def analysis_worker(frame, landmarks, args, console, tts_queue, log_queue, latest_feedback):
    """Analyzes a single frame for pose, generates feedback, and queues it."""
    joint_angles = {}
    if landmarks:
        lm = landmarks.landmark
        mp_pose = mp.solutions.pose
        try:
            left_knee_angle = calculate_angle([lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y], [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y], [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y])
            right_knee_angle = calculate_angle([lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y], [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y], [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y])
            left_elbow_angle = calculate_angle([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y], [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y], [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y])
            right_elbow_angle = calculate_angle([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y], [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y], [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
            joint_angles = {
                'left_knee': round(left_knee_angle, 2),
                'right_knee': round(right_knee_angle, 2),
                'left_elbow': round(left_elbow_angle, 2),
                'right_elbow': round(right_elbow_angle, 2)
            }
        except Exception as e:
            pass  # Ignore if landmarks are not visible

    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    landmarks_str = str(landmarks) if landmarks else ""

    feedback = analyze_pose(args.model, base64_image, landmarks_str, joint_angles, console)

    if feedback and isinstance(feedback, dict):
        with latest_feedback['lock']:
            latest_feedback['data'] = feedback
        
        try:
            tts_queue.put_nowait(feedback.get('feedback', ''))
        except queue.Full:
            pass  # Ignore if queue is full, new feedback is coming

        if args.log:
            try:
                log_queue.put_nowait((time.time(), feedback, landmarks, frame))
            except queue.Full:
                pass

def log_session_worker(log_queue, console):
    """Logs session data (CSV and images) from a queue."""
    log_dir = Path("yoga_logs")
    log_dir.mkdir(exist_ok=True)
    
    session_time = time.strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"session_{session_time}.csv"
    img_dir = log_dir / f"session_{session_time}_images"
    img_dir.mkdir(exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "pose", "feedback", "score", "image_path"])

        while True:
            try:
                timestamp, feedback, landmarks, frame = log_queue.get(timeout=1)
                if feedback is None:  # Sentinel value to stop
                    break
                
                img_path = img_dir / f"{timestamp:.0f}_{feedback.get('pose', 'unknown')}.jpg"
                
                # Draw landmarks on the image before saving
                if landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, 
                        landmarks, 
                        mp.solutions.pose.POSE_CONNECTIONS
                    )
                cv2.imwrite(str(img_path), frame)
                
                writer.writerow([
                    timestamp, 
                    feedback.get('pose', 'N/A'), 
                    feedback.get('feedback', ''), 
                    feedback.get('score', 0.0),
                    str(img_path)
                ])
            except queue.Empty:
                continue
            except Exception as e:
                console.print(f"[bold error]Error in logging worker: {e}[/bold error]")
