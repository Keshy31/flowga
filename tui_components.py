import time
from rich.table import Table
from rich import box

from config import LANDMARK_NAMES

def create_pose_table(results):
    """Creates a Rich table to display pose landmarks."""
    table = Table(title="Pose Landmarks", show_header=True, header_style="bold #ff79c6", box=box.ROUNDED)
    table.add_column("ID", style="#6272a4", width=3)
    table.add_column("Landmark", width=20)
    table.add_column("Visibility", justify="right")

    if not results or not results.pose_landmarks:
        table.add_row("-", "No pose detected", "-")
        return table

    for i, landmark in enumerate(results.pose_landmarks.landmark):
        table.add_row(str(i), LANDMARK_NAMES[i], f"{landmark.visibility:.2f}")

    return table

def create_tracking_table(console, pose_history, current_pose=None, current_pose_start_time=0, is_final_summary=False):
    """Creates a Rich table to display pose history and session tracking."""
    table = Table(show_header=True, header_style="bold #8be9fd", box=box.ROUNDED)
    table.add_column("Pose", style="white")
    table.add_column("Duration (s)", justify="right", style="#50fa7b")

    temp_history = pose_history.copy()
    if current_pose and current_pose != "N/A":
        duration = time.time() - current_pose_start_time
        temp_history[current_pose] += duration

    sorted_poses = sorted(temp_history.items(), key=lambda item: item[1], reverse=True)

    if is_final_summary:
        table.title = "Final Session Summary"
        table.add_column("Time (%)", justify="right", style="#f1fa8c")
        poses_to_show = sorted_poses
        total_duration = sum(duration for _, duration in poses_to_show)
        table.caption = f"[bold]Total Poses: {len(poses_to_show)} | Total Duration: {total_duration:.1f}s[/bold]"
    else:
        table.title = "Pose Session Tracking"
        # Show top 7 poses in live view
        poses_to_show = sorted_poses[:7]
        total_duration = sum(duration for _, duration in poses_to_show)

    for pose, duration in poses_to_show:
        pose_name = pose
        if not is_final_summary and pose == current_pose:
            pose_name = f"[bold][info]{pose} (current)[/info][/bold]"
        
        row = [pose_name, f"{duration:.1f}"]
        if is_final_summary and total_duration > 0:
            percentage = (duration / total_duration) * 100
            row.append(f"{percentage:.1f}%")
        
        table.add_row(*row)

    return table
