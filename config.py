from rich.theme import Theme

# Dracula theme for the Rich Console
dracula_theme = Theme({
    "info": "#8be9fd",      # Cyan
    "warning": "#f1fa8c",   # Yellow
    "error": "#ff5555",      # Red
    "success": "#50fa7b",   # Green
    "comment": "#6272a4",   # Gray
    "keyword": "#ff79c6",   # Pink
    "string": "#f1fa8c",   # Yellow
    "number": "#bd93f9",   # Purple
    "class": "#50fa7b",      # Green
    "method": "#8be9fd",   # Cyan
    "panel_border": "#6272a4", # Gray
    "progress_bar": "#bd93f9", # Purple
})

# Canonical landmark names from MediaPipe
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]
