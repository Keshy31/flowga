import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import time

# Since we are in a different file, we need to import the functions to be tested.
# We assume the file is named yoga_tui.py
from yoga_tui import (
    are_poses_different, POSE_DIFFERENCE_THRESHOLD,
    create_tracking_table, analyze_pose
)

# Helper class to mock MediaPipe landmarks
class MockLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class MockPoseLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class TestPoseDifference(unittest.TestCase):

    def test_no_old_landmarks(self):
        """Test that it returns True when there are no previous landmarks."""
        new_lm = MockPoseLandmarks([MockLandmark(0.1, 0.1, 0.1, 0.9)])
        self.assertTrue(are_poses_different(new_lm, None, POSE_DIFFERENCE_THRESHOLD))

    def test_poses_are_similar(self):
        """Test that it returns False when poses are almost identical."""
        landmarks1 = [MockLandmark(0.5, 0.5, 0.5, 0.95)]
        landmarks2 = [MockLandmark(0.51, 0.51, 0.51, 0.94)] # Very small change
        
        pose1 = MockPoseLandmarks(landmarks1)
        pose2 = MockPoseLandmarks(landmarks2)
        
        self.assertFalse(are_poses_different(pose2, pose1, POSE_DIFFERENCE_THRESHOLD))

    def test_poses_are_different(self):
        """Test that it returns True when poses are significantly different."""
        landmarks1 = [MockLandmark(0.5, 0.5, 0.5, 0.95)]
        landmarks2 = [MockLandmark(0.8, 0.8, 0.8, 0.95)] # Large change

        pose1 = MockPoseLandmarks(landmarks1)
        pose2 = MockPoseLandmarks(landmarks2)

        self.assertTrue(are_poses_different(pose2, pose1, POSE_DIFFERENCE_THRESHOLD))

    def test_visibility_change_matters(self):
        """Test that a significant change in visibility also triggers detection."""
        landmarks1 = [MockLandmark(0.5, 0.5, 0.5, 0.95)]
        landmarks2 = [MockLandmark(0.5, 0.5, 0.5, 0.1)] # Visibility dropped

        pose1 = MockPoseLandmarks(landmarks1)
        pose2 = MockPoseLandmarks(landmarks2)

        self.assertTrue(are_poses_different(pose2, pose1, POSE_DIFFERENCE_THRESHOLD))


class TestTrackingTable(unittest.TestCase):

    def test_initial_empty_table(self):
        """Test that the table is correct when no poses are tracked."""
        table = create_tracking_table({}, "N/A", 0)
        self.assertEqual(len(table.rows), 1)
        self.assertIn("No poses tracked yet", str(table.rows[0]))

    def test_table_with_history(self):
        """Test that the table correctly displays pose history."""
        history = {"Warrior I": 30.5, "Downward Dog": 45.2}
        table = create_tracking_table(history, "N/A", 0)
        self.assertEqual(len(table.rows), 2)
        self.assertIn("Warrior I", str(table.rows[0]))
        self.assertIn("30.5", str(table.rows[0]))

    def test_table_with_current_pose(self):
        """Test that the table correctly displays the current pose."""
        history = {"Warrior I": 30.5}
        start_time = time.time() - 10  # 10 seconds ago
        table = create_tracking_table(history, "Tree Pose", start_time)
        self.assertEqual(len(table.rows), 2)
        self.assertIn("Tree Pose", str(table.rows[1]))
        self.assertIn("(current)", str(table.rows[1]))


class TestAnalysisFunction(unittest.TestCase):

    @patch('yoga_tui.ollama.chat')
    def test_analyze_pose_success(self, mock_ollama_chat):
        """Test successful analysis with a valid JSON response."""
        mock_response = {
            'message': {
                'content': '{"pose": "Warrior II", "feedback": "Good form!", "score": 9}'
            }
        }
        mock_ollama_chat.return_value = mock_response

        result = analyze_pose("test-model", "fake-image", "fake-landmarks")
        self.assertEqual(result['pose'], "Warrior II")
        self.assertEqual(result['score'], 9)

    @patch('yoga_tui.ollama.chat')
    def test_analyze_pose_malformed_json(self, mock_ollama_chat):
        """Test handling of a malformed JSON string."""
        mock_response = {
            'message': {
                'content': '{"pose": "Warrior II", "feedback": "Good form!",,}' # Invalid JSON
            }
        }
        mock_ollama_chat.return_value = mock_response

        result = analyze_pose("test-model", "fake-image", "fake-landmarks")
        self.assertEqual(result['pose'], "Error")
        self.assertIn("Failed to analyze", result['feedback'])

    @patch('yoga_tui.ollama.chat')
    def test_analyze_pose_incomplete_json(self, mock_ollama_chat):
        """Test handling of a JSON object with missing keys."""
        mock_response = {
            'message': {
                'content': '{"pose": "Warrior II", "feedback": "Good form!"}' # Missing 'score'
            }
        }
        mock_ollama_chat.return_value = mock_response

        result = analyze_pose("test-model", "fake-image", "fake-landmarks")
        self.assertEqual(result['pose'], "Error")


if __name__ == '__main__':
    unittest.main()
