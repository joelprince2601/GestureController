import time
import pyautogui

class GestureRecognizer:
    def __init__(self):
        # Store the gesture data
        self.gesture_data = {}
        # For preventing rapid-fire actions
        self.last_action_time = 0
        
    def set_gesture_data(self, gesture_data):
        """Set the gesture data to use for recognition"""
        self.gesture_data = gesture_data
    
    def recognize(self, landmarks):
        """Compare current hand landmarks with saved gestures"""
        if not self.gesture_data:
            return None, 0, 0
        
        best_match = None
        best_score = float('inf')
        
        # Compare with each saved gesture
        for name, data in self.gesture_data.items():
            samples = data['samples']
            
            # Calculate minimum distance to any sample
            sample_scores = []
            for sample in samples:
                # Calculate Euclidean distance between landmarks
                total_distance = 0
                for i in range(min(len(landmarks), len(sample))):
                    dist = sum((a - b) ** 2 for a, b in zip(landmarks[i], sample[i]))
                    total_distance += dist
                
                sample_scores.append(total_distance)
            
            # Get average score across samples
            avg_score = sum(sample_scores) / len(sample_scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_match = name
        
        # Determine if the match is good enough
        # This threshold might need tuning based on testing
        threshold = 0.1
        confidence = max(0, min(100, int(100 * (1 - best_score / threshold))))
        
        if best_score < threshold:
            return best_match, best_score, confidence
        else:
            return None, best_score, confidence
    
    def execute_action(self, action_type, action_value):
        """Execute the associated action for a recognized gesture"""
        try:
            # Add a cooldown to prevent rapid-fire actions
            current_time = time.time()
            if current_time - self.last_action_time < 1.0:
                return
            
            self.last_action_time = current_time
            
            if action_type == "keyboard":
                pyautogui.press(action_value)
            elif action_type == "mouse_click":
                pyautogui.click(button=action_value)
            elif action_type == "mouse_move":
                # For mouse_move, action_value should be direction: up, down, left, right
                distance = 20  # pixels to move
                if action_value == "up":
                    pyautogui.moveRel(0, -distance)
                elif action_value == "down":
                    pyautogui.moveRel(0, distance)
                elif action_value == "left":
                    pyautogui.moveRel(-distance, 0)
                elif action_value == "right":
                    pyautogui.moveRel(distance, 0)
        except Exception as e:
            print(f"Error executing action: {str(e)}")