import sys
import os
import tkinter as tk
from hand_gesture_trainer import HandGestureTrainer

def check_dependencies():
    """Check for required packages"""
    required_packages = {
        'opencv-python': 'cv2', 
        'mediapipe': 'mediapipe', 
        'numpy': 'numpy',
        'pyautogui': 'pyautogui',
        'pillow': 'PIL'
    }
    
    missing_packages = []
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages. Please install the following packages:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nYou can install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        input("Press Enter to exit...")
        sys.exit(1)

def main():
    # Check for required packages
    check_dependencies()
    
    # Start the application
    root = tk.Tk()
    app = HandGestureTrainer(root)
    root.mainloop()

if __name__ == "__main__":
    main()