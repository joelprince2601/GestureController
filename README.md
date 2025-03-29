# Hand Gesture Control

## Accessibility Through Motion

A computer interface system that empowers individuals with mobility limitations to control their devices through customized hand gestures, providing an accessible alternative to conventional keyboard and mouse input.

## Overview

This application uses computer vision technology to detect and recognize hand gestures, allowing users to perform various computer actions without physical contact with input devices. Users can train the system to recognize personalized gestures and map them to keyboard commands or mouse controls.

## Features

- **Custom Gesture Training**: Create personalized hand gestures mapped to specific computer actions
- **Accessibility-Focused**: Designed for individuals with limited mobility or dexterity
- **Multiple Control Options**: Map gestures to keyboard presses, mouse clicks, or cursor movements
- **User-Friendly Interface**: Simple interface with clear, intuitive controls
- **Gesture Management**: Save, edit, and organize your gesture library
- **Real-Time Testing**: Test gestures and see immediate feedback

## Requirements

- Python 3.7+
- Webcam
- Libraries:
  - OpenCV
  - MediaPipe
  - NumPy
  - PyAutoGUI
  - Tkinter
  - PIL (Pillow)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hand-gesture-control.git
   cd hand-gesture-control
   ```

2. Install required packages:
   ```
   pip install opencv-python mediapipe numpy pyautogui pillow
   ```

3. Run the application:
   ```
   python app.py
   ```

## How to Use

### Training Mode
1. Start the camera
2. Enter a gesture name (e.g., "SwipeRight")
3. Select action type (keyboard, mouse_click, mouse_move)
4. Enter action value (e.g., "right" for arrow key)
5. Start recording and hold your gesture in front of the camera
6. Save your gesture after collecting samples

### Testing Mode
1. Navigate to the Testing tab
2. Start Testing to see your gestures recognized in real-time
3. When gestures are recognized with high confidence, their actions will execute

### Management Tab
- View all saved gestures
- Delete unwanted gestures
- Import/Export gesture libraries to share or backup

## Use Cases

- Accessibility for individuals with limited hand mobility
- Alternative input for people with repetitive strain injuries
- Touchless computer control in clean environments
- Assistive technology for various physical disabilities

## Contributing

Contributions to improve accessibility features are especially welcome. Please feel free to submit pull requests or open issues to suggest improvements.

## License

[MIT License](LICENSE)
