import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from threading import Thread
import time
from gesture_recognizer import GestureRecognizer

class HandGestureTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Trainer")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = None
        self.camera_active = False
        self.thread = None
        self.stop_thread = False
        
        # Training data
        self.gesture_data = {}
        self.current_samples = []
        self.is_recording = False
        self.current_gesture_name = ""
        self.current_action_type = "keyboard"
        self.current_action_value = ""
        self.sample_count = 0
        self.required_samples = 30
        
        # Create recognizer object
        self.recognizer = GestureRecognizer()
        
        # Load existing gestures if available
        self.load_gestures()
        
        # Create UI
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame setup with tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Tab 1: Gesture Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        
        # Tab 2: Gesture Testing
        self.testing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.testing_tab, text="Testing")
        
        # Tab 3: Gesture Management
        self.management_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.management_tab, text="Management")
        
        self.notebook.pack(expand=1, fill="both")
        
        # Setup each tab
        self.setup_training_tab()
        self.setup_testing_tab()
        self.setup_management_tab()
    
    def setup_training_tab(self):
        main_frame = ttk.Frame(self.training_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Split into left (video) and right (controls) sections
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="y", padx=10)
        
        # Video display
        self.video_label = ttk.Label(left_frame)
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(right_frame, text="Training Controls")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Gesture name input
        ttk.Label(control_frame, text="Gesture Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.gesture_name_entry = ttk.Entry(control_frame, width=20)
        self.gesture_name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Action type selection
        ttk.Label(control_frame, text="Action Type:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.action_type_combo = ttk.Combobox(control_frame, values=["keyboard", "mouse_click", "mouse_move"], state="readonly")
        self.action_type_combo.current(0)
        self.action_type_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Action value input
        ttk.Label(control_frame, text="Action Value:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.action_value_entry = ttk.Entry(control_frame, width=20)
        self.action_value_entry.grid(row=2, column=1, padx=5, pady=5)
        self.action_value_info = ttk.Label(control_frame, text="Keyboard: key name (e.g., 'space', 'a')\nMouse: button name (e.g., 'left', 'right')")
        self.action_value_info.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Sample count
        ttk.Label(control_frame, text="Required Samples:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.sample_count_var = tk.StringVar(value=f"0/{self.required_samples}")
        ttk.Label(control_frame, textvariable=self.sample_count_var).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Camera control button
        self.camera_button = ttk.Button(buttons_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(side="left", padx=5)
        
        # Record button
        self.record_button = ttk.Button(buttons_frame, text="Start Recording", command=self.toggle_recording, state="disabled")
        self.record_button.pack(side="left", padx=5)
        
        # Save button
        self.save_button = ttk.Button(buttons_frame, text="Save Gesture", command=self.save_gesture, state="disabled")
        self.save_button.pack(side="left", padx=5)
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "italic")).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=10)
    
    def setup_testing_tab(self):
        main_frame = ttk.Frame(self.testing_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Split into video and info sections
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="y", padx=10)
        
        # Video display for testing
        self.test_video_label = ttk.Label(left_frame)
        self.test_video_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Testing controls
        control_frame = ttk.LabelFrame(right_frame, text="Testing Controls")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Toggle testing
        self.testing_active = False
        self.test_button = ttk.Button(control_frame, text="Start Testing", command=self.toggle_testing)
        self.test_button.pack(pady=10)
        
        # Recognition info
        self.recognition_frame = ttk.LabelFrame(right_frame, text="Recognition Results")
        self.recognition_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Label(self.recognition_frame, text="Detected Gesture:").pack(anchor="w", padx=5, pady=2)
        self.detected_gesture_var = tk.StringVar(value="None")
        ttk.Label(self.recognition_frame, textvariable=self.detected_gesture_var, font=("Arial", 12, "bold")).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(self.recognition_frame, text="Associated Action:").pack(anchor="w", padx=5, pady=2)
        self.detected_action_var = tk.StringVar(value="None")
        ttk.Label(self.recognition_frame, textvariable=self.detected_action_var, font=("Arial", 12)).pack(anchor="w", padx=5, pady=2)
        
        # Confidence indicator
        ttk.Label(self.recognition_frame, text="Confidence:").pack(anchor="w", padx=5, pady=2)
        self.confidence_var = tk.StringVar(value="0%")
        ttk.Label(self.recognition_frame, textvariable=self.confidence_var).pack(anchor="w", padx=5, pady=2)
    
    def setup_management_tab(self):
        main_frame = ttk.Frame(self.management_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Gestures list
        list_frame = ttk.LabelFrame(main_frame, text="Saved Gestures")
        list_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for gesture list
        columns = ("name", "action_type", "action_value", "samples")
        self.gesture_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        # Define headings
        self.gesture_tree.heading("name", text="Gesture Name")
        self.gesture_tree.heading("action_type", text="Action Type")
        self.gesture_tree.heading("action_value", text="Action Value")
        self.gesture_tree.heading("samples", text="Samples")
        
        # Define columns
        self.gesture_tree.column("name", width=150)
        self.gesture_tree.column("action_type", width=100)
        self.gesture_tree.column("action_value", width=100)
        self.gesture_tree.column("samples", width=70)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.gesture_tree.yview)
        self.gesture_tree.configure(yscrollcommand=scrollbar.set)
        
        self.gesture_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Management buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side="right", fill="y", padx=10)
        
        ttk.Button(button_frame, text="Delete Selected", command=self.delete_gesture).pack(pady=5)
        ttk.Button(button_frame, text="Reload Gestures", command=self.reload_gestures).pack(pady=5)
        ttk.Button(button_frame, text="Export Gestures", command=self.export_gestures).pack(pady=5)
        ttk.Button(button_frame, text="Import Gestures", command=self.import_gestures).pack(pady=5)
        
        # Populate the list
        self.update_gesture_list()
    
    def toggle_camera(self):
        if not self.camera_active:
            # Start camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_active = True
            self.camera_button.config(text="Stop Camera")
            self.record_button.config(state="normal")
            
            # Start video processing thread
            self.stop_thread = False
            self.thread = Thread(target=self.process_video)
            self.thread.daemon = True
            self.thread.start()
        else:
            # Stop camera
            self.stop_thread = True
            if self.thread:
                self.thread.join(timeout=1.0)
            
            if self.cap:
                self.cap.release()
            
            self.camera_active = False
            self.camera_button.config(text="Start Camera")
            self.record_button.config(state="disabled")
            self.save_button.config(state="disabled")
            
            # Reset the video display
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            self.update_video(blank, self.video_label)
            self.update_video(blank, self.test_video_label)
    
    def process_video(self):
        while not self.stop_thread and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract hand landmarks for training/testing
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    # For recording mode
                    if self.is_recording and len(self.current_samples) < self.required_samples:
                        self.current_samples.append(landmarks)
                        self.sample_count = len(self.current_samples)
                        self.sample_count_var.set(f"{self.sample_count}/{self.required_samples}")
                        
                        if self.sample_count >= self.required_samples:
                            self.is_recording = False
                            self.record_button.config(text="Start Recording")
                            self.save_button.config(state="normal")
                            self.status_var.set("Samples collected! Ready to save.")
                    
                    # For testing mode
                    if self.testing_active:
                        self.recognize_gesture(landmarks, frame)
            
            # Display status text
            if self.is_recording:
                cv2.putText(frame, f"Recording: {self.sample_count}/{self.required_samples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the resulting frame
            if self.notebook.index(self.notebook.select()) == 0:  # Training tab
                self.update_video(frame, self.video_label)
            elif self.notebook.index(self.notebook.select()) == 1:  # Testing tab
                self.update_video(frame, self.test_video_label)
            
            time.sleep(0.01)  # Small delay to reduce CPU usage
    
    def update_video(self, frame, label):
        # Convert the frame to a format compatible with tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        
        # Update the label
        label.configure(image=img)
        label.image = img  # Keep a reference to prevent garbage collection
    
    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            self.current_gesture_name = self.gesture_name_entry.get().strip()
            self.current_action_type = self.action_type_combo.get()
            self.current_action_value = self.action_value_entry.get().strip()
            
            # Validate inputs
            if not self.current_gesture_name:
                messagebox.showerror("Error", "Please enter a gesture name")
                return
            
            if not self.current_action_value:
                messagebox.showerror("Error", "Please enter an action value")
                return
            
            if self.current_gesture_name in self.gesture_data:
                response = messagebox.askyesno("Warning", 
                                              f"Gesture '{self.current_gesture_name}' already exists. Overwrite?")
                if not response:
                    return
            
            # Reset samples and start recording
            self.current_samples = []
            self.sample_count = 0
            self.sample_count_var.set(f"{self.sample_count}/{self.required_samples}")
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.save_button.config(state="disabled")
            self.status_var.set("Recording... Hold your gesture steady.")
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            
            if self.sample_count > 0:
                self.save_button.config(state="normal")
                self.status_var.set(f"Recording stopped. {self.sample_count} samples collected.")
            else:
                self.status_var.set("Recording stopped. No samples collected.")
    
    def save_gesture(self):
        if len(self.current_samples) == 0:
            messagebox.showerror("Error", "No samples recorded")
            return
        
        # Save the gesture data
        self.gesture_data[self.current_gesture_name] = {
            'samples': self.current_samples,
            'action_type': self.current_action_type,
            'action_value': self.current_action_value
        }
        
        # Save to file
        try:
            if not os.path.exists('gestures'):
                os.makedirs('gestures')
            
            with open('gestures/gestures.pkl', 'wb') as f:
                pickle.dump(self.gesture_data, f)
            
            self.status_var.set(f"Gesture '{self.current_gesture_name}' saved successfully!")
            self.save_button.config(state="disabled")
            
            # Reset for new recording
            self.current_samples = []
            self.sample_count = 0
            self.sample_count_var.set(f"{self.sample_count}/{self.required_samples}")
            
            # Update gesture list
            self.update_gesture_list()
            
            # Update recognizer
            self.recognizer.set_gesture_data(self.gesture_data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save gesture: {str(e)}")
    
    def load_gestures(self):
        try:
            if os.path.exists('gestures/gestures.pkl'):
                with open('gestures/gestures.pkl', 'rb') as f:
                    self.gesture_data = pickle.load(f)
                    
                # Update recognizer
                self.recognizer.set_gesture_data(self.gesture_data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load gestures: {str(e)}")
            self.gesture_data = {}
    
    def update_gesture_list(self):
        # Clear existing items
        for item in self.gesture_tree.get_children():
            self.gesture_tree.delete(item)
        
        # Add gestures to the list
        for name, data in self.gesture_data.items():
            self.gesture_tree.insert('', 'end', values=(
                name, 
                data['action_type'], 
                data['action_value'], 
                len(data['samples'])
            ))
    
    def delete_gesture(self):
        selected = self.gesture_tree.selection()
        if not selected:
            messagebox.showerror("Error", "No gesture selected")
            return
        
        # Get the selected gesture name
        gesture_name = self.gesture_tree.item(selected[0])['values'][0]
        
        # Confirm deletion
        if messagebox.askyesno("Confirm", f"Delete gesture '{gesture_name}'?"):
            if gesture_name in self.gesture_data:
                del self.gesture_data[gesture_name]
                
                # Save changes
                try:
                    with open('gestures/gestures.pkl', 'wb') as f:
                        pickle.dump(self.gesture_data, f)
                    
                    # Update the list
                    self.update_gesture_list()
                    
                    # Update recognizer
                    self.recognizer.set_gesture_data(self.gesture_data)
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save changes: {str(e)}")
    
    def reload_gestures(self):
        self.load_gestures()
        self.update_gesture_list()
        messagebox.showinfo("Info", "Gestures reloaded")
    
    def export_gestures(self):
        try:
            if not self.gesture_data:
                messagebox.showinfo("Info", "No gestures to export")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'wb') as f:
                    pickle.dump(self.gesture_data, f)
                messagebox.showinfo("Success", f"Gestures exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def import_gestures(self):
        try:
            filename = filedialog.askopenfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'rb') as f:
                    imported_data = pickle.load(f)
                
                # Merge with existing gestures
                if messagebox.askyesno("Confirm", "Replace existing gestures with the same name?"):
                    self.gesture_data.update(imported_data)
                else:
                    # Only add gestures that don't exist
                    for name, data in imported_data.items():
                        if name not in self.gesture_data:
                            self.gesture_data[name] = data
                
                # Save changes
                with open('gestures/gestures.pkl', 'wb') as f:
                    pickle.dump(self.gesture_data, f)
                
                # Update recognizer
                self.recognizer.set_gesture_data(self.gesture_data)
                
                self.update_gesture_list()
                messagebox.showinfo("Success", f"Gestures imported from {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {str(e)}")
    
    def toggle_testing(self):
        if not self.testing_active and self.camera_active:
            # Check if we have gestures to test
            if not self.gesture_data:
                messagebox.showerror("Error", "No gestures available for testing")
                return
            
            self.testing_active = True
            self.test_button.config(text="Stop Testing")
            self.detected_gesture_var.set("Waiting...")
            self.detected_action_var.set("None")
        else:
            self.testing_active = False
            self.test_button.config(text="Start Testing")
            self.detected_gesture_var.set("None")
            self.detected_action_var.set("None")
            self.confidence_var.set("0%")
    
    def recognize_gesture(self, landmarks, frame):
        """Use the recognizer to identify the gesture"""
        if not self.gesture_data:
            return
        
        best_match, best_score, confidence = self.recognizer.recognize(landmarks)
        
        if best_match:
            # Update UI
            self.detected_gesture_var.set(best_match)
            
            action_type = self.gesture_data[best_match]['action_type']
            action_value = self.gesture_data[best_match]['action_value']
            self.detected_action_var.set(f"{action_type}: {action_value}")
            self.confidence_var.set(f"{confidence}%")
            
            # Display on frame
            cv2.putText(frame, f"Gesture: {best_match}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Execute the action if confidence is high enough
            if confidence > 70:
                self.recognizer.execute_action(action_type, action_value)
        else:
            self.detected_gesture_var.set("Unknown")
            self.detected_action_var.set("None")
            self.confidence_var.set(f"{confidence}%")
            
            cv2.putText(frame, "Unknown gesture", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_thread = True
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.root.destroy()