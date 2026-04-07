"""Open your Webcam.

Use MediaPipe to get 3D landmarks.

Normalize them (View-Invariance).

Pass a 30-frame window into your ST-GCN.

Display the Mudra Name on the screen."""

"""
#test_1
import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import MudraClassifier

# 1. Load Model and Constants
# AUTOMATIC WAY: This looks at your folders so the count is always correct
dataset_path = "Mudras(Symbolic Hand Poses)dataset"
class_names = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

print(f"Loading model for {len(class_names)} mudras: {class_names}")

# Now num_classes will be 5, matching your .pth file!
model = MudraClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('mudra_stgcn.pth'))
model.eval()

# 2. Setup MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_window = []

print("\n--- Live Deep Learning Inference Started ---")
print("Show your hand to the camera. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Flip the frame for a mirror effect (more natural for the user)
    frame = cv2.flip(frame, 1)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        # Take the first hand detected
        hand_lms = result.hand_landmarks[0]
        current_frame_coords = []
        for lm in hand_lms:
            current_frame_coords.extend([lm.x, lm.y, lm.z])
        
        frame_window.append(current_frame_coords)
        if len(frame_window) > 30:
            frame_window.pop(0)
            
        if len(frame_window) == 30:
            # 3. Pre-process (Spatio-Temporal Normalization)
            data = np.array(frame_window).reshape(30, 21, 3)
            # Translation invariance (subtract wrist)
            data = data - data[:, 0, :].reshape(30, 1, 3)
            # Scale invariance
            dist = np.linalg.norm(data[0, 0] - data[0, 9])
            if dist > 0: data = data / dist
            
            # 4. Deep Learning Prediction
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                # Calculate confidence (Softmax)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                mudra_label = class_names[predicted.item()]
                conf_score = confidence.item() * 100
                
            # Display Prediction + Confidence
            color = (0, 255, 0) if conf_score > 70 else (0, 165, 255) # Green if sure, Orange if not
            cv2.putText(frame, f"{mudra_label} ({conf_score:.1f}%)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('ST-GCN Mudra Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

# accuracy achieved is upto 92% , 
"""but failed to capture the mudras properly , was not able to lcassify becos of diffeent lighting and background """



#test_2
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from model import MudraClassifier

# --- 1. CONFIG & MODEL ---
dataset_path = "Mudras(Symbolic Hand Poses)dataset"
class_names = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

model = MudraClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('mudra_stgcn.pth'))
model.eval()

# --- 2. MEDIAPIPE TASKS SETUP ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. LIVE LOOP ---
cap = cv2.VideoCapture(0)
frame_window = []

print(f"ST-GCN Active. Detecting: {class_names}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        # Get the first hand
        hand_lms = result.hand_landmarks[0]
        
        # Draw Dots for visual feedback
        current_frame = []
        for lm in hand_lms:
            # Draw on screen
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            # Save coordinates
            current_frame.extend([lm.x, lm.y, lm.z])
        
        frame_window.append(current_frame)
        if len(frame_window) > 30:
            frame_window.pop(0)
            
        if len(frame_window) == 30:
            # --- 4. ROBUST NORMALIZATION ---
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # Translation: Wrist(0) to (0,0,0)
            data = data - data[:, 0, :].reshape(30, 1, 3)
            
            # Scaling: Wrist(0) to Middle Finger Base(9)
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 0:
                    data[i] = data[i] / scale
            
            # --- 5. DEEP LEARNING INFERENCE ---
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                label = class_names[pred.item()]
                score = conf.item() * 100
                
            # UI Overlay
            cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
            color = (0, 255, 0) if score > 80 else (0, 165, 255)
            cv2.putText(frame, f"{label}: {score:.1f}%", (15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Mudra ST-GCN (Tasks API)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()