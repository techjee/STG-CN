import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from model import MudraClassifier

# --- 1. FINAL MAPPING ---
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']

# --- 2. ASSET CHECKS ---
if not os.path.exists('hand_landmarker.task'):
    exit("CRITICAL: 'hand_landmarker.task' missing!")

# --- 3. MODEL INITIALIZATION ---
model = MudraClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('mudra_stgcn_v2.pth'))
model.eval()

# --- 4. MEDIAPIPE SETUP ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO 
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 5. WEBCAM ---
cap = cv2.VideoCapture(0) 

frame_window = []
print("\n--- VERSION 2: FULL SYSTEM ONLINE ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1) # Mirror for natural feedback
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        current_frame = []
        
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            current_frame.extend([lm.x, lm.y, lm.z])
        
        frame_window.append(current_frame)
        if len(frame_window) > 30:
            frame_window.pop(0)
            
        if len(frame_window) == 30:
            # --- 6. PREPROCESSING ---
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- THE "CHIRALITY" FIX ---
            # Compare Thumb tip (4) and Pinky tip (20).
            # If the thumb is on the 'wrong' side relative to your training data,
            # we flip the X-axis. This makes 'front' and 'back' look the same.
            thumb_x = data[0, 4, 0]
            pinky_x = data[0, 20, 0]
            
            # Try this logic: If it works 'backward' but not 'front', 
            # the model expects Thumb to be on a specific side.
            if thumb_x < pinky_x: 
                data[:, :, 0] = data[:, :, 0] * -1
            
            # NORMALIZATION
            data = data - data[:, 0, :].reshape(30, 1, 3) 
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] = data[i] / scale
            
            # --- 7. INFERENCE ---
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                label = class_names[pred.item()]
                score = conf.item() * 100
                
            # --- 8. UI OVERLAY ---
            cv2.rectangle(frame, (0, 0), (550, 75), (0, 0, 0), -1)
            color = (0, 255, 0) if score > 80 else (0, 165, 255)
            cv2.putText(frame, f"V2: {label} ({score:.1f}%)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    cv2.imshow('ST-GCN Mudra System V2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()