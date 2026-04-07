import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from model import MudraClassifier

# --- 1. FINAL MAPPING (Must match your ID Scanner exactly) ---
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']

# --- 2. ASSET CHECKS ---
if not os.path.exists('hand_landmarker.task'):
    exit("ERROR: 'hand_landmarker.task' missing! Download it to this folder.")
if not os.path.exists('mudra_stgcn_v2.pth'):
    exit("ERROR: 'mudra_stgcn_v2.pth' not found! Run train_v2.py first.")

# --- 3. MODEL INITIALIZATION ---
model = MudraClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location=torch.device('cpu')))
model.eval()

# --- 4. MEDIAPIPE SETUP (VIDEO MODE) ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO 
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 5. WEBCAM INITIALIZATION ---
cap = cv2.VideoCapture(0) # If screen is black, try cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_window = []
print("\n--- ST-GCN VERSION 2: LIVE DEMO ACTIVE ---")
print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1) # Natural Mirror View
    
    # Process with MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        current_frame = []
        
        # Draw Landmarks for the UI
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            current_frame.extend([lm.x, lm.y, lm.z])
        
        frame_window.append(current_frame)
        if len(frame_window) > 30:
            frame_window.pop(0)
            
        if len(frame_window) == 30:
            # --- 6. AGNOSTIC PREPROCESSING ---
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # MIRROR CHECK: Ensures left and right hands look identical to the model
            # Uses Landmark 5 (Index Base) vs 17 (Pinky Base)
            if data[0, 17, 0] < data[0, 5, 0]:
                data[:, :, 0] = data[:, :, 0] * -1
            
            # NORMALIZATION (Sync with Training V2)
            data = data - data[:, 0, :].reshape(30, 1, 3) # Center on Wrist
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] = data[i] / scale
            
            # --- 7. ST-GCN INFERENCE ---
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                label = class_names[pred.item()]
                if conf.item() > 0.95: 
                     label = class_names[pred.item()]
                else:
                    label = "Unknown / Adjust Hand"
                    score = conf.item() * 100
                
            # --- 8. UI OVERLAY ---
            cv2.rectangle(frame, (0, 0), (500, 80), (0, 0, 0), -1)
            color = (0, 255, 0) if score > 85 else (0, 165, 255)
            cv2.putText(frame, f"{label} ({score:.1f}%)", (20, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow('Mudra Recognition System V2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()