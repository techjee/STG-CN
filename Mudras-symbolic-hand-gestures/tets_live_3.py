import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque
from model import MudraClassifier

# --- 1. CONFIGURATION ---
# Ensure these match your training order exactly
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']
CONFIDENCE_THRESHOLD = 0.95  # Only show if 95% sure
STABILITY_FRAMES = 5        # Must see the same mudra 5 times in a row
prediction_buffer = deque(maxlen=STABILITY_FRAMES)

# --- 2. MODEL INITIALIZATION ---
model = MudraClassifier(num_classes=len(class_names))
# Loading the ST-GCN V2 weights
model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location=torch.device('cpu')))
model.eval()

# --- 3. MEDIAPIPE ASSET SETUP ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO 
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 4. LIVE STREAM SETUP ---
cap = cv2.VideoCapture(0)
frame_window = []

print("--- ST-GCN V2: STABILITY MODE ACTIVE ---")
print("Requirement: Hold Mudra steady for 5 frames with >95% confidence.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    # Mirror the frame for a natural user experience
    display_frame = cv2.flip(frame, 1)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=display_frame)
    # MediaPipe detection
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    
    display_label = "Scanning..."
    display_score = 0.0

    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        coords = []
        for lm in hand_lms:
            coords.extend([lm.x, lm.y, lm.z])
            # Draw green dots on landmarks
            cv2.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        
        frame_window.append(coords)
        if len(frame_window) > 30: frame_window.pop(0)
            
        if len(frame_window) == 30:
            # Convert window to numpy for preprocessing
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- AGNOSTIC MIRROR LOGIC ---
            # Flips Left Hand to look like Right Hand
            if data[0, 17, 0] < data[0, 5, 0]:
                data[:, :, 0] *= -1
            
            # NORMALIZATION (Wrist-centered and Palm-scaled)
            data = data - data[:, 0, :].reshape(30, 1, 3)
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            # ST-GCN INFERENCE
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                # --- STABILITY LOGIC ---
                # Only add to buffer if confidence is very high
                if conf.item() > CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(pred.item())
                else:
                    prediction_buffer.append(-1) # Low confidence reset

                # Check if all 5 frames in buffer are the same Mudra
                if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                    idx = prediction_buffer[0]
                    if idx != -1:
                        display_label = class_names[idx]
                        display_score = conf.item() * 100
                    else:
                        display_label = "Invalid Gesture"
                else:
                    display_label = "Stabilizing..."

    # --- UI RENDERING ---
    cv2.rectangle(display_frame, (0, 0), (450, 100), (0, 0, 0), -1)
    # Color changes to green only when a valid Mudra is locked in
    text_color = (0, 255, 0) if display_score > 0 else (0, 165, 255)
    
    cv2.putText(display_frame, f"Mudra: {display_label}", (20, 45), 
                cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 2)
    
    if display_score > 0:
        cv2.putText(display_frame, f"Confidence: {display_score:.1f}%", (20, 85), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('ST-GCN Mudra Recognition V2', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()