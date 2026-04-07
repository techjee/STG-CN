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
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']
CONFIDENCE_THRESHOLD = 0.95  # Only show if 95% sure
STABILITY_FRAMES = 5        # Must see same mudra 5 times in a row
prediction_buffer = deque(maxlen=STABILITY_FRAMES)

# --- 2. MODEL SETUP ---
model = MudraClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location=torch.device('cpu')))
model.eval()

# --- 3. MEDIAPIPE SETUP ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO 
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_window = []

print("--- REDEFINED V2 LIVE TEST STARTING ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    display_frame = cv2.flip(frame, 1)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=display_frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    
    current_label = "Scanning..."
    current_score = 0.0

    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        coords = []
        for lm in hand_lms:
            coords.extend([lm.x, lm.y, lm.z])
            # Draw landmarks for visual feedback
            cv2.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        
        frame_window.append(coords)
        if len(frame_window) > 30: frame_window.pop(0)
            
        if len(frame_window) == 30:
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- V2 MIRROR TRICK (Agnostic Logic) ---
            if data[0, 17, 0] < data[0, 5, 0]:
                data[:, :, 0] *= -1
            
            # NORMALIZATION
            data = data - data[:, 0, :].reshape(30, 1, 3) # Center on Wrist
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            # INFERENCE
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                # --- STABILITY FILTER ---
                if conf.item() > CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(pred.item())
                else:
                    prediction_buffer.append(-1) # Reset buffer if low confidence

                # Only update label if buffer is full of the same ID
                if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                    idx = prediction_buffer[0]
                    if idx != -1:
                        current_label = class_names[idx]
                        current_score = conf.item() * 100
                    else:
                        current_label = "Adjust Position"
                else:
                    current_label = "Stabilizing..."

    # UI OVERLAY
    cv2.rectangle(display_frame, (10, 10), (450, 90), (0, 0, 0), -1)
    color = (0, 255, 0) if current_score > 0 else (0, 165, 255)
    cv2.putText(display_frame, f"Mudra: {current_label}", (20, 50), 2, 1, color, 2)
    if current_score > 0:
        cv2.putText(display_frame, f"Confidence: {current_score:.1f}%", (20, 80), 2, 0.7, (255, 255, 255), 1)

    cv2.imshow('Mudra V2 - Stability Mode', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()