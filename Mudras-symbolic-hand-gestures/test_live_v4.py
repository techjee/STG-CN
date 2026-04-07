import cv2, torch, numpy as np, os, time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import MudraClassifier
from collections import deque

# --- CONFIG ---
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']
CONF_THRESHOLD = 0.88 
STABILITY_BUFFER = deque(maxlen=5) # Smooths out background "glitches"

model = MudraClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
model.eval()

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_window = []
label, score = "Scanning...", 0.0

print("\n--- V2 SYSTEM ONLINE: STABILITY MODE ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    
    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        current_frame = []
        for lm in hand_lms:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)
            current_frame.extend([lm.x, lm.y, lm.z])
        
        frame_window.append(current_frame)
        if len(frame_window) > 30: frame_window.pop(0)
            
        if len(frame_window) == 30:
            data = np.array(frame_window).reshape(30, 21, 3)
            # AGNOSTIC MIRROR (V2): Compare Node 5 (Index) and Node 17 (Pinky)
            if data[0, 17, 0] < data[0, 5, 0]: data[:, :, 0] *= -1
            
            # NORMALIZATION
            data = data - data[:, 0, :].reshape(30, 1, 3) 
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            input_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                # UPDATE UI LOGIC
                score = conf.item() * 100
                if conf.item() > CONF_THRESHOLD:
                    STABILITY_BUFFER.append(pred.item())
                    if len(set(STABILITY_BUFFER)) == 1: # All 5 frames must match
                        label = class_names[pred.item()]
                else:
                    label = "Unknown / Adjust Hand"
                    STABILITY_BUFFER.clear()

    # UI RENDERING
    cv2.rectangle(frame, (0, 0), (580, 80), (0, 0, 0), -1)
    color = (0, 255, 0) if score > (CONF_THRESHOLD*100) else (0, 165, 255)
    cv2.putText(frame, f"{label} ({score:.1f}%)", (20, 50), 2, 1.1, color, 2)

    cv2.imshow('ST-GCN V2 Live', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()