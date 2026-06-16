"""import cv2, torch, numpy as np, time
from collections import deque
from model import MudraClassifier
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. FIXED CONFIGURATION (Final Sync) ---
# Order: Alapadmam(0), Katakamukam(1), Thripathakam(2), Shikaram(3), Kapitham(4)
class_names = ['Alapadmam', 'Katakamukam', 'Thripathakam', 'Shikaram', 'Kapitham']
MODEL_PATH = 'hand_landmarker.task'
CONF_THRESHOLD = 0.75  # Balanced for stability and speed

# --- 2. INITIALIZE AI ---
classifier = MudraClassifier(num_classes=len(class_names))
classifier.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
classifier.eval()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1, 
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. STATE & SMOOTHING ---
cap = cv2.VideoCapture(0)
frame_window = []
# Moving Average Buffer: Smoothes the bar graph so it doesn't flicker
prob_history = deque(maxlen=10) 
final_decision = "Scanning..."
current_probs = np.zeros(len(class_names))

print("\n--- BHARATANATYAM AI TUTOR: FINAL VERSION ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        
        # UI: Draw landmarks
        for lm in hand_lms:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)
            
        coords = []
        for lm in hand_lms: coords.extend([lm.x, lm.y, lm.z])
        frame_window.append(coords)
        
        if len(frame_window) == 30:
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- CHIRALITY MIRROR FIX (Left Hand Support) ---
            # Using Thumb Base (2) vs Pinky Base (17) for orientation
            if data[0, 2, 0] > data[0, 17, 0]: 
                data[:, :, 0] *= -1 

            # WRIST-CENTRIC NORMALIZATION
            data -= data[:, 0, :].reshape(30, 1, 3)
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            # INFERENCE
            input_t = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(classifier(input_t), dim=1)[0].numpy()
                
                # TEMPORAL SMOOTHING
                prob_history.append(probs)
                current_probs = np.mean(prob_history, axis=0)
                
                max_idx = np.argmax(current_probs)
                if current_probs[max_idx] > CONF_THRESHOLD:
                    final_decision = class_names[max_idx]
                else:
                    final_decision = f"Checking {class_names[max_idx]}..."

            frame_window.pop(0)
    else:
        # Reset buffers when no hand is detected
        frame_window.clear()
        prob_history.clear()
        current_probs = np.zeros(len(class_names))
        final_decision = "Scanning..."

    # --- 4. ANALYTICAL HUD (Graph UI) ---
    # Draw Background Panel
    cv2.rectangle(frame, (w-350, 20), (w-20, 240), (40, 40, 40), -1)
    cv2.putText(frame, "SKELETAL CONFIDENCE GRAPH", (w-330, 45), 2, 0.4, (200, 200, 200), 1)
    
    for i, prob in enumerate(current_probs):
        bar_w = int(prob * 180)
        y_pos = 80 + (i * 32)
        
        # Class Label
        txt_color = (0, 255, 0) if prob == max(current_probs) and prob > 0.1 else (255, 255, 255)
        cv2.putText(frame, f"{class_names[i][:4]}", (w-330, y_pos), 2, 0.6, txt_color, 1)
        
        # Bar Background
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+180, y_pos+5), (60, 60, 60), -1)
        # Prediction Bar (Blue for scanning, Green for locked)
        bar_color = (0, 255, 0) if prob > CONF_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+bar_w, y_pos+5), bar_color, -1)
        
        cv2.putText(frame, f"{prob*100:.0f}%", (w-60, y_pos), 2, 0.5, (200, 200, 200), 1)

    # --- 5. FINAL DECISION CARD ---
    cv2.rectangle(frame, (30, h-90), (500, h-30), (30, 30, 30), -1)
    status_color = (0, 255, 0) if final_decision in class_names else (0, 165, 255)
    cv2.putText(frame, f"STATUS: {final_decision}", (50, h-50), 2, 0.9, status_color, 2)

    cv2.imshow('Mudra Pro Tutor - ST-GCN V5 Final', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()"""





"""
import cv2, torch, numpy as np, time
from collections import deque
from model import MudraClassifier
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CRITICAL: UPDATED MAPPING (Synced with your Screenshot) ---
# Order must be: 0:Alapadmam, 1:Kapitham, 2:Katakamukam, 3:Shikaram, 4:Thripathakam
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']
MODEL_PATH = 'hand_landmarker.task'
CONF_THRESHOLD = 0.80  # Increased slightly because your model is now 87% accurate!

# --- 2. INITIALIZE AI ---
classifier = MudraClassifier(num_classes=len(class_names))
classifier.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
classifier.eval()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1, 
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. STATE & SMOOTHING ---
cap = cv2.VideoCapture(0)
frame_window = []
prob_history = deque(maxlen=10) 
final_decision = "Scanning..."
current_probs = np.zeros(len(class_names))

print("\n--- BHARATANATYAM AI TUTOR: 87.3% ACCURACY VERSION ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        for lm in hand_lms:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)
            
        coords = []
        for lm in hand_lms: coords.extend([lm.x, lm.y, lm.z])
        frame_window.append(coords)
        
        if len(frame_window) == 30:
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- CHIRALITY MIRROR FIX (Synced with Training logic) ---
            # Using Index Base (5) vs Pinky Base (17) to detect hand flip
            if data[0, 17, 0] < data[0, 5, 0]: 
                data[:, :, 0] *= -1 

            # WRIST-CENTRIC NORMALIZATION
            data -= data[:, 0, :].reshape(30, 1, 3)
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            # INFERENCE
            input_t = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(classifier(input_t), dim=1)[0].numpy()
                prob_history.append(probs)
                current_probs = np.mean(prob_history, axis=0)
                
                max_idx = np.argmax(current_probs)
                if current_probs[max_idx] > CONF_THRESHOLD:
                    final_decision = class_names[max_idx]
                else:
                    final_decision = f"Analyzing {class_names[max_idx]}..."

            frame_window.pop(0)
    else:
        frame_window.clear()
        prob_history.clear()
        current_probs = np.zeros(len(class_names))
        final_decision = "Scanning..."

    # --- 4. ANALYTICAL HUD (Similarity Graph) ---
    cv2.rectangle(frame, (w-350, 20), (w-20, 240), (40, 40, 40), -1)
    cv2.putText(frame, "SKELETAL SIMILARITY", (w-330, 45), 2, 0.4, (200, 200, 200), 1)
    
    for i, prob in enumerate(current_probs):
        bar_w = int(prob * 180)
        y_pos = 80 + (i * 32)
        txt_color = (0, 255, 0) if prob == max(current_probs) and prob > 0.1 else (255, 255, 255)
        cv2.putText(frame, f"{class_names[i][:4]}", (w-330, y_pos), 2, 0.6, txt_color, 1)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+180, y_pos+5), (60, 60, 60), -1)
        bar_color = (0, 255, 0) if prob > CONF_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+bar_w, y_pos+5), bar_color, -1)
        cv2.putText(frame, f"{prob*100:.0f}%", (w-60, y_pos), 2, 0.5, (200, 200, 200), 1)

    # --- 5. FINAL DECISION CARD ---
    cv2.rectangle(frame, (30, h-90), (500, h-30), (30, 30, 30), -1)
    status_color = (0, 255, 0) if final_decision in class_names else (0, 165, 255)
    cv2.putText(frame, f"STATUS: {final_decision}", (50, h-50), 2, 0.9, status_color, 2)

    cv2.imshow('ST-GCN Mudra Tutor 87% Accuracy', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()"""








"""

import cv2, torch, numpy as np, time
from collections import deque
from model import MudraClassifier
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIGURATION (SYNCED MAPPING) ---
# Order must match your 87.3% model: Alapadmam(0), Kapitham(1), Katakamukam(2), Shikaram(3), Thripathakam(4)
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']
MODEL_PATH = 'hand_landmarker.task'
CONF_THRESHOLD = 0.80  # Required for Status Lock
ALPHA = 0.25           # Smoothing speed (Lower = more stable)

# --- 2. INITIALIZE AI ---
classifier = MudraClassifier(num_classes=len(class_names))
classifier.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
classifier.eval()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. STATE & SMOOTHING ---
cap = cv2.VideoCapture(0)
frame_window = []
smoothed_probs = np.zeros(len(class_names))
final_status = "Scanning..."

print("\n--- ST-GCN V8: PERFECT CLASSIFICATION ONLINE ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1) # Mirror for user
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        for lm in hand_lms: cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)
            
        coords = []
        for lm in hand_lms: coords.extend([lm.x, lm.y, lm.z])
        frame_window.append(coords)
        
        if len(frame_window) == 30:
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- CHIRALITY FIX (Palm Vector Logic) ---
            # Calculates the hand's orientation vector to accurately flip Left Hand
            v1 = data[0, 5, :2] - data[0, 0, :2] # Wrist to Index
            v2 = data[0, 17, :2] - data[0, 0, :2] # Wrist to Pinky
            if (v1[0]*v2[1] - v1[1]*v2[0]) < 0: data[:, :, 0] *= -1 

            # NORMALIZATION
            data -= data[:, 0, :].reshape(30, 1, 3) # Center
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            # INFERENCE
            input_t = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                raw_probs = torch.nn.functional.softmax(classifier(input_t), dim=1)[0].numpy()
                # EMA FILTER: Smoothes the "Look Alike" graph
                smoothed_probs = ALPHA * raw_probs + (1 - ALPHA) * smoothed_probs
                
                max_idx = np.argmax(smoothed_probs)
                if smoothed_probs[max_idx] > CONF_THRESHOLD:
                    final_status = class_names[max_idx]
                else:
                    final_status = f"Matching {class_names[max_idx]}..."
            frame_window.pop(0)
    else:
        frame_window.clear(); smoothed_probs = np.zeros(len(class_names))
        final_status = "Scanning..."

    # --- 4. ANALYTICAL HUD (Graph UI) ---
    cv2.rectangle(frame, (w-350, 20), (w-20, 240), (40, 40, 40), -1)
    cv2.putText(frame, "AI SKELETAL SIMILARITY", (w-330, 45), 2, 0.4, (200, 200, 200), 1)
    for i, prob in enumerate(smoothed_probs):
        bar_w = int(prob * 180)
        y_pos = 80 + (i * 32)
        txt_color = (0, 255, 0) if prob == max(smoothed_probs) and prob > 0.1 else (255, 255, 255)
        cv2.putText(frame, f"{class_names[i][:4]}", (w-330, y_pos), 2, 0.6, txt_color, 1)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+180, y_pos+5), (60, 60, 60), -1)
        bar_color = (0, 255, 0) if prob > CONF_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+bar_w, y_pos+5), bar_color, -1)
        cv2.putText(frame, f"{prob*100:.0f}%", (w-60, y_pos), 2, 0.5, (200, 200, 200), 1)

    # --- 5. STATUS CARD ---
    cv2.rectangle(frame, (30, h-90), (500, h-30), (30, 30, 30), -1)
    status_color = (0, 255, 0) if final_status in class_names else (0, 165, 255)
    cv2.putText(frame, f"STATUS: {final_status}", (50, h-50), 2, 0.9, status_color, 2)

    cv2.imshow('ST-GCN Mudra Tutor V8 Perfect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()"""







import cv2, torch, numpy as np, time
from collections import deque
from model import MudraClassifier
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIGURATION (SYNCED MAPPING) ---
# Alapadmam(0), Kapitham(1), Katakamukam(2), Shikaram(3), Thripathakam(4)
class_names = ['Alapadmam', 'Kapitham', 'Katakamukam', 'Shikaram', 'Thripathakam']
MODEL_PATH = 'hand_landmarker.task'
CONF_THRESHOLD = 0.80  # Lock status above 80%
ALPHA = 0.20           # EMA Smoothing factor (Lower = smoother graph)

# --- 2. INITIALIZE AI ---
classifier = MudraClassifier(num_classes=len(class_names))
classifier.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
classifier.eval()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. STATE & SMOOTHING ---
cap = cv2.VideoCapture(0)
frame_window = []
smoothed_probs = np.zeros(len(class_names))
final_status = "Scanning..."

print("\n--- ST-GCN V8 MASTER: KAPITHAM & LEFT-HAND OPTIMIZED ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1) # User mirror
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        # Clean skeletal UI
        for lm in hand_lms: cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)
            
        coords = []
        for lm in hand_lms: coords.extend([lm.x, lm.y, lm.z])
        frame_window.append(coords)
        
        if len(frame_window) == 30:
            data = np.array(frame_window).reshape(30, 21, 3)
            
            # --- CHIRALITY FIX (Palm Orientation Vector) ---
            # Corrects Left Hand coordinates to match Right Hand training
            v1 = data[0, 5, :2] - data[0, 0, :2] 
            v2 = data[0, 17, :2] - data[0, 0, :2] 
            if (v1[0]*v2[1] - v1[1]*v2[0]) < 0: data[:, :, 0] *= -1 

            # --- NORMALIZATION & KAPITHAM FIX ---
            data -= data[:, 0, :].reshape(30, 1, 3) # Center at wrist
            
            # AMPLIFY DEPTH: Fixes "crowded joints" by stretching 3D space
            data[:, :, 2] *= 1.5 
            
            for i in range(30):
                scale = np.linalg.norm(data[i, 0] - data[i, 9])
                if scale > 1e-6: data[i] /= scale
            
            # INFERENCE
            input_t = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                raw_probs = torch.nn.functional.softmax(classifier(input_t), dim=1)[0].numpy()
                # Exponential Moving Average for smooth bar transitions
                smoothed_probs = ALPHA * raw_probs + (1 - ALPHA) * smoothed_probs
                
                max_idx = np.argmax(smoothed_probs)
                if smoothed_probs[max_idx] > CONF_THRESHOLD:
                    final_status = class_names[max_idx]
                else:
                    final_status = f"Matching {class_names[max_idx]}..."
            frame_window.pop(0)
    else:
        frame_window.clear(); smoothed_probs = np.zeros(len(class_names))
        final_status = "Scanning..."

    # --- 4. ANALYTICAL HUD (The Graph) ---
    cv2.rectangle(frame, (w-350, 20), (w-20, 240), (40, 40, 40), -1)
    cv2.putText(frame, "AI SKELETAL SIMILARITY", (w-330, 45), 2, 0.4, (200, 200, 200), 1)
    for i, prob in enumerate(smoothed_probs):
        bar_w = int(prob * 180)
        y_pos = 80 + (i * 32)
        txt_color = (0, 255, 0) if prob == max(smoothed_probs) and prob > 0.1 else (255, 255, 255)
        cv2.putText(frame, f"{class_names[i][:4]}", (w-330, y_pos), 2, 0.6, txt_color, 1)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+180, y_pos+5), (60, 60, 60), -1)
        bar_color = (0, 255, 0) if prob > CONF_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (w-250, y_pos-15), (w-250+bar_w, y_pos+5), bar_color, -1)
        cv2.putText(frame, f"{prob*100:.0f}%", (w-60, y_pos), 2, 0.5, (200, 200, 200), 1)

    # --- 5. STATUS HUD ---
    cv2.rectangle(frame, (30, h-90), (520, h-30), (30, 30, 30), -1)
    status_color = (0, 255, 0) if final_status in class_names else (0, 165, 255)
    cv2.putText(frame, f"STATUS: {final_status}", (50, h-50), 2, 0.9, status_color, 2)

    cv2.imshow('ST-GCN Mudra Tutor V8 Final', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()