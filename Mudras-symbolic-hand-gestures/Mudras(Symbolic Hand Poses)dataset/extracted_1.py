import cv2
import mediapipe as mp
import pandas as pd
import os

# 1. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 2. Path to your dataset (Based on your sidebar image)
dataset_path = "Mudras(Symbolic Hand Poses)dataset/Alapadmam"
output_file = "alapadmam_landmarks.csv"

data_list = []

print(f"Starting extraction from {dataset_path}...")

# 3. Loop through all videos in the folder
for video_file in os.listdir(dataset_path):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(dataset_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [video_file, frame_count]
                    # Extract 21 landmarks (x, y, z)
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    data_list.append(row)
            
            frame_count += 1
        cap.release()
        print(f"Finished: {video_file}")

# 4. Save to CSV
columns = ['video_name', 'frame_id']
for i in range(21):
    columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

df = pd.DataFrame(data_list, columns=columns)
df.to_csv(output_file, index=False)
print(f"Successfully saved all landmarks to {output_file}")