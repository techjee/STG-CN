# smoke test

"""import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

# 1. Setup the New Tasks API (Requires hand_landmarker.task file)
model_path = 'hand_landmarker.task' 
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Paths - Exactly as they appear in your sidebar
dataset_path = "Mudras(Symbolic Hand Poses)dataset/Alapadmam"
output_file = "alapadmam_landmarks.csv"
data_list = []

if not os.path.exists(dataset_path):
    print(f"ERROR: Folder not found at {dataset_path}. Check the folder name for spaces!")
else:
    print(f"Starting extraction from: {dataset_path}")

# 3. Process Videos
for video_file in os.listdir(dataset_path):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(dataset_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Convert frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    row = [video_file, frame_count]
                    for lm in hand_lms:
                        row.extend([lm.x, lm.y, lm.z])
                    data_list.append(row)
            frame_count += 1
        cap.release()
        print(f"Finished: {video_file}")

# 4. Save to CSV
if data_list:
    columns = ['video_name', 'frame_id']
    for i in range(21): columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    pd.DataFrame(data_list, columns=columns).to_csv(output_file, index=False)
    print(f"SUCCESS! Created {output_file} with {len(data_list)} rows of spatial data.")
else:
    print("No landmarks were detected. Check if the videos are clear!")"""









#actuall code with looping for  all mudras 

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os
from tqdm import tqdm # This adds a progress bar

# 1. Setup Tasks API
model_path = 'hand_landmarker.task' 
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Base Path (Point to the main folder containing all your Mudra subfolders)
base_dataset_path = "Mudras(Symbolic Hand Poses)dataset"
output_file = "full_mudras_dataset.csv"
data_list = []

# Get list of all folders (Alapadmam, Kapitham, etc.)
mudra_folders = [f for f in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, f))]

print(f"Found {len(mudra_folders)} mudras to process: {mudra_folders}")

# 3. Loop through every folder
for mudra_name in mudra_folders:
    mudra_path = os.path.join(base_dataset_path, mudra_name)
    video_files = [v for v in os.listdir(mudra_path) if v.endswith(".mp4")]
    
    print(f"\nProcessing Mudra: {mudra_name} ({len(video_files)} videos)")
    
    # tqdm creates the progress bar for videos in this folder
    for video_file in tqdm(video_files, desc=f"Folder: {mudra_name}"):
        video_path = os.path.join(mudra_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Convert frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    # 'label' allows the ST-GCN to differentiate between mudras
                    row = [mudra_name, video_file, frame_count]
                    for lm in hand_lms:
                        row.extend([lm.x, lm.y, lm.z])
                    data_list.append(row)
            frame_count += 1
        cap.release()

# 4. Save the full dataset
if data_list:
    columns = ['label', 'video_name', 'frame_id']
    for i in range(21): columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"\n--- MISSION ACCOMPLISHED ---")
    print(f"Total Rows Extracted: {len(df)}")
    print(f"Final Dataset Saved as: {output_file}")
else:
    print("\nNo data was extracted. Please check your folder paths!")