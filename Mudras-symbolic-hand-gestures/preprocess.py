# data normalizing

"""We need to do two things to your data:Translation Invariance: 
Move the "Wrist" (Node 0) to the center $(0,0,0)$ for every single frame.
Scale Invariance: Resize the hand so the distance from the wrist to 
the middle finger base is always 1.0."""






import pandas as pd
import numpy as np

def normalize_landmarks(df):
    cols = [c for c in df.columns if any(x in c for x in ['x_', 'y_', 'z_'])]
    data = df[cols].values.reshape(-1, 21, 3)
    
    # 1. Translation: Subtract wrist (Node 0) from all points
    wrists = data[:, 0, :].reshape(-1, 1, 3)
    data = data - wrists
    
    # 2. Scaling: Normalize by distance between Wrist(0) and Middle Finger Base(9)
    for i in range(len(data)):
        dist = np.linalg.norm(data[i, 0] - data[i, 9])
        if dist > 0:
            data[i] = data[i] / dist
            
    # Put back into dataframe
    df[cols] = data.reshape(len(df), -1)
    return df

print("Loading dataset...")
df = pd.read_csv('full_mudras_dataset.csv')
df_norm = normalize_landmarks(df)
df_norm.to_csv('normalized_mudras.csv', index=False)
print("SUCCESS: Data is now View-Invariant and saved to normalized_mudras.csv")