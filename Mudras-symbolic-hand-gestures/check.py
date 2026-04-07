"""import mediapipe as mp
print("--- MediaPipe Attributes ---")
print(dir(mp))
print("---------------------------")
if hasattr(mp, 'solutions'):
    print("SUCCESS: 'solutions' found!")
else:
    print("ERROR: 'solutions' not found. You are likely on a version/Python combo that doesn't support the legacy API.")"""



import pandas as pd
df = pd.read_csv('normalized_mudras.csv')
# This shows exactly how the "Brain" was fed the labels
mapping = sorted(df['label'].unique())
for i, name in enumerate(mapping):
    print(f"ID {i}: {name}")