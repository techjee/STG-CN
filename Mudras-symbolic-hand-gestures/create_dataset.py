# DATA LOADING 

"""5: Temporal Windowing (Data Loading)
We need to convert your long CSV into "Samples." 
For example, if a video is 90 frames long, we might break it
 into three 30-frame samples."""




import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MudraDataset(Dataset):
    def __init__(self, csv_file, window_size=30):
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.classes = self.df['label'].unique()
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        self.samples = []
        self.labels = []
        
        # Group by video to keep sequences together
        for vid_name, group in self.df.groupby('video_name'):
            label = group['label'].iloc[0]
            # Extract only the x, y, z columns
            coords = group.filter(regex='[xyz]_').values
            
            # Create windows of 'window_size'
            for i in range(0, len(coords) - window_size, window_size // 2):
                window = coords[i : i + window_size]
                if len(window) == window_size:
                    # Reshape to (Window, Nodes, XYC) -> (30, 21, 3)
                    self.samples.append(window.reshape(window_size, 21, 3))
                    self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ST-GCN usually expects (Channels, Frames, Nodes)
        # We transpose (30, 21, 3) -> (3, 30, 21)
        data = torch.tensor(self.samples[idx], dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

# Test the loader
dataset = MudraDataset('normalized_mudras.csv')
print(f"Total sequences created: {len(dataset)}")
print(f"Data shape (Channels, Frames, Nodes): {dataset[0][0].shape}")