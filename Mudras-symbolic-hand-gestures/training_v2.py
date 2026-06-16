import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from model import MudraClassifier

class MudraDatasetV2(Dataset):
    def __init__(self, csv_file, window_size=30):
        self.df = pd.read_csv(csv_file)
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples, self.labels = [], []

        for vid_name, group in self.df.groupby('video_name'):
            label = group['label'].iloc[0]
            coords = group.filter(regex='[xyz]_').values
            
            for i in range(0, len(coords) - window_size, window_size // 2):
                window = coords[i : i + window_size].reshape(window_size, 21, 3).copy()
                if len(window) == window_size:
                    # --- INTERPRETABILITY: AGNOSTIC MIRROR LOGIC ---
                    # We compare Node 17 (Pinky Base) and Node 5 (Index Base).
                    # If Pinky is 'left' of Index, it's likely a Left Hand. 
                    # By flipping X, we make the model 'Hand-Blind' (Chirality Agnostic).
                    if window[0, 17, 0] < window[0, 5, 0]:
                        window[:, :, 0] *= -1
                    
                    # --- INTERPRETABILITY: WRIST-CENTRIC NORMALIZATION ---
                    # We subtract the Wrist (Node 0) from all points.
                    # This makes the gesture 'Position Invariant' (doesn't matter where you are in the frame).
                    wrist = window[:, 0, :].copy()
                    window = window - wrist[:, np.newaxis, :]
                    
                    # Scaling by Palm Size (Node 0 to Node 9) for distance invariance
                    for f in range(window_size):
                        scale = np.linalg.norm(window[f, 0] - window[f, 9])
                        if scale > 1e-6: window[f] /= scale
                        
                    self.samples.append(window)
                    self.labels.append(self.class_to_idx[label])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32).permute(2, 0, 1), \
               torch.tensor(self.labels[idx], dtype=torch.long)

def train_system():
    full_ds = MudraDatasetV2('normalized_mudras.csv')
    
    # --- GENUINE DATA SCIENCE: 80/20 SPLIT ---
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_set, val_set = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    model = MudraClassifier(num_classes=len(full_ds.classes))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Genuine Training: {train_size} Train / {val_size} Val")

    for epoch in range(50):
        model.train()
        for data, labels in train_loader:
            # --- GAUSSIAN JITTER AUGMENTATION ---
            # Adding 0.002 std-dev noise to landmarks. This forces the model
            # to learn the skeletal 'topology' rather than hardcoded coordinates.
            noise = torch.randn_like(data) * 0.002
            data = data + noise
            
            optimizer.zero_grad()
            loss = criterion(model(data), labels)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} Completed.")

    torch.save(model.state_dict(), 'mudra_stgcn_v2.pth')
    print("Model Saved: mudra_stgcn_v2.pth")

if __name__ == "__main__": train_system()