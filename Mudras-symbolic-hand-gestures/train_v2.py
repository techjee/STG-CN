import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from model import MudraClassifier

# --- 1. AGNOSTIC DATASET CLASS ---
class MudraDatasetV2(Dataset):
    def __init__(self, csv_file, window_size=30):
        self.df = pd.read_csv(csv_file)
        # CRITICAL: Always sort classes so Index 0 is always 'Alapadmam'
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        self.samples = []
        self.labels = []
        
        print(f"V2 Mapping: {self.class_to_idx}")

        for vid_name, group in self.df.groupby('video_name'):
            label = group['label'].iloc[0]
            # Get all x, y, z columns
            coords = group.filter(regex='[xyz]_').values
            
            # Sliding window with 50% overlap for more data
            for i in range(0, len(coords) - window_size, window_size // 2):
                window = coords[i : i + window_size].reshape(window_size, 21, 3).copy()
                
                if len(window) == window_size:
                    # --- THE V2 MIRROR TRICK ---
                    # Logic: If Pinky-Base-X (node 17) < Index-Base-X (node 5), 
                    # it's a Left Hand. We flip it to look like a Right Hand.
                    if window[0, 17, 0] < window[0, 5, 0]:
                        window[:, :, 0] = window[:, :, 0] * -1 # Flip X axis
                    
                    # Normalization (Centered at Wrist, Scaled by Palm)
                    wrist = window[:, 0, :].copy()
                    window = window - wrist[:, np.newaxis, :]
                    
                    for f in range(window_size):
                        scale = np.linalg.norm(window[f, 0] - window[f, 9])
                        if scale > 0:
                            window[f] = window[f] / scale
                    
                    self.samples.append(window)
                    self.labels.append(self.class_to_idx[label])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        # Convert to Tensor (C, T, V) -> (3, 30, 21)
        data = torch.tensor(self.samples[idx], dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

# --- 2. TRAINING CONFIGURATION ---
def train_v2():
    dataset = MudraDatasetV2('normalized_mudras.csv') # Use your existing CSV
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MudraClassifier(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting Version 2 Training on {len(dataset)} augmented sequences...")

    for epoch in range(50):
        model.train()
        total_loss = 0
        correct = 0
        
        for data, labels in train_loader:
            # Add tiny random noise (Jitter) for robustness
            noise = torch.randn_like(data) * 0.002
            data = data + noise
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / len(dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/50] - Loss: {total_loss/len(train_loader):.4f} - Acc: {accuracy:.2f}%")

    # Save as V2
    torch.save(model.state_dict(), 'mudra_stgcn_v2.pth')
    print("Version 2 Training Complete! Saved as mudra_stgcn_v2.pth")

if __name__ == "__main__":
    train_v2()