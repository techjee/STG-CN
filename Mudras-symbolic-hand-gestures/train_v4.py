import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from model import MudraClassifier

class MudraDatasetV2(Dataset):
    def __init__(self, csv_file, window_size=30):
        self.df = pd.read_csv(csv_file)
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples, self.labels = [], []
        
        print(f"V2 Mapping: {self.class_to_idx}")

        for vid_name, group in self.df.groupby('video_name'):
            label = group['label'].iloc[0]
            coords = group.filter(regex='[xyz]_').values
            
            # Sliding window with 50% overlap for more data
            for i in range(0, len(coords) - window_size, window_size // 2):
                window = coords[i : i + window_size].reshape(window_size, 21, 3).copy()
                if len(window) == window_size:
                    # AGNOSTIC LOGIC: Flip X-axis if it's a Left Hand (node 17 vs node 5)
                    if window[0, 17, 0] < window[0, 5, 0]:
                        window[:, :, 0] *= -1
                    
                    # NORMALIZATION: Wrist-Centered & Palm-Scaled
                    wrist = window[:, 0, :].copy()
                    window = window - wrist[:, np.newaxis, :]
                    for f in range(window_size):
                        scale = np.linalg.norm(window[f, 0] - window[f, 9])
                        if scale > 1e-6: window[f] /= scale
                    
                    self.samples.append(window)
                    self.labels.append(self.class_to_idx[label])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        data = torch.tensor(self.samples[idx], dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

def train_v2():
    dataset = MudraDatasetV2('normalized_mudras.csv') 
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = MudraClassifier(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training V2 on {len(dataset)} sequences...")
    for epoch in range(50):
        model.train()
        total_loss, correct = 0, 0
        for data, labels in train_loader:
            # ADDING JITTER: Makes the model robust to background noise
            noise = torch.randn_like(data) * 0.002
            data += noise
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/50] - Acc: {(100*correct/len(dataset)):.2f}%")

    torch.save(model.state_dict(), 'mudra_stgcn_v2.pth')
    print("V2 Training Complete!")

if __name__ == "__main__": train_v2()