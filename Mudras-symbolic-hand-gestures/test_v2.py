import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train_v2 import MudraDatasetV2 # Reuses the V2 logic
from model import MudraClassifier

def evaluate_v2():
    # 1. Load Dataset using the V2 Agnostic Logic
    dataset = MudraDatasetV2('normalized_mudras.csv') 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Initialize Model
    model = MudraClassifier(num_classes=len(dataset.classes))
    
    # 3. Load the V2 Brain
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('mudra_stgcn_v2.pth'))
    
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Evaluating Version 2 on {len(dataset)} sequences...")

    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # 4. Generate Professional Report
    print("\n--- VERSION 2 PERFORMANCE REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    
    # 5. Confusion Matrix (Optional but great for Viva)
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate_v2()