import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train_v2 import MudraDatasetV2 
from model import MudraClassifier

def evaluate_v2():
    dataset = MudraDatasetV2('normalized_mudras.csv') 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = MudraClassifier(num_classes=len(dataset.classes))
    model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    print("\n--- VERSION 2 PERFORMANCE REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

if __name__ == "__main__": evaluate_v2()