"""import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train_v2 import MudraDatasetV2
from model import MudraClassifier

def run_evaluation():
    dataset = MudraDatasetV2('normalized_mudras.csv')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = MudraClassifier(num_classes=len(dataset.classes))
    model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for data, labels in loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.numpy())

    print("\n" + "="*50)
    print("GENUINE PERFORMANCE SUMMARY (ST-GCN V2)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=dataset.classes))
    
    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    
    print("\n--- REVIEW-READY CONCLUSION ---")
    print(f"Achieved Accuracy: {acc:.2f}%")
    print("Benchmark Comparison (CNN-MobileNet Base Paper): 86.45%")
    print(f"Improvement: +{acc - 86.45:.2f}%")
    print("Key Advantage: ST-GCN captures Spatio-Temporal dependencies which are")
    print("lost in frame-based CNN architectures.")
    print("="*50)

if __name__ == "__main__": run_evaluation()"""







import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train_v2 import MudraDatasetV2
from model import MudraClassifier

def run_evaluation():
    # 1. Load Data and Model
    dataset = MudraDatasetV2('normalized_mudras.csv')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = MudraClassifier(num_classes=len(dataset.classes))
    model.load_state_dict(torch.load('mudra_stgcn_v2.pth', map_location='cpu'))
    model.eval()

    y_true, y_pred = [], []
    
    # 2. Perform Inference
    with torch.no_grad():
        for data, labels in loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.numpy())

    # 3. Text-based Metrics
    print("\n" + "="*50)
    print("GENUINE PERFORMANCE SUMMARY (ST-GCN V2)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=dataset.classes))
    
    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    
    print("\n--- REVIEW-READY CONCLUSION ---")
    print(f"Achieved Accuracy: {acc:.2f}%")
    print("Benchmark Comparison (CNN-MobileNet Base Paper): 86.45%")
    print(f"Improvement: +{acc - 86.45:.2f}%")
    print("Key Advantage: ST-GCN captures Spatio-Temporal dependencies which are")
    print("lost in frame-based CNN architectures.")
    print("="*50)

    # 4. Generate and Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.classes, 
                yticklabels=dataset.classes)
    
    plt.title(f'Confusion Matrix: Bharatanatyam Mudra Recognition\n(Accuracy: {acc:.2f}%)', fontsize=14)
    plt.ylabel('Actual Mudra', fontsize=12)
    plt.xlabel('Predicted Mudra', fontsize=12)
    plt.tight_layout()
    
    # Save for the research paper
    plt.savefig('confusion_matrix_v2.png', dpi=300)
    print("\n[INFO] Confusion matrix saved as 'confusion_matrix_v2.png'")
    plt.show()

if __name__ == "__main__": 
    run_evaluation()