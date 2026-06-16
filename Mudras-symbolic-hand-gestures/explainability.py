import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Import your existing local modules
from train_v2 import MudraDatasetV2
from model import MudraClassifier

def run_comprehensive_evaluation():
    # 1. Initialization
    dataset = MudraDatasetV2('normalized_mudras.csv')
    classes = dataset.classes
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = MudraClassifier(num_classes=len(classes))
    model_path = 'mudra_stgcn_v2.pth'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    y_true, y_pred = [], []
    joints = [
        "Wrist", "Th_CMC", "Th_MCP", "Th_IP", "Th_Tip",
        "In_MCP", "In_PIP", "In_DIP", "In_Tip",
        "Mi_MCP", "Mi_PIP", "Mi_DIP", "Mi_Tip",
        "Ri_MCP", "Ri_PIP", "Ri_DIP", "Ri_Tip",
        "Pi_MCP", "Pi_PIP", "Pi_DIP", "Pi_Tip"
    ]

    # --- PART 1: QUANTITATIVE PERFORMANCE ---
    print("\n[STEP 1/3] Calculating Performance Metrics...")
    with torch.no_grad():
        for data, labels in loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.numpy())

    # Print Classification Report to Terminal
    print("\n" + "="*50)
    print("FINAL RESEARCH PERFORMANCE SUMMARY")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=classes))
    
    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    print(f"Verified Accuracy: {acc:.2f}%")
    print(f"Improvement over MobileNet: +{acc - 86.45:.2f}%")
    print("="*50)

    # --- PART 2: CONFUSION MATRIX VISUALIZATION ---
    print("\n[STEP 2/3] Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Confusion Matrix: Bharatanatyam Mudra Recognition\n(Accuracy: {acc:.2f}%)', fontsize=14)
    plt.ylabel('Actual Mudra', fontsize=12)
    plt.xlabel('Predicted Mudra', fontsize=12)
    plt.tight_layout()
    plt.savefig('figure_confusion_matrix.png', dpi=300)
    print("[SUCCESS] Matrix saved as 'figure_confusion_matrix.png'")

    # --- PART 3: XAI (EXPLAINABLE AI) SALIENCY MAPS ---
    print("\n[STEP 3/3] Generating Joint Importance Maps...")
    for target_class in classes:
        # Find a representative sample for the class
        label_idx = classes.index(target_class)
        sample_idx = next(i for i, label in enumerate(dataset.labels) if label == label_idx)
        
        data, label = dataset[sample_idx]
        data = data.unsqueeze(0)
        data.requires_grad_()

        # Backward pass to find gradients (influence)
        output = model(data)
        score = output[0, label]
        model.zero_grad()
        score.backward()

        # Calculate importance (sum across coords, average over time)
        gradients = data.grad.data.abs()
        importance = gradients.sum(dim=1).mean(dim=1).squeeze().numpy()
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        # Plot individual bar charts
        plt.figure(figsize=(12, 5))
        plt.bar(joints, importance, color=plt.cm.plasma(importance))
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Topological Influence Map: {target_class}', fontsize=14)
        plt.ylabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(f'figure_xai_{target_class.lower()}.png', dpi=300)
        plt.close()
    
    print(f"[SUCCESS] {len(classes)} XAI maps generated.")
    print("\n[DONE] All research artifacts are ready for your publication.")

if __name__ == "__main__":
    run_comprehensive_evaluation()