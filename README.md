# A View-Invariant Skeleton-Based Spatio-Temporal Graph Convolutional Framework for Bharatanatyam Mudra Sequence Recognition

An artificial intelligence tutoring and digital preservation framework that leverages Spatio-Temporal Graph Convolutional Networks (ST-GCN) and MediaPipe to recognize classical Indian dance hand gestures (Mudras) from video sequences. 

This architecture transitions mudra classification from traditional static, pixel-dependent frame matching to dynamic, topology-aware sequence modeling—achieving a **92.06% overall accuracy** and outperforming a benchmark CNN-MobileNet pipeline by **+5.61%**.

---

## 🚀 Key Framework Features

* **Geometric & Spatial Invariance:** Built-in translation and scale normalization layers anchor coordinate landmarks with the wrist as the mathematical origin $(0,0,0)$ and normalize features by palm size to counteract camera distance variations.
* **Chirality-Agnostic Learning:** Incorporates real-time index-to-pinky finger relative position tracking to dynamically mirror left-hand coordinates, creating a "hand-blind" training dataset profile.
* **Explainable AI (XAI):** Uses post-hoc gradient-based saliency mapping to spatially aggregate and temporally average coordinate backpropagation gradients, producing joint-level topological influence maps.

---

## 📂 Repository Structure

Based on the core project space initialization configurations:

```text
Mudras-symbolic-hand-gestures/
├── Mudras(Symbolic Hand Poses)dataset/ # Raw .mp4 video directories grouped by class labels [cite: 582, 591]
├── venv/                              # Python virtual environment isolated binaries
├── graph_utils.py                     # Hand skeleton topological definition & adjacency matrix buffer [cite: 624-625]
├── model.py                           # PyTorch definition of the STGCN_Block and MudraClassifier
├── training_v2.py                     # Unified data loading (MudraDatasetV2) and 80/20 train/validation loop
├── explainability.py                  # Evaluation suite for reports, matrices, and gradient saliency mapping [cite: 687, 802]
├── testing_live_v2.py                 # Real-time inference application utilizing webcam video streams
├── create_dataset.py                  # Raw feature engineering video extraction script [cite: 593-594]
├── preprocess.py                      # Translation, scale reduction, and view-invariant baseline normalizer [cite: 609-611]
├── full_mudras_dataset.csv            # Extracted raw spatial keypoint tracking target dataset
├── normalized_mudras.csv              # Fully transformed view-invariant coordinate matrix dataset
├── hand_landmarker.task               # Bundled MediaPipe hand landmarker task asset model 
├── mudra_stgcn_v2.pth                 # Saved production state dictionary deep learning model weights [cite: 871]
├── figure_confusion_matrix.png        # Coded confusion matrix plot asset [cite: 779-781]
└── figure_xai_*.png                   # Generated topological joint importance map bar charts

```
## 🛠️ Installation & Setup
1. Clone the Workspace
```Bash
git clone [https://github.com/yourusername/Mudras-symbolic-hand-gestures.git](https://github.com/yourusername/Mudras-symbolic-hand-gestures.git)
cd Mudras-symbolic-hand-gestures
```

2. Activate Virtual Environment & Install Dependencies
Ensure you have Python 3.8+ initialized, then configure your environment paths:
# On Windows PowerShell:
```Bash
.\venv\Scripts\Activate.ps1
```

# Install required deep learning and processing dependencies
```Bash
pip install torch numpy pandas opencv-python mediapipe scikit-learn matplotlib seaborn tqdm
```

3. Verify MediaPipe Model Task Placement
```Bash
Path: ./hand_landmarker.task
```


## 🏃 Execution PipelineStep 

 Step 1: Automated Keypoint ExtractionProcesses the raw video datasets folder by folder using OpenCV, maps landmarks via MediaPipe, and saves them:
```Bash
python create_dataset.py
```

 Step 2: Spatial NormalizationTransforms raw data arrays into normalized, view-invariant coordinates anchored around a central wrist origin:
```Bash
python preprocess.py
```

 Step 3: Model OptimizationExecutes the data loader segmenting sequences into overlapping sliding windows (15-frame stride) with random Gaussian jitter regularization ($\sigma = 0.002$). Runs an 80/20 training cycle for 50 epochs and exports model parameters to disk:
```Bash
python training_v2.py
```

 Step 4: Run Post-Hoc Interpretability MetricsGenerates precision/recall metrics, plots the confusion matrix, and maps backpropagation gradients to generate individual joint saliency charts:
```Bash
python explainability.py
```

 Step 5: Launch Live Interactive TutorDeploys real-time hardware inference over a live webcam feed. Includes horizontal mirror mapping, Exponential Moving Average (EMA) filtering (alpha=0.20), and a live analytical similarity HUD:
```Bash
python testing_live_v2.py
```

## 🧠 Architectural Insights (XAI Interpretation)
Rather than operating as an uninterpretable "black box," our gradient-based saliency pipeline explicitly demonstrates that the ST-GCN maps decisions directly onto human biomechanical rules:

Katakamukam: Peak activations cluster tightly near thumb knuckles (CMC, MCP) and little finger distal joints, capturing grasp-like finger oppositions.

Kapitham: Focus shifts intensely to the proximal boundaries of the index and thumb base joints, identifying global hand closure.

Alapadmam: Relative structural importance is distributed dynamically across all digits, verifying open palm patterns and high intra-class freedom.