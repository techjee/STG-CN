#The ST-GCN Model Architecture

"""Why this is "Research Grade"
Conv2d for GCN: We use a 1x1 convolution combined with a matrix multiplication 
of your adj (Adjacency Matrix). This is the standard "Kipf & Welling" approach
to Graph Convolutions.

Temporal Kernel: The kernel_size=(9, 1) means the model looks at 9 frames
 at a time to understand the "flow" of the mudra.

Dropout: Since you have 63 sequences, I added Dropout(0.3) to
 prevent overfitting (where the model just memorizes your videos instead of 
 learning the gestures)."""





import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import HandGraph

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super().__init__()
        # 1. Spatial Graph Convolution
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 2. Temporal Convolution (looking at frames)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=(kernel_size, 1), 
                      padding=((kernel_size-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3)
        )

    def forward(self, x, adj):
        # x shape: (Batch, Channels, Frames, Nodes)
        # Apply Spatial Graph Conv: Multiply by Adjacency Matrix
        x = self.gcn(x)
        x = torch.matmul(x, adj)
        
        # Apply Temporal Conv
        x = self.tcn(x)
        return x

class MudraClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.graph = HandGraph()
        # Register adjacency matrix as a buffer (it stays with the model)
        adj = torch.tensor(self.graph.adj, dtype=torch.float32)
        self.register_buffer('adj', adj)

        # Layers: (Channels: 3 -> 64 -> 128)
        self.block1 = STGCN_Block(3, 64)
        self.block2 = STGCN_Block(64, 128)
        
        # Global Pooling and Output
        self.fcn = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x: (Batch, 3, 30, 21)
        x = self.block1(x, self.adj)
        x = self.block2(x, self.adj)
        
        # Pool across frames and nodes
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        return self.fcn(x)

print("ST-GCN Model Architecture defined successfully.")