import numpy as np

class HandGraph:
    def __init__(self):
        self.num_nodes = 21
        self.edges = [
            (0,1), (1,2), (2,3), (3,4),       # Thumb
            (0,5), (5,6), (6,7), (7,8),       # Index
            (0,9), (9,10), (10,11), (11,12),  # Middle
            (0,13), (13,14), (14,15), (15,16),# Ring
            (0,17), (17,18), (18,19), (19,20) # Pinky
        ]
        self.adj = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        a = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edges:
            a[i, j] = 1
            a[j, i] = 1
        # Add self-loops (important for GCN math)
        return a + np.eye(self.num_nodes)

# Test it
graph = HandGraph()
print("Adjacency Matrix Shape:", graph.adj.shape)