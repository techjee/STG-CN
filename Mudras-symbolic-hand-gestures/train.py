#MODEL TRAINING 

"""Shuffle=True: This is critical. It ensures the model doesn't just
 memorize the order of your videos (e.g., all Alapadmam then all Kapitham), 
 but actually learns the skeletal patterns .
Backpropagation: Every time the model is "wrong," loss.backward() calculates
 exactly how much to adjust the "weights" in your Graph Convolutional layers 
 to be more accurate next time."""







import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from create_dataset import MudraDataset
from model import MudraClassifier

# 1. Hyperparameters
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# 2. Load Data
dataset = MudraDataset('normalized_mudras.csv')
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialize Model, Loss, and Optimizer
num_classes = len(dataset.classes)
model = MudraClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training for {num_classes} mudras...")

# 4. The Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass (Backpropagation)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate Accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {accuracy:.2f}%")

# 5. Save the Brain
torch.save(model.state_dict(), 'mudra_stgcn.pth')
print("Training Complete! Model saved as mudra_stgcn.pth")