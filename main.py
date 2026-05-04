import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_builder import get_combined_dataloaders

# 1. Device Configuration
# This automatically uses Apple Silicon (MPS), Nvidia GPU (CUDA), or defaults to CPU

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. Load the Combined Data (Digits + Symbols)

train_loader = get_combined_dataloaders(batch_size=64)
print(f"Total Training Batches: {len(train_loader)}")

# 3. The CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        
        # CRITICAL UPDATE: 14 classes (0-9 digits + 4 symbols)
        self.fc2 = nn.Linear(in_features=128, out_features=14) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x


# Initialize the model and send it to your M1's GPU
model = SimpleCNN().to(device)
print(model)

#now, train loop

# 1. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
# Adam is a widely used optimizer that adapts the learning rate dynamically
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. The Training Loop
epochs = 5 # Number of times to loop through the entire dataset

print("Starting training...")
for epoch in range(epochs):
    running_loss = 0.0
    
    # Iterate through the batches of images and labels
    for i, (images, labels) in enumerate(train_loader):
        # Move data to the same device as the model (MPS in your case)
        images = images.to(device)
        labels = labels.to(device)
        
        # Step 1: Zero the parameter gradients
        # PyTorch accumulates gradients, so we must clear them every batch
        optimizer.zero_grad()
        
        # Step 2: Forward pass (make predictions)
        outputs = model(images)
        
        # Step 3: Calculate the loss (how wrong the predictions were)
        loss = criterion(outputs, labels)
        
        # Step 4: Backward pass (calculate the gradients)
        loss.backward()
        
        # Step 5: Optimize (update the weights)
        optimizer.step()
        
        # Keep track of the loss for monitoring
        running_loss += loss.item()
        
        # Print an update every 300 batches
        if (i + 1) % 300 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 300:.4f}")
            running_loss = 0.0

print("Training complete!")


""" 

# 3. Evaluation on Test Data
print("\nEvaluating model on test data...")
model.eval() # Switch model from training to evaluation mode
correct = 0
total = 0

# Disable gradient calculation for testing to speed things up
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        # Grab the highest probability prediction
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%') """

# 4. Save the Model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model weights saved to 'mnist_cnn.pth'")

