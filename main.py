import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from data_builder import get_combined_dataloaders as get_dataloaders
from data_builder2 import get_dataloaders


# ignore this line, its for selecting which GPU to use 
# it automatically uses Apple Silicon (MPS), Nvidia GPU (CUDA)
# or defaults to CPU if neither available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#Loading the Combined Data (Digits + Symbols)
train_loader = get_dataloaders(batch_size=64)
print(f"Total Training Batches: {len(train_loader)}")

#The CNN Architecture; main shiii
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        
        # isko phir dekhna ek baar 
        self.fc2 = nn.Linear(in_features=128, out_features=14) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

# Initializing the model and sending it to gpu/cpu (see device variable upar)
model = SimpleCNN().to(device)
print(model)

#now we train the loop

# 1. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
# Adam is a widely used optimizer that adapts the learning rate dynamically
optimizer = optim.Adam(model.parameters(), lr=0.001)

# wtv gemini says bro^^

# 2. The Training Loop
epochs = 5 # Number of times to loop through the entire dataset

# see NN theory for more understanding of this loop

print("Starting training...")
for epoch in range(epochs):
    running_loss = 0.0
    
    # Iterating through the batches of images and labels
    for i, (images, labels) in enumerate(train_loader):
        # Move data to the same device as the model (MPS in our case)
        images = images.to(device)
        labels = labels.to(device)
        
        # Step 1: Zero the parameter gradients
        # PyTorch accumulates gradients, so we must clear them every batch
        optimizer.zero_grad()
         # idk what this^^ mean, ask chatgpt
        
        # Step 2: Forward pass (make predictions)
        outputs = model(images)
        
        # Step 3: Calculate the loss (how wrong the predictions were)
        loss = criterion(outputs, labels)
        
        # Step 4: Backward pass (calculate the gradients)
        loss.backward()
        
        # Step 5: Optimize (update the weights)
        optimizer.step()
        
        # Keeps track of the loss for monitoring
        running_loss += loss.item()
        
        # prints an update every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training complete!")


# 4. Saving the Model
torch.save(model.state_dict(), 'mnist_cnn.pth') 
print("Model weights saved to 'mnist_cnn.pth'")

