import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

# 1. Re-define the Model Architecture so the script knows what to load into
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Setup Device and Load Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Load the trained weights you saved earlier
model.load_state_dict(torch.load('mnist_cnn.pth', weights_only=True))
model.eval() # Lock the weights for testing

# 3. The Vision Pipeline
def predict_digit(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}. Check the filename!")
        return
        
    # MNIST images are white digits on a black background.
    # If you drew a black digit on a white background, we must invert the colors.
    if img[0, 0] > 127: # If the top-left corner is light, assume white background
        img = cv2.bitwise_not(img)
        
    # Resize the image to 28x28 to match what the CNN expects
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Apply the exact same math transformations used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Add a batch dimension [1, 1, 28, 28] and send to your M1 GPU
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # 4. Make the Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1) # Convert raw scores to percentages
        confidence, predicted_class = torch.max(probabilities, 1)
        
    print(f"\n--- Prediction ---")
    print(f"I am {confidence.item() * 100:.2f}% sure this is a {predicted_class.item()}")
    
    # Pop up a window to show you exactly what the CNN saw
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted: {predicted_class.item()} ({confidence.item()*100:.1f}%)")
    plt.axis('off')
    plt.show()

# --- RUN IT ---
# Change 'my_digit.png' to the name of your drawing!
predict_digit('six.png')