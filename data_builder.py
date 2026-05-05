
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageOps

# A custom transform to invert colors (black ink to white ink)
class InvertColor(object):
    def __call__(self, img):
        return ImageOps.invert(img)

def get_dataloaders(batch_size=64):
    print("Loading perfectly balanced dataset...")
    
    # 1. Standardize everything to look like MNIST (28x28, white ink on black background)
    transform = transforms.Compose([
        transforms.Grayscale(), 
        InvertColor(),          
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. Load the perfectly balanced dataset from a single folder
    # IMPORTANT: Make sure all 14 folders (0-9, +, -, div, multiply) are inside a folder named 'symbols_dataset'
    dataset = ImageFolder(
        root='./symbols_dataset', 
        transform=transform
    )

    # Print how PyTorch sorted the folders so we can update the dictionary later
    print("\n--- NEW CLASS MAPPING ---")
    for class_name, class_index in dataset.class_to_idx.items():
        print(f"Class {class_index}: '{class_name}'")
    print("-------------------------\n")

    # 3. Create the DataLoader
    # Because the data is naturally balanced, we can go back to simple random shuffling!
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True 
    )
    
    print(f"Successfully loaded {len(dataset)} total images.")
    return loader