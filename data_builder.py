
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageOps

class InvertColor(object):
    def __call__(self, img):
        return ImageOps.invert(img)

def get_dataloaders(batch_size=64):
    print("Loading dataset...")
    
    # standardising all images
    transform = transforms.Compose([
        transforms.Grayscale(), 
        InvertColor(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # loading the dataset from folder 
    dataset = ImageFolder(
        root='./symbols_dataset', 
        transform=transform
    )

    #Creating the DataLoader
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True 
    )
    
    print(f"Successfully loaded {len(dataset)} total images.")
    return loader