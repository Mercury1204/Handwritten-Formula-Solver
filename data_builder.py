from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import ImageOps

# A custom transform to invert the HASY images (black ink to white ink)
class InvertColor(object):
    def __call__(self, img):
        return ImageOps.invert(img)

def get_combined_dataloaders(batch_size=64):
    
    # 1. Transform for MNIST (just sizing and tensor)
    mnist_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 2. Transform for Symbols (Invert colors, size, tensor)
    symbol_transform = transforms.Compose([
        transforms.Grayscale(), # Ensure it's 1-channel
        InvertColor(),          # Make it white ink on black background
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 3. Load the Datasets
    mnist_train = MNIST(root='./data', train=True, transform=mnist_transform, download=True)
    
    # PyTorch's ImageFolder automatically assigns labels based on the folder names!
    symbol_train = ImageFolder(root='./math_symbol_new', transform=symbol_transform)

    # Note: We need to adjust the symbol labels to start at 10 (since MNIST is 0-9)
    # We can do this with a quick lambda function on the dataset targets
    symbol_train.targets = [label + 10 for label in symbol_train.targets]

    # Combine and Load
    
    # Multiply the symbol dataset by 15 so the CNN sees it 15x more often!

    #combined_dataset = ConcatDataset([mnist_train, symbol_train])
    combined_dataset = ConcatDataset([mnist_train] + [symbol_train] * 20)


    combined_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True)
    
    return combined_loader