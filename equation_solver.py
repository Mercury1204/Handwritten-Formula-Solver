import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Your 14-Class Brain
class MathCNN(nn.Module):
    def __init__(self):
        super(MathCNN, self).__init__()
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = MathCNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', weights_only=True)) 
model.eval()

class_mapping = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 
    5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'+', 11:'-', 12:'/', 13:'*'
}


# Aspect Ratio Preserving Resize
def pad_and_resize(img_crop, target_size=28, ink_size=20):
    h, w = img_crop.shape[:2]
    ratio = ink_size / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    
    resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    padded_img = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

# 2. The Vision & Parsing Pipeline
def solve_equation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}. Check the filename!")
        return

    if img[0, 0] > 127: 
        img = cv2.bitwise_not(img)

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    """    # NEW: Thin out the white ink so thick lines look like HASYv2 fine pens
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1) """

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(c) for c in contours]
    rects = [r for r in rects if r[2] > 5 and r[3] > 5] 
    rects = sorted(rects, key=lambda r: r[0])

    equation_string = ""
    processed_boxes = []

    # Beautifully simple loop now that the heuristic is gone
    for x, y, w, h in rects:
        padding = 5
        crop = thresh[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]
        
        crop_resized = pad_and_resize(crop)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(crop_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output.data, 1)
            char = class_mapping[predicted_class.item()]

            """# NEW: Pop up a window to see exactly what the CNN is looking at!
            plt.imshow(crop_resized, cmap='gray')
            plt.title(f"CNN is looking at this. It guessed: {char}")
            plt.show() """
 
        equation_string += char
        processed_boxes.append((x, y, w, h, char))

    # 3. Execution & Visualization
    print(f"\nExtracted Equation: {equation_string}")
    
    try:
        # Evaluate the string directly!
        answer = eval(equation_string)
        print(f"Solved Answer: {answer}")
    except Exception as e:
        answer = "Error"
        print("Could not compute the math. Did the CNN misread a character?")

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, char) in processed_boxes:
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(color_img, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    plt.imshow(color_img)
    plt.title(f"{equation_string} = {answer}")
    plt.axis('off')
    plt.show()

# --- RUN IT ---
solve_equation('equation2.png')