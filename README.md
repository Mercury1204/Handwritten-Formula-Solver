# Handwritten Formula Solver

This project implements a multimodal machine learning pipeline designed to parse, recognize, and evaluate images of handwritten mathematical equations. It leverages **OpenCV** for computer vision segmentation and a custom **PyTorch** Convolutional Neural Network (CNN) for character classification.

## Features

* **Multimodal Pipeline:** Seamlessly integrates computer vision (OpenCV) for bounding box extraction with deep learning (PyTorch) for image classification.
* **Custom CNN Architecture:** A PyTorch-based neural network trained from scratch to classify 14 distinct symbols (Digits 0-9, `+`, `-`, `*`, `/`).
* **Intelligent Vision Parsing:** Features a custom aspect-ratio preserving bounding box algorithm that prevents spatial distortion and groups disconnected multi-stroke symbols (like division or equals signs).
* **Balanced Data Engineering:** Merges the official MNIST dataset with the Kaggle Handwritten Math Symbols dataset. Utilizes a custom `InvertColor` PIL transform to normalize datasets to a uniform white-ink-on-black-canvas format.
* **Live Evaluation:** Uses Python's native `eval()` function to calculate and output the mathematical result of the drawn equation.

## Installation

**Prerequisites:** You must have Python 3.8+ installed.

1.  **Clone the repository:**
```bash
    git clone [https://github.com/yourusername/handwritten-formula-solver.git](https://github.com/yourusername/handwritten-formula-solver.git)
    cd handwritten-formula-solver
```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Requirements:**
    Install the necessary packages using `pip`:
    ```bash
    pip install torch torchvision opencv-python numpy matplotlib Pillow
    ```

## Dataset Preparation

This project uses a hybrid dataset approach.

1.  Create a folder named `symbols_dataset` in your project root.
2.  Download the **Handwritten Math Symbols Dataset** from Kaggle.
3.  Extract the 0-9 and `+`, `-`, `*` (dot/multiply), and `/` (div) folders into your `symbols_dataset` directory. Ensure the symbols are formatted as 28x28 grayscale PNGs (black ink on white paper).

## Usage

### Training the Model

Run the `main.py` script to train the CNN from scratch. 
```bash
python main.py
```
This will generate a model_cnn.pth file containing the model weights.

### Solving an Equation
1.  Draw an equation (e.g., 8 - 3) on a black canvas using white or light gray ink. Make sure to use a thick brush (approx. 3-5px).
2.  Save the drawing in your project folder (e.g., equation1.png).
3.  Run the solver script:

```bash
python equation_solver.py
```

The script will:

1.  Load your image and run the OpenCV binarization and morphological dilation pipeline.
2.  Draw bounding boxes around the characters, sorting them left-to-right.
3.  Pad and resize the boxes to 28x28 (preserving aspect ratios).
4.  Feed them to the CNN for classification.
5.  Print the extracted string and mathematically evaluate the answer.

## File Structure
1.  main.py: The PyTorch deep learning architecture and training loop.
2.  data_builder.py: Data engineering pipeline handling the ConcatDataset, transforms, color inversions, and Dataloaders.
3.  equation_solver.py: The core OpenCV vision pipeline, bounding box logic, inference step, and math parser.
4.  requirements.txt: List of dependencies.
