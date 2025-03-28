# TransparencyGradCAM4AnimatedGIF

A tool for applying Grad-CAM visualization to animated GIFs, highlighting regions of interest with transparency effects.

## Overview

This project implements Gradient-weighted Class Activation Mapping (Grad-CAM) for animated GIF files. The tool analyzes each frame of a GIF using a pre-trained deep learning model and creates a new GIF where areas of interest are highlighted while the rest is made transparent.

## Features

- Applies Grad-CAM visualization to each frame of an animated GIF
- Creates smooth transparency masks based on activation heatmaps
- Supports multiple target classes (specifically designed for dog detection)
- Compatible with various devices (CPU, CUDA, Apple MPS)

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- NumPy
- OpenCV
- PIL (Pillow)
- imageio

## Installation

1. Clone this repository:
```bash
git clone https://github.com/bemoregt/TransparencyGradCAM4AnimatedGIF.git
cd TransparencyGradCAM4AnimatedGIF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from gradcam import process_gif

# Provide input and output paths
input_gif = "path/to/input.gif"
output_gif = "path/to/output.gif"

# Choose device - "cuda" for NVIDIA GPU, "mps" for Apple Silicon, "cpu" for CPU
device = "mps"  # Change as needed

# Process the GIF
process_gif(input_gif, output_gif, device)
```

Or run the example script:
```bash
python example.py
```

## How It Works

1. **Initialization**: The tool loads a pre-trained ResNet18 model and selects the last convolutional layer for Grad-CAM visualization.

2. **Frame Processing**: Each frame of the input GIF is:
   - Preprocessed and resized to 320x240
   - Fed through the neural network
   - Analyzed with Grad-CAM to generate activation heatmaps for dog classes
   - Converted to a smooth transparency mask

3. **Mask Application**: The transparency mask is applied to the original frame, keeping only the areas of interest visible.

4. **Output Generation**: The processed frames are combined into a new animated GIF.

## Grad-CAM Method

Grad-CAM uses the gradients of the target class flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept. This implementation:

1. Forwards an image through the model to obtain class predictions
2. Computes gradients of the score for the target class with respect to feature maps
3. Pools the gradients to obtain importance weights for each channel
4. Computes a weighted combination of activation maps
5. Applies ReLU to highlight features that have a positive influence on the class of interest

## Customization

- Adjust `low_threshold` and `high_threshold` in `create_smooth_mask()` to control the transparency effect
- Modify the target classes (currently set to ImageNet dog classes 151-268)
- Change the resolution by modifying the resize parameters

## License

MIT

## Author

bemoregt
