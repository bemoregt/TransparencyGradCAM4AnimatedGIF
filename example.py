from gradcam import process_gif

# Usage example
if __name__ == "__main__":
    input_gif = "input.gif"  # Path to your input GIF
    output_gif = "output.gif"  # Path for the output GIF
    
    # Select device:
    # - "cuda" for NVIDIA GPU
    # - "mps" for Apple Silicon
    # - "cpu" for CPU only
    device = "mps"
    
    # Process the GIF
    process_gif(input_gif, output_gif, device)