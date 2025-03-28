import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from imageio import mimread, mimsave

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_image, target_classes):
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        combined_cam = torch.zeros(self.activations.shape[2:]).to(input_image.device)
        for target_class in target_classes:
            target = model_output[0, target_class]
            target.backward(retain_graph=True)
            
            gradients = self.gradients[0].cpu().data.numpy()
            activations = self.activations[0].cpu().data.numpy()
            
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
            
            cam = np.maximum(cam, 0)
            combined_cam += torch.from_numpy(cam).to(input_image.device)
        
        combined_cam = combined_cam.cpu().numpy()
        combined_cam = cv2.resize(combined_cam, (320, 240))
        combined_cam = combined_cam - np.min(combined_cam)
        combined_cam = combined_cam / np.max(combined_cam)
        return combined_cam

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def create_smooth_mask(heatmap, low_threshold=0.4, high_threshold=0.7):
    heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    mask = np.clip((heatmap_normalized - low_threshold) / (high_threshold - low_threshold), 0, 1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask

def apply_gradcam_mask(image, mask):
    mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    result = image * mask_3channel + np.zeros_like(image) * (1 - mask_3channel)
    return result.astype(np.uint8)

def process_gif(input_gif, output_gif, device_name="mps"):
    device = torch.device(device_name)
    model = models.resnet18(pretrained=True).to(device)
    model.eval()
    
    grad_cam = GradCAM(model, model.layer4[-1])
    
    frames = mimread(input_gif)
    processed_frames = []
    
    dog_classes = range(151, 269)  # ImageNet dog classes (151-268)
    
    for frame in frames:
        pil_image = Image.fromarray(frame).convert('RGB')
        input_tensor = preprocess_image(pil_image).to(device)
        
        combined_cam = grad_cam.generate_cam(input_tensor, dog_classes)
        
        frame_resized = cv2.resize(np.array(pil_image), (320, 240))
        
        mask = create_smooth_mask(combined_cam, low_threshold=0.4, high_threshold=0.7)
        result_frame = apply_gradcam_mask(frame_resized, mask)
        
        processed_frames.append(result_frame)
    
    mimsave(output_gif, processed_frames, format='GIF', duration=100, loop=0)

# Example usage
if __name__ == "__main__":
    input_gif = "input.gif"
    output_gif = 'output.gif'
    
    # Choose device - 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' for CPU
    device = "mps"  # Change as needed
    
    process_gif(input_gif, output_gif, device)