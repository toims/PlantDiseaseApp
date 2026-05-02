import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

def apply_noise(image_tensor, std):
    """Adds Gaussian noise to a normalized image tensor."""
    noise = torch.randn_like(image_tensor) * std
    noisy_tensor = image_tensor + noise
    return noisy_tensor

def evaluate_robustness(model_path, test_dir, class_names, intensities, mode='noise'):
    """Evaluates model accuracy under different levels of noise or blur."""
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model Architecture and Weights
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Setup Dataloader
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    results = {}

    print(f"Starting Robustness Evaluation (Mode: {mode}) for {model_path}...")

    for intensity in intensities:
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Apply interference based on mode
                if mode == 'noise':
                    inputs = apply_noise(inputs, intensity)
                elif mode == 'blur':
                    # intensity acts as kernel size for GaussianBlur
                    blur_transform = transforms.GaussianBlur(kernel_size=(intensity, intensity), sigma=(0.1, 2.0))
                    inputs = blur_transform(inputs)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        accuracy = running_corrects.double() / total_samples
        results[intensity] = accuracy.item()
        print(f"  Intensity {intensity}: Accuracy = {accuracy.item():.4f}")

    return results
if __name__ == '__main__':
    print("Functions apply_noise and evaluate_robustness are defined.")