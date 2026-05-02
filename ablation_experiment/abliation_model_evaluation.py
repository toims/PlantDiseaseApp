import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
import numpy as np
import os
from ablation_train_model import BATCH_SIZE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_model_on_test_set(model_path, num_classes, class_names, device, data_dir='./data'):
    # Prepare data transforms for the test set (same as validation)
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate per-class recall
    recalls = recall_score(all_labels, all_preds, average=None, labels=range(num_classes))

    # Get overall accuracy
    overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Evaluated model: {model_path}")
    print(f"Overall Test Accuracy: {overall_accuracy:.4f}")
    print("Per-class Recall:")
    recall_dict = {}
    for i, class_name in enumerate(class_names):
        recall_dict[class_name] = recalls[i]
        print(f"  {class_name}: {recalls[i]:.4f}")

    return overall_accuracy, recall_dict



print(f"Using device: {DEVICE}")

# 从保存的文件中加载类别名称
if os.path.exists('./classes/classes.txt'):
    with open('./classes/classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")
else:
    print("Error: classes.txt not found. Please ensure the training step has completed.")

# 执行评估
if 'class_names' in locals():
    baseline_overall_accuracy, baseline_recall = evaluate_model_on_test_set(
        '../pth/ablation_experiment/plant_disease_mobilenet_baseline.pth',
        num_classes,
        class_names,
        DEVICE
    )

    augmentation_overall_accuracy, augmentation_recall = evaluate_model_on_test_set(
        '../pth/ablation_experiment/plant_disease_mobilenet_augmentation.pth',
        num_classes,
        class_names,
        DEVICE
    )

    class_weights_overall_accuracy, class_weights_recall = evaluate_model_on_test_set(
        '../pth/ablation_experiment/plant_disease_mobilenet_class_weights.pth',
        num_classes,
        class_names,
        DEVICE
    )

    no_pretrained_overall_accuracy, no_pretrained_recall = evaluate_model_on_test_set(
        '../pth/ablation_experiment/plant_disease_mobilenet_no_pretrained.pth',
        num_classes,
        class_names,
        DEVICE
    )

    print("\n--- Ablation Study Quantitative Results ---")
    print(f"Baseline Model: Overall Accuracy = {baseline_overall_accuracy:.4f}")
    print(f"Augmentation Model: Overall Accuracy = {augmentation_overall_accuracy:.4f}")
    print(f"Class Weights Model: Overall Accuracy = {class_weights_overall_accuracy:.4f}")
    print(f"No Pretrained Model: Overall Accuracy = {no_pretrained_overall_accuracy:.4f}")

    # Example of comparing a minority class's recall
    minority_class = 'Potato___healthy'
    print(f"\nRecall for '{minority_class}':")
    print(f"  Baseline: {baseline_recall.get(minority_class, 'N/A'):.4f}")
    print(f"  Augmentation: {augmentation_recall.get(minority_class, 'N/A'):.4f}")
    print(f"  Class Weights: {class_weights_recall.get(minority_class, 'N/A'):.4f}")
    print(f"  No Pretrained: {no_pretrained_recall.get(minority_class, 'N/A'):.4f}")