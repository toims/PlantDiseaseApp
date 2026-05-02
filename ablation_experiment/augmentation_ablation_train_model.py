from ablation_train_model import train_model_ablation

augmentation_history = train_model_ablation(
    use_augmentation=True,
    use_class_weights=False,
    use_pretrained=True,
    model_save_path='plant_disease_mobilenet_augmentation.pth'
)
print("Model with augmentation training complete. History stored in `augmentation_history`.")