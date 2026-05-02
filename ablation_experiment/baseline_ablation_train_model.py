from ablation_train_model import train_model_ablation

baseline_history = train_model_ablation(
    use_augmentation=False,
    use_class_weights=False,
    use_pretrained=True,
    model_save_path='plant_disease_mobilenet_baseline.pth'
)
print("Baseline model training complete. History stored in `baseline_history`.")