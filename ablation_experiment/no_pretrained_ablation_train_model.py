from ablation_train_model import train_model_ablation

no_pretrained_history = train_model_ablation(
    use_augmentation=True,
    use_class_weights=True,
    use_pretrained=False,
    model_save_path='plant_disease_mobilenet_no_pretrained.pth'
)
print("Model without pretrained weights training complete. History stored in `no_pretrained_history`.")