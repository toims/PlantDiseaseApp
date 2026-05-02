from ablation_train_model import train_model_ablation

class_weights_history = train_model_ablation(
    use_augmentation=True,
    use_class_weights=True,
    use_pretrained=True,
    model_save_path='plant_disease_mobilenet_class_weights.pth'
)
print("Model with augmentation and class weights training complete. History stored in `class_weights_history`.")