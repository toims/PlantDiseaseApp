import matplotlib.pyplot as plt

from baseline_ablation_train_model import baseline_history
from augmentation_ablation_train_model import augmentation_history
from class_weights_ablation_train_model import class_weights_history
from no_pretrained_ablation_train_model import no_pretrained_history


def plot_ablation_history_comparison(histories, labels):
    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        label = labels[i]
        epochs_range = range(1, len(history['train_acc']) + 1)
        plt.plot(epochs_range, history['train_acc'], label=f'{label} Training Accuracy')
        plt.plot(epochs_range, history['val_acc'], label=f'{label} Validation Accuracy', linestyle='--')
    plt.title('Training and Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        label = labels[i]
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs_range, history['train_loss'], label=f'{label} Training Loss')
        plt.plot(epochs_range, history['val_loss'], label=f'{label} Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Defined plot_ablation_history_comparison function.")
    histories_to_compare = [
    baseline_history,
    augmentation_history,
    class_weights_history,
    no_pretrained_history
    ]

    labels_for_comparison = [
        'Baseline (No Aug, No CW, Pretrained)',
        '+ Augmentation (No CW, Pretrained)',
        '+ Class Weights (Aug, Pretrained)',
        'No Pretrained (Aug, CW)'
    ]

    plot_ablation_history_comparison(histories_to_compare, labels_for_comparison)
    print("Generated comparison plots for ablation study.")