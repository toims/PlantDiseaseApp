import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
DATA_DIR = './data'
MODEL_SAVE_PATH = 'plant_disease_mobilenet.pth'
# 遷移學習開關
USE_PRETRAINED = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model():
    # 2. 增强版數據預處理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    class_counts = [len(os.listdir(os.path.join(DATA_DIR, 'train', c))) for c in class_names]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = weights / weights.sum()
    samples_weights = samples_weights.to(DEVICE)

    model = models.mobilenet_v3_small(weights='DEFAULT' if USE_PRETRAINED else None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=samples_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 用於記錄繪圖數據
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"開始訓練... 類別數: {num_classes}")
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            # 儲存歷史記錄
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"訓練完成! 最佳驗證準確率: {best_acc:.4f}")
    with open("./classes/classes.txt", "w") as f:
        f.write("\n".join(class_names))
    return history

if __name__ == '__main__':
    history = train_model()