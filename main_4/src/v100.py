from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_training_history(history, num_epochs, filename_suffix=''):
    "绘制训练和验证准确率与损失曲线"
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.suptitle(f'Model Training History {filename_suffix}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'training_history{filename_suffix}.png')
    print(f"Training history plot saved to training_history{filename_suffix}.png")

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    "训练模型并返回训练历史"
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
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
                    progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/inputs.size(0))
            if phase == 'train' and scheduler:
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}")
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def main_adamw_cosine():
    "主流程：使用AdamW优化器和CosineAnnealingLR调度器训练"
    print("--- Running Script with AdamW and CosineAnnealingLR ---")
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'Rock Data'
    model_name_suffix = '_adamw_cosine'
    model_save_path = f'rock_classifier_efficientnet_b4{model_name_suffix}.pth'
    confusion_matrix_save_path = f'confusion_matrix_efficientnet_b4{model_name_suffix}.png'
    input_size = 380
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4 if device.type == 'cuda' else 0) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    if dataset_sizes['train'] == 0: print("Error: Training dataset is empty."); return
    print(f"Num classes: {num_classes}, Class names: {class_names}")
    model_ft = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    for param in model_ft.parameters(): param.requires_grad = False
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=learning_rate, weight_decay=weight_decay)
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs, eta_min=1e-6)
    print("Starting training with AdamW and CosineAnnealingLR...")
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs)
    plot_training_history(history, num_epochs, filename_suffix=model_name_suffix)
    print("\nEvaluating on Test Set...")
    y_pred_list = []
    y_true_list = []
    model_ft.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred_list.extend(preds.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
            running_corrects += torch.sum(preds == labels.data)
    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'\nTest Accuracy: {test_acc:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_true_list, y_pred_list, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true_list, y_pred_list)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix {model_name_suffix}')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.savefig(confusion_matrix_save_path)
    print(f"Confusion matrix saved to {confusion_matrix_save_path}")
    print(f"\nSaving model to {model_save_path}...")
    torch.save(model_ft, model_save_path)
    print("Model saved successfully.")
    print("--- AdamW and CosineAnnealingLR Script Complete ---")

if __name__ == '__main__':
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main_adamw_cosine()
