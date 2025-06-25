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

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, current_epoch_offset=0):
    "训练模型并返回训练历史"
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history_stage = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        epoch_display = epoch + 1 + current_epoch_offset
        print(f'\nEpoch {epoch_display}/{num_epochs + current_epoch_offset}')
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch_display}")
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
                history_stage['train_loss'].append(epoch_loss)
                history_stage['train_acc'].append(epoch_acc.item())
            else:
                history_stage['val_loss'].append(epoch_loss)
                history_stage['val_acc'].append(epoch_acc.item())
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy (overall): {best_acc:.4f}")
    time_elapsed = time.time() - since
    print(f'\nTraining stage complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, history_stage, best_model_wts, best_acc

def main_gradual_unfreezing():
    "主流程：分阶段解冻EfficientNet-B4进行训练和评估"
    print("--- Running Script with Gradual Unfreezing (2 Stages) ---")
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'Rock Data'
    model_name_suffix = '_gradual_unfreeze'
    model_save_path = f'rock_classifier_efficientnet_b4{model_name_suffix}.pth'
    confusion_matrix_save_path = f'confusion_matrix_efficientnet_b4{model_name_suffix}.png'
    input_size = 380
    batch_size = 16
    num_epochs_stage1 = 20
    lr_stage1 = 0.001
    num_epochs_stage2 = 30
    lr_stage2 = 0.0001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(int(input_size/0.875)), transforms.CenterCrop(input_size),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(input_size/0.875)), transforms.CenterCrop(input_size),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4 if device.type == 'cuda' else 0) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    if dataset_sizes['train'] == 0:
        print("Error: Training dataset is empty.")
        return
    print(f"Num classes: {num_classes}, Class names: {class_names}")
    model_ft = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    print("\n--- Starting Stage 1: Training Classifier Head ---")
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.classifier[1].parameters():
        param.requires_grad = True
    optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=lr_stage1)
    scheduler_stage1 = optim.lr_scheduler.StepLR(optimizer_stage1, step_size=4, gamma=0.1)
    model_ft, history_s1, best_model_wts_overall, best_acc_overall = train_model(
        model_ft, criterion, optimizer_stage1, scheduler_stage1, dataloaders, dataset_sizes, device, num_epochs=num_epochs_stage1
    )
    print("\n--- Starting Stage 2: Fine-tuning All Layers ---")
    for param in model_ft.parameters():
        param.requires_grad = True
    optimizer_stage2 = optim.AdamW(model_ft.parameters(), lr=lr_stage2)
    scheduler_stage2 = optim.lr_scheduler.StepLR(optimizer_stage2, step_size=5, gamma=0.1)
    model_ft.load_state_dict(best_model_wts_overall)
    model_ft, history_s2, best_model_wts_overall, best_acc_overall = train_model(
        model_ft, criterion, optimizer_stage2, scheduler_stage2, dataloaders, dataset_sizes, device,
        num_epochs=num_epochs_stage2, current_epoch_offset=num_epochs_stage1
    )
    model_ft.load_state_dict(best_model_wts_overall)
    print(f"Overall best validation Acc: {best_acc_overall:4f}")
    combined_history = {
        'train_acc': history_s1['train_acc'] + history_s2['train_acc'],
        'val_acc': history_s1['val_acc'] + history_s2['val_acc'],
        'train_loss': history_s1['train_loss'] + history_s2['train_loss'],
        'val_loss': history_s1['val_loss'] + history_s2['val_loss'],
    }
    plot_training_history(combined_history, num_epochs_stage1 + num_epochs_stage2, filename_suffix=model_name_suffix)
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
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(confusion_matrix_save_path)

if __name__ == '__main__':
    "主程序入口，设置随机种子并运行主流程"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main_gradual_unfreezing()