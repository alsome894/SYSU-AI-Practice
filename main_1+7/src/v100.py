import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
import time

from pathlib import Path

# --- CBAM Implementation (Convolutional Block Attention Module) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_ca = self.ca(x)
        x = x * x_ca
        x_sa = self.sa(x)
        x = x * x_sa
        return x

# --- Model with EfficientNet-B2 and CBAM ---
class EfficientNetWithCBAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetWithCBAM, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            self.efficientnet = models.efficientnet_b2(weights=None)

        # Number of features before the classifier in efficientnet_b2 is 1408
        num_ftrs = self.efficientnet.classifier[1].in_features # Should be 1408 for B2

        # Add CBAM after the feature extraction layers
        # The output channels of efficientnet_b2.features is 1408
        self.cbam = CBAM(num_ftrs) # CBAM for the features before avgpool

        # Replace the classifier
        # EfficientNet B2 classifier (from source): Sequential(Dropout(p=0.3, inplace=True), Linear(in_features=1408, out_features=1000, bias=True))
        # We'll keep a similar dropout before our new linear layer.
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Dropout for EfficientNet-B2 is typically 0.3
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.cbam(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x

# --- Configuration ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
data_dir = project_root / 'Rock Data'
TRAIN_DIR = os.path.join(data_dir, 'train')
VALID_DIR = os.path.join(data_dir, 'valid')
TEST_DIR = os.path.join(data_dir, 'test')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

BATCH_SIZE = 24 # Adjusted for potentially larger model, monitor GPU memory
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 260 # EfficientNet-B2 expects 260x260

# --- Data Transforms ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
    ]),
    'valid': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32), # e.g., 260 + 32 = 292
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Load Data ---
print("加载数据集...")
try:
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'valid': datasets.ImageFolder(VALID_DIR, data_transforms['valid']),
        'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
    }
except FileNotFoundError:
    print(f"错误: 找不到数据集目录之一 ({TRAIN_DIR}, {VALID_DIR}, {TEST_DIR})。")
    print("请确保 'Rock Data' 文件夹及其 'train'、'valid'、'test' 子文件夹存在并包含图像。")
    exit()

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=min(os.cpu_count(), 4) if DEVICE.type == 'cuda' else 0, pin_memory=True if DEVICE.type == 'cuda' else False)
    for x in ['train', 'valid', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes
NUM_CLASSES = len(class_names)

print(f"类别数量: {NUM_CLASSES}")
print(f"类别名称: {class_names}")
for x in ['train', 'valid', 'test']:
    if dataset_sizes[x] == 0:
        print(f"警告: 数据集 '{x}' 为空。请检查目录: {image_datasets[x].root}")
if dataset_sizes['train'] == 0:
    print("错误: 训练数据集为空，无法继续训练。")
    exit()

# --- Initialize Model, Loss, Optimizer ---
model = EfficientNetWithCBAM(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

# --- Training Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}", unit="batch")
            for inputs, labels in progress_bar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                current_loss = loss.item()
                current_acc = torch.sum(preds == labels.data).item() / inputs.size(0)
                running_loss += current_loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else: # phase == 'valid'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                scheduler.step(epoch_loss) 

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'最佳验证准确率: {best_acc:4f}，将模型权重保存到 best_rock_classifier_efficientnet_b2_cbam.pth')
                torch.save(model.state_dict(), 'best_rock_classifier_efficientnet_b2_cbam.pth')
        print()

    time_elapsed = time.time() - since
    print(f'训练完成，用时 {time_elapsed // 60:.0f} 分 {time_elapsed % 60:.0f} 秒')
    print(f'最佳验证准确率: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, class_names_list):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating on Test Set", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n分类报告:")
    report = classification_report(all_labels, all_preds, target_names=class_names_list, zero_division=0, digits=4)
    print(report)

    print("\n混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    plt.figure(figsize=(max(10, len(class_names_list)), max(8, len(class_names_list) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list, annot_kws={"size": 8})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (EfficientNet-B2 + CBAM)', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_efficientnet_b2_cbam.png')
    print("\n混淆矩阵已保存到 confusion_matrix_efficientnet_b2_cbam.png")
    plt.show()

    return all_labels, all_preds

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.isdir(data_dir):
        print(f"错误: 数据集文件夹 '{data_dir}' 未找到。")
        exit()
    for subdir in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        if not os.path.isdir(subdir):
            print(f"错误: 子文件夹 '{subdir}' 在 '{data_dir}' 中未找到。")
            exit()
        try:
            if not any(os.scandir(subdir)):
                if subdir != TEST_DIR:
                    print(f"警告: 文件夹 '{subdir}' 为空。请检查您的数据集。")
                    if subdir == TRAIN_DIR:
                        print("错误: 训练文件夹为空，无法继续。")
                        exit()
        except Exception as e:
            print(f"访问数据集子文件夹时出错: {e}")
            exit()

    print("开始使用 EfficientNet-B2 + CBAM 进行训练...")
    model_ft, history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
    print("Training complete.")

    torch.save(model_ft, 'rock_classifier_efficientnet_b2_cbam_full_model.pth')
    print("完整模型已保存到 rock_classifier_efficientnet_b2_cbam_full_model.pth")
    print("最佳模型权重（state_dict）已在训练期间保存到 best_rock_classifier_efficientnet_b2_cbam.pth")

    if dataset_sizes['test'] > 0:
        print("\n在测试集上进行评估...")
        true_labels, predicted_labels = evaluate_model(model_ft, dataloaders['test'], DEVICE, class_names)
    else:
        print("\nTest set is empty or not found. Skipping evaluation on test set.")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs (EfficientNet-B2 + CBAM)')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs (EfficientNet-B2 + CBAM)')
    
    plt.tight_layout()
    plt.savefig('training_history_efficientnet_b2_cbam.png')
    print("训练历史图已保存到 training_history_efficientnet_b2_cbam.png")
    plt.show()


