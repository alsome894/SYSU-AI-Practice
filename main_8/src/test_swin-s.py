import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_file = Path(__file__).resolve()
project_root = current_file.parent
data_dir = project_root.parent.parent / 'Rock Data'
save_dir = project_root / 'output_swin_s'

# 数据集参数
DATA_DIR = data_dir  # 划分后的数据目录
BATCH_SIZE = 32
IMG_SIZE = 224  # Swin-T 默认输入尺寸也是 224

# 保存路径
SAVE_DIR = save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
def load_datasets(data_dir):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
    return train_dataset, val_dataset, test_dataset

# 创建 DataLoader
def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader

# 初始化 Swin-T 模型
def initialize_model(num_classes):
    # 从 torchvision 加载 Swin-T
    model = models.swin_s(pretrained=False)  
    # Swin 的分类头在 model.head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model.to(device)

# 计算指标
def calculate_metrics(outputs, labels, class_names):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = np.argmax(outputs, axis=1)
    
    # 生成混淆矩阵和分类报告
    cm = confusion_matrix(labels, preds)
    cr = classification_report(labels, preds, target_names=class_names, digits=4)
    
    return cm, cr

# 可视化混淆矩阵
def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

# 模型评估
def evaluate_model(model, data_loader, criterion, class_names):
    model.eval()
    running_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)

    total_loss = running_loss / len(data_loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    cm, cr = calculate_metrics(all_outputs, all_labels, class_names)
    return total_loss, cm, cr

# 推理主函数
def main():
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_datasets(DATA_DIR)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")

    # 初始化并加载模型权重
    model = initialize_model(len(train_dataset.classes))
    #checkpoint_path = os.path.join(SAVE_DIR, 'best_model_81.pth')
    checkpoint_path = os.path.join(SAVE_DIR, 'best_model_82.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 定义损失
    criterion = nn.CrossEntropyLoss()

    # 在测试集上评估
    test_loss, test_cm, test_cr = evaluate_model(model, test_loader, criterion, train_dataset.classes)
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print("\nConfusion Matrix:")
    print(test_cm)
    print("\nClassification Report:")
    print(test_cr)
    
    # 可视化并保存混淆矩阵
    cm_save_path = os.path.join(SAVE_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(test_cm, train_dataset.classes, cm_save_path)

if __name__ == "__main__":
    main()
