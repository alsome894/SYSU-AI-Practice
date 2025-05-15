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

from pathlib import Path
def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")

    # 配置参数
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'Rock Data'
    model_save_path = 'rock_classifier_model.pth'
    confusion_matrix_save_path = 'confusion_matrix.png'
    
    input_size = 224  # 设置输入尺寸为224x224，这是ResNet的常用输入尺寸
    batch_size = 32   # 批处理大小，可根据GPU内存调整
    num_epochs = 25   # 训练轮数，可根据需要增加
    learning_rate = 0.001
    momentum = 0.9
    lr_scheduler_step_size = 7
    lr_scheduler_gamma = 0.1

    # 设置设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据增强和归一化设置
    # 训练集使用数据增强，验证集和测试集只做归一化处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(),         # 随机水平翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.RandomRotation(15),            # 随机旋转
            transforms.ToTensor(),                    # 转换为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet统计数据归一化
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size + 32),       # 调整大小略大于输入尺寸
            transforms.CenterCrop(input_size),        # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # 创建训练、验证和测试数据集
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'valid', 'test']}
    except FileNotFoundError:
        print(f"Error: Data directory '{data_dir}' not found. Please ensure the directory exists and is structured correctly.")
        print("Expected structure: Rock Data/{train, valid, test}/{class_folders}")
        return

    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4 if device.type == 'cuda' else 0)
                   for x in ['train', 'valid', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: Train: {dataset_sizes['train']}, Valid: {dataset_sizes['valid']}, Test: {dataset_sizes['test']}")

    if dataset_sizes['train'] == 0:
        print("Error: Training dataset is empty. Please check your 'Rock Data/train' directory.")
        return

    # 初始化模型（ResNet50预训练模型）
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结所有网络参数，只训练最后的全连接层
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # 修改最后的全连接层以适应我们的分类任务
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    print("Model Initialized (ResNet50 with new classifier head).")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器 - 仅优化最后一层的参数
    optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=learning_rate, momentum=momentum)
    
    # 学习率调度器
    # 每7个epoch将学习率降低为原来的0.1倍
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    # 训练和评估模型
    print("Starting training...")
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs)

    # 绘制训练历史（准确率和损失）
    plot_training_history(history, num_epochs)

    # 在测试集上评估模型
    print("\nEvaluating on Test Set...")
    y_pred_list = []
    y_true_list = []
    model_ft.eval()  # 设置模型为评估模式
    
    running_corrects = 0
    
    with torch.no_grad():  # 禁用梯度计算
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

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(y_true_list, y_pred_list, target_names=class_names, zero_division=0))

    # 混淆矩阵
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_list, y_pred_list)
    print(cm)

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(confusion_matrix_save_path)
    print(f"Confusion matrix saved to {confusion_matrix_save_path}")

    # 保存训练好的模型
    print(f"\nSaving model to {model_save_path}...")
    torch.save(model_ft, model_save_path)  # 保存整个模型
    print("Model saved successfully.")

    print("\n--- 处理完成 ---")
print("要进一步提高准确率，可以考虑以下方案：")
print("1. 微调更多层：在初始训练后，解冻ResNet50的一些早期层，并使用较低的学习率继续训练。")
print("2. 使用不同的预训练架构：EfficientNet（例如B0-B7）或Vision Transformer (ViT)可能提供更好的性能。")
print("3. 超参数调优：尝试不同的学习率、批处理大小、优化器（例如AdamW）和学习率调度器。")
print("4. 更广泛的数据增强：尽管Roboflow提供了增强，但使用PyTorch定制的增强可以进行更精细的调整。")
print("5. 更长时间的训练：如果模型仍在学习（验证损失在下降），增加训练轮数。实施早停以防止过拟合。")
print("6. 使用更高分辨率训练：由于原始图像是640x640，使用384x384或512x512的输入（如果VRAM允许）可能捕获更多细节，不过这可能需要调整模型或使用专为高分辨率设计的模型。")



def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    """
    训练模型函数
    
    参数:
        model: 要训练的模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        dataloaders: 数据加载器字典
        dataset_sizes: 数据集大小字典
        device: 计算设备(GPU/CPU)
        num_epochs: 训练轮数
    
    返回:
        model: 训练好的模型
        history: 训练历史记录
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            # 使用tqdm显示进度条
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 只在训练阶段跟踪梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播+优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计信息
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新tqdm进度条
                if phase == 'train':
                    progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/inputs.size(0))

            if phase == 'train':
                scheduler.step()  # 更新学习率

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:  # phase == 'valid'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())


            # 如果验证准确率提高，则保存模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history, num_epochs):
    """
    绘制训练历史图表
    
    参数:
        history: 包含训练和验证损失/准确率的字典
        num_epochs: 训练轮数
    """
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
    
    plt.suptitle('Model Training History')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")

if __name__ == '__main__':
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    main()
