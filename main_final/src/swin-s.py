import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 自动降低学习率的调度器
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torchvision.models import swin_s, Swin_S_Weights  # 导入Swin-S模型和预训练权重
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 评估指标
import numpy as np
from tqdm import tqdm  # 进度条显示
import pandas as pd

from pathlib import Path

# 设置随机种子保证可重复性
torch.manual_seed(42)  # PyTorch的随机种子
np.random.seed(42)  # NumPy的随机种子

# 设备配置 - 优先使用GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_file = Path(__file__).resolve()
project_root = current_file.parent
data_dir = project_root.parent.parent / 'Rock Data'
save_dir = project_root / 'output_swin_s'

# 数据集参数
DATA_DIR = data_dir  # 数据集所在目录
BATCH_SIZE = 48  # 每批处理的图像数量
IMG_SIZE = 224  # 输入图像尺寸，Swin Transformer的标准输入尺寸

# 训练参数
NUM_EPOCHS = 30  # 训练轮数
LEARNING_RATE = 1e-4  # 初始学习率
WEIGHT_DECAY = 1e-4  # 权重衰减系数，用于L2正则化防止过拟合
PATIENCE = 5  # 学习率调度器的耐心值，连续5个epoch验证损失不改善则降低学习率

# 模型输出保存路径
SAVE_DIR = save_dir
os.makedirs(SAVE_DIR, exist_ok=True)  # 创建输出目录，如果已存在则不报错

# 数据预处理和增强
# 训练集使用数据增强提高模型泛化能力
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转±15度
    #transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # 随机裁剪并调整大小（已注释）
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动（已注释）
    transforms.ToTensor(),  # 转换为张量 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化参数
])

# 验证集和测试集只需调整大小和标准化，不需要数据增强
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_datasets(data_dir):
    """加载训练、验证和测试数据集
    
    Args:
        data_dir: 包含train、val、test三个子目录的数据根目录
        
    Returns:
        三个数据集对象：训练集、验证集和测试集
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'valid'),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=val_test_transform
    )
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """创建数据加载器
    
    Args:
        train_dataset, val_dataset, test_dataset: 三个数据集对象
        
    Returns:
        三个DataLoader对象：训练、验证和测试
    """
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
        # shuffle=True确保每个epoch的数据顺序不同
        # num_workers=8使用多进程加速数据加载
        # pin_memory=True使用固定内存加速GPU训练
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
        # 验证和测试集不需要打乱顺序
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
    )
    return train_loader, val_loader, test_loader

def initialize_model(num_classes):
    """初始化并返回Swin-S模型
    
    Args:
        num_classes: 分类数量
        
    Returns:
        配置好的模型，已转移到指定设备(CPU/GPU)
    """
    # 使用预训练的Swin-S权重
    weights = Swin_S_Weights.IMAGENET1K_V1  # 使用在ImageNet上预训练的权重
    model = swin_s(weights=weights)  # 加载带预训练权重的模型
    
    # 修改最后的全连接层以匹配我们的分类任务
    num_ftrs = model.head.in_features  # 获取特征数量
    model.head = nn.Linear(num_ftrs, num_classes)  # 替换分类头
    
    return model.to(device)  # 将模型移至指定设备(GPU/CPU)

def calculate_metrics(outputs, labels, num_classes):
    """计算模型性能指标
    
    Args:
        outputs: 模型输出的预测结果
        labels: 真实标签
        num_classes: 类别数量
        
    Returns:
        包含各项指标的字典
    """
    metrics = {}
    
    # 确保输入是NumPy数组，便于使用sklearn计算指标
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 从softmax输出获取预测类别
    preds = np.argmax(outputs, axis=1)
    
    # 计算各项分类指标
    metrics['accuracy'] = accuracy_score(labels, preds)  # 准确率
    metrics['f1'] = f1_score(labels, preds, average='weighted')  # F1分数(weighted考虑类别不平衡)
    metrics['precision'] = precision_score(labels, preds, average='weighted')  # 精确率
    metrics['recall'] = recall_score(labels, preds, average='weighted')  # 召回率
    
    return metrics

def train_model(model, train_loader, val_loader, num_classes, optimizer, scheduler, num_epochs):
    """训练模型的主函数
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 类别数量
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        
    Returns:
        包含训练过程中各项指标历史的字典
    """
    # 用于记录训练过程中的各项指标
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
    }
    criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失函数
    best_val_loss = float('inf')  # 记录最佳验证损失
    best_acc = 0  # 记录最佳验证准确率
    
    # 开始训练循环
    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式（启用Dropout、BatchNorm等）
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        # 训练一个epoch
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = inputs.to(device)  # 将输入数据移至设备
            labels = labels.to(device)  # 将标签移至设备
            
            optimizer.zero_grad()  # 清除之前的梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新权重
            
            # 累积统计数据
            running_loss += loss.item() * inputs.size(0)  # 计入批次总损失
            all_outputs.append(outputs)  # 收集所有输出
            all_labels.append(labels)  # 收集所有标签
        
        # 计算训练集平均损失
        train_loss = running_loss / len(train_loader.dataset)
        # 合并整个epoch的预测和标签用于计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        train_metrics = calculate_metrics(all_outputs, all_labels, num_classes)
        
        # 在验证集上评估模型
        val_loss, val_metrics = evaluate_model(model, val_loader, num_classes, criterion)
        # 根据验证损失调整学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 记录所有指标
        for metric_name in train_metrics:
            history[f'train_{metric_name}'].append(train_metrics[metric_name])
            history[f'val_{metric_name}'].append(val_metrics[metric_name])
        
        # 打印当前epoch的训练结果
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Train Precision: {train_metrics['precision']:.4f} | Val Precision: {val_metrics['precision']:.4f}")
        print(f"Train Recall: {train_metrics['recall']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print("-" * 60)
        
        # 如果当前模型性能最好，则保存模型
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,'best_model.pth'))
            print(f"Saved new best model with val_loss: {val_loss:.4f}")
    
    return history

def evaluate_model(model, data_loader, num_classes, criterion):
    """在给定数据集上评估模型性能
    
    Args:
        model: 待评估的模型
        data_loader: 数据加载器
        num_classes: 类别数量
        criterion: 损失函数
        
    Returns:
        平均损失和性能指标字典
    """
    model.eval()  # 切换到评估模式（禁用Dropout等）
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():  # 禁用梯度计算提高推理速度和减少内存使用
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            
            # 累积统计数据
            running_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    # 计算平均损失和指标
    total_loss = running_loss / len(data_loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_outputs, all_labels, num_classes)
    
    return total_loss, metrics

def plot_training_history(history, num_epochs):
    """可视化训练过程中的各项指标变化
    
    Args:
        history: 包含训练指标历史的字典
        num_epochs: 训练轮数
    """
    plt.figure(figsize=(18, 12))  # 创建大图
    
    # 绘制损失曲线
    plt.subplot(2, 3, 1)  # 2行3列的第1个子图
    plt.plot(range(1, num_epochs+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs+1), history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(2, 3, 2)  # 2行3列的第2个子图
    plt.plot(range(1, num_epochs+1), history['train_accuracy'], label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # 绘制F1分数曲线
    plt.subplot(2, 3, 3)  # 2行3列的第3个子图
    plt.plot(range(1, num_epochs+1), history['train_f1'], label='Train F1')
    plt.plot(range(1, num_epochs+1), history['val_f1'], label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    
    # 绘制精确率曲线
    plt.subplot(2, 3, 4)  # 2行3列的第4个子图
    plt.plot(range(1, num_epochs+1), history['train_precision'], label='Train Precision')
    plt.plot(range(1, num_epochs+1), history['val_precision'], label='Val Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    
    # 绘制召回率曲线
    plt.subplot(2, 3, 5)  # 2行3列的第5个子图
    plt.plot(range(1, num_epochs+1), history['train_recall'], label='Train Recall')
    plt.plot(range(1, num_epochs+1), history['val_recall'], label='Val Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()
    
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(os.path.join(SAVE_DIR,'training_metrics.png'))  # 保存图像
    plt.close()  # 关闭图像，释放内存

def main():
    """主函数：加载数据，训练模型，评估结果"""
    # 加载数据集
    train_dataset, val_dataset, test_dataset = load_datasets(DATA_DIR)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    
    # 保存类别映射信息
    class_to_idx = train_dataset.class_to_idx  # 获取类别名称到索引的映射
    with open(os.path.join(SAVE_DIR, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f)  # 保存为JSON文件，便于后续使用
    print(f"Saved class index mapping to {os.path.join(SAVE_DIR, 'class_to_idx.json')}")
    
    # 打印数据集信息
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    
    # 初始化模型、优化器和学习率调度器
    model = initialize_model(len(train_dataset.classes))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # AdamW优化器
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',  # 监控验证损失最小化
        factor=0.5,  # 学习率衰减因子，每次降低为原来的一半
        patience=PATIENCE,  # 5个epoch损失不改善则降低学习率
        #verbose=True  # 打印学习率变化信息
    )
    
    # 训练模型
    history = train_model(
        model, train_loader, val_loader, len(train_dataset.classes), optimizer, scheduler, NUM_EPOCHS
    )
    
    # 可视化训练过程并保存训练历史
    plot_training_history(history, NUM_EPOCHS)
    history_df = pd.DataFrame(history)  # 转换为DataFrame
    history_df.to_csv(os.path.join(SAVE_DIR,'training_history.csv'), index=False)  # 保存为CSV
    
    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR,'best_model.pth')))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = evaluate_model(model, test_loader, len(train_dataset.classes), criterion)
    
    # 打印测试结果
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

# 程序入口点
if __name__ == "__main__":
    main()