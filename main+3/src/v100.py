"""
岩石图像分类优化方案（针对预增强数据集）
包含准确率提升策略和完整训练流程
"""

# 主程序保护模块（解决多进程问题）
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import time
    import os
    import torch.cuda.amp as amp

    # ------------------- 初始化设置 -------------------
    # 设置随机种子保证可重复性
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------- 数据预处理配置 -------------------
    # 根据数据集描述，训练数据已经应用了完整的预处理和增强
    # 因此只需要进行标准化和尺寸调整
    base_transform = transforms.Compose([
        transforms.Resize(256),          # 保持长宽比缩放
        transforms.CenterCrop(224),       # 中心裁剪到模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------- 数据加载 -------------------
    # 数据集路径配置
    data_dir = 'Rock Data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # 创建数据集对象（所有数据集使用相同预处理）
    train_dataset = datasets.ImageFolder(train_dir, transform=base_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=base_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=base_transform)

    # 检查类别平衡并计算样本权重
    class_counts = np.array([len(os.listdir(os.path.join(train_dir, cls))) for cls in train_dataset.classes])
    class_weights = 1. / class_counts
    sample_weights = [class_weights[train_dataset.targets[i]] for i in range(len(train_dataset))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 配置数据加载器
    batch_size = 64  # 增大批大小提升训练速度
    num_workers = 0 if os.name == 'nt' else 8  # Windows下设为0避免多进程问题

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # 使用加权采样处理类别不平衡
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ------------------- 模型构建 -------------------
    # 使用EfficientNet_V2_S（更高精度模型）
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    # 修改分类头
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  # 增加Dropout防止过拟合
        nn.Linear(num_features, len(train_dataset.classes))
    )

    # 设备配置与混合精度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = amp.GradScaler()  # 自动混合精度

    # ------------------- 训练配置 -------------------
    # 优化器组合（AdamW + 权重衰减）
    optimizer = optim.AdamW(model.parameters(), 
                          lr=1e-4, 
                          weight_decay=1e-4)

    # 学习率调度器组合
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    early_stop_patience = 5
    no_improve_epochs = 0

    # 训练记录
    history = {
        'train_loss': [],
        'val_acc': [],
        'lr': []
    }

    # ------------------- 训练循环 -------------------
    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"训练") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 混合精度训练
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # 反向传播优化
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 记录损失
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=loss.item())
        
        # 更新学习率
        scheduler.step()
        
        # 计算训练指标
        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 验证阶段
        model.eval()
        correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        val_acc = correct / len(val_dataset)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve_epochs = 0
            print(f"验证准确率提升至 {val_acc:.4f}，保存模型！")
        else:
            no_improve_epochs += 1
            print(f"验证准确率未提升，当前最佳：{best_val_acc:.4f}")
        
        # 早停检测
        if no_improve_epochs >= early_stop_patience:
            print(f"\n早停触发，连续{early_stop_patience}轮未提升")
            break

    # ------------------- 训练后处理 -------------------
    # 统计训练时间
    time_elapsed = time.time() - start_time
    print(f"\n训练完成，耗时: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 可视化训练过程
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='学习率')
    plt.title('学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # ------------------- 测试评估 -------------------
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
    
    test_acc = correct / len(test_dataset)
    print(f"\n测试准确率: {test_acc:.4f}")

    # 详细评估报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, test_preds))

    # 保存完整模型
    torch.save(model, 'final_model.pth')