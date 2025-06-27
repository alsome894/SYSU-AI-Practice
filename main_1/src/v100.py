if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import time
    import os
    import torch.cuda.amp as amp

    from pathlib import Path

    # ------------------- CUDA配置检查 -------------------
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ------------------- 初始化设置 -------------------
    # 固定随机种子保证可重复性
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 允许CUDA优化卷积算法

    # ------------------- 数据预处理 -------------------
    # 根据数据集描述，所有预处理已在数据增强阶段完成
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------- 数据加载 -------------------
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'Rock Data'

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=base_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=base_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=base_transform)

    # 类别平衡采样
    class_counts = np.array([len(os.listdir(os.path.join(train_dir, cls))) for cls in train_dataset.classes])
    class_weights = 1. / class_counts
    sample_weights = [class_weights[train_dataset.targets[i]] for i in range(len(train_dataset))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 数据加载器配置（优化CUDA内存使用）
    batch_size = 64
    num_workers = 0 if os.name == 'nt' else 8  # Windows下设为0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,  # 加速数据到GPU的传输
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
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
    # 使用ResNet34（根据文献建议）
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # 网络解冻策略（参考Yosinski的迁移学习研究）
    def unfreeze_layers(model):
        # 首先冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻最后两个残差块和全连接层（layer3、layer4）
        layers_to_unfreeze = ['layer3', 'layer4', 'fc']
        for name, module in model.named_children():
            if name in layers_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True
        return model

    model = unfreeze_layers(model)

    # 修改分类头（适配当前任务）
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(train_dataset.classes))
    )
    # ------------------- CUDA加速配置 -------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 混合精度训练配置
    scaler = amp.GradScaler()
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} | 总参数: {total_params} | 训练比例: {trainable_params/total_params:.2%}")

    # ------------------- 训练配置 -------------------
    # 分组优化器（不同层使用不同学习率）
    optimizer = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 1e-5},    # 中层较低学习率
        {'params': model.layer4.parameters(), 'lr': 1e-4},    # 高层中等学习率
        {'params': model.fc.parameters(), 'lr': 1e-3}         # 分类头较高学习率
    ], weight_decay=1e-4)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True)

    # 带标签平滑的损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    early_stop_patience = 5
    no_improve = 0

    # 训练记录
    history = {
        'train_loss': [],
        'val_acc': [],
        'lr': []
    }

    # ------------------- 训练循环 -------------------
    print("\n开始深度微调训练...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"训练") as pbar:
            for inputs, labels in pbar:
                # CUDA异步数据传输
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 混合精度前向传播
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # 梯度裁剪防止爆炸
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 参数更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # 更高效的内存释放
                
                # 记录损失
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=loss.item())
        
        # 计算平均train_loss
        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        history['val_acc'].append(val_acc)
        
        # 学习率调整
        scheduler.step(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
            print(f"验证准确率提升至 {val_acc:.4f}，模型已保存！")
        else:
            no_improve += 1
            print(f"验证准确率未提升，当前最佳：{best_val_acc:.4f}")
        
        # 早停检测
        '''if no_improve >= early_stop_patience:
            print(f"\n早停触发，连续{early_stop_patience}轮未提升")
            break'''
        
        # 打印epoch耗时
        epoch_time = time.time() - epoch_start
        print(f"Epoch耗时: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")

    # ------------------- 训练后处理 -------------------
    total_time = time.time() - start_time
    print(f"\n训练完成，总耗时: {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 可视化训练过程
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.title('train_loss_curve')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='test_accuracy')
    plt.title('test_accuracy_curve')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='learn_rate')
    plt.title('learn_rate变化')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # ------------------- 测试评估 -------------------
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        test_correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_correct += (preds == labels).sum().item()
    
    test_acc = test_correct / len(test_dataset)
    print(f"\n最终测试准确率: {test_acc:.4f}")

    print("\n详细分类报告:")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, test_preds))

    # 保存完整模型
    torch.save(model, 'final_model.pth')