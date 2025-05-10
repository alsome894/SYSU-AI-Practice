"""
基于RockS2Net的岩石分类统一格式实现
包含训练曲线、混合精度训练和完整评估流程
"""

# 主程序保护模块（解决多进程问题）
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import time
    import os
    import torch.nn.functional as F
    import torch.cuda.amp as amp

    # ------------------- 初始化配置 -------------------
    # CUDA可用性检查
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 固定随机种子
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # ------------------- 数据预处理 -------------------
    class DualInputDataset(Dataset):
        """双输入数据集：全局图像 + 局部裁剪"""
        def __init__(self, root_dir):
            self.dataset = datasets.ImageFolder(
                root_dir,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            )
            
        def __getitem__(self, index):
            img, label = self.dataset[index]
            # 生成局部裁剪（原图的50%区域）
            _, h, w = img.shape
            crop_size = h // 2
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)
            local_img = img[:, top:top+crop_size, left:left+crop_size]
            local_img = F.interpolate(local_img.unsqueeze(0), size=112, mode='bilinear').squeeze()
            return (img, local_img), label
        
        def __len__(self):
            return len(self.dataset)

    # ------------------- 数据加载 -------------------
    data_dir = 'Rock Data'
    train_dataset = DualInputDataset(os.path.join(data_dir, 'train'))
    val_dataset = DualInputDataset(os.path.join(data_dir, 'valid'))
    test_dataset = DualInputDataset(os.path.join(data_dir, 'test'))

    # 数据加载器配置
    batch_size = 32
    num_workers = 0 if os.name == 'nt' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ------------------- 模型定义 -------------------
    class RockS2Net(nn.Module):
        """双分支岩石分类网络"""
        def __init__(self, num_classes):
            super().__init__()
            # 全局分支（预训练ResNet34）
            self.global_net = nn.Sequential(
                *list(models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).children())[:-2],
                nn.AdaptiveAvgPool2d((1, 1))
            )
            # 局部分支（自定义CNN）
            self.local_net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            # 特征融合
            self.classifier = nn.Sequential(
                nn.Linear(512 + 128, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, global_x, local_x):
            global_feat = self.global_net(global_x).flatten(1)
            local_feat = self.local_net(local_x).flatten(1)
            combined = torch.cat([global_feat, local_feat], dim=1)
            return self.classifier(combined)

    # ------------------- 训练配置 -------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RockS2Net(len(train_dataset.dataset.classes)).to(device)
    scaler = amp.GradScaler()

    # 优化器配置
    optimizer = optim.AdamW([
        {'params': model.global_net.parameters(), 'lr': 1e-4},
        {'params': model.local_net.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    early_stop_patience = 5
    history = {'train_loss': [], 'val_acc': [], 'lr': []}

    # ------------------- 训练循环 -------------------
    print("\n开始训练双分支网络...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc="训练") as pbar:
            for (global_imgs, local_imgs), labels in pbar:
                global_imgs = global_imgs.to(device, non_blocking=True)
                local_imgs = local_imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 混合精度训练
                with amp.autocast():
                    outputs = model(global_imgs, local_imgs)
                    loss = criterion(outputs, labels)
                
                # 反向传播优化
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # 记录损失
                running_loss += loss.item() * global_imgs.size(0)
                pbar.set_postfix(loss=loss.item())
        
        # 计算epoch损失
        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for (global_imgs, local_imgs), labels in val_loader:
                global_imgs = global_imgs.to(device)
                local_imgs = local_imgs.to(device)
                labels = labels.to(device)
                
                outputs = model(global_imgs, local_imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"验证准确率提升至 {val_acc:.4f}，模型已保存！")
        else:
            print(f"验证准确率未提升，当前最佳：{best_val_acc:.4f}")
        
        # 早停检测
        if (epoch - np.argmax(history['val_acc'])) > early_stop_patience:
            print(f"\n早停触发，连续{early_stop_patience}轮未提升")
            break
        
        # 打印时间统计
        epoch_time = time.time() - epoch_start
        print(f"Epoch耗时: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")

    # ------------------- 训练后处理 -------------------
    total_time = time.time() - start_time
    print(f"\n训练完成，总耗时: {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 可视化训练曲线
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
    test_correct = 0
    
    with torch.no_grad():
        for (global_imgs, local_imgs), labels in test_loader:
            global_imgs = global_imgs.to(device)
            local_imgs = local_imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(global_imgs, local_imgs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print(f"\n测试准确率: {test_correct/len(test_dataset):.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.dataset.classes))
    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, test_preds))

    # 保存完整模型
    torch.save(model, 'final_model.pth')