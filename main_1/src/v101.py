"""
岩石图像分类优化方案（正确增强顺序版）
包含先切片后增强、迁移学习与可视化
"""

# 主程序保护模块（解决多进程问题）
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    import time
    import os
    import random
    from pathlib import Path
    import torch.nn.functional as F

    # ------------------- 初始化配置 -------------------

    # CUDA可用性检查
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ------------------- 自定义数据增强 -------------------
    class SliceFirstTransform:
        """先切片后增强的复合变换"""
        def __init__(self):
            self.base_size = 640  # 原始图像尺寸
            self.slice_size = 320  # 切片尺寸
            
        def __call__(self, img):
            """
            正确增强流程：
            1. 原始图像切片
            2. 对每个切片独立进行增强
            """
            # 第一步：图像切片
            slices = []
            for i in range(2):
                for j in range(2):
                    left = j * self.slice_size
                    upper = i * self.slice_size
                    right = left + self.slice_size
                    lower = upper + self.slice_size
                    slice_img = transforms.functional.crop(img, upper, left, 
                                                         self.slice_size, self.slice_size)
                    slices.append(slice_img)
            
            # 第二步：对每个切片独立增强
            augmented_slices = []
            for slice_img in slices:
                # 随机缩放 (0.5-2倍)
                scale_factor = random.uniform(0.5, 2.0)
                new_size = int(self.slice_size * scale_factor)
                slice_img = transforms.Resize(new_size)(slice_img)
                
                # 随机旋转 (0-360度)
                angle = random.randint(0, 360)
                slice_img = transforms.functional.rotate(slice_img, angle)
                
                # 随机平移 (10-20像素)
                max_offset = random.randint(10, 20)
                translate_x = random.randint(-max_offset, max_offset)
                translate_y = random.randint(-max_offset, max_offset)
                slice_img = transforms.functional.affine(
                    slice_img, angle=0, translate=(translate_x, translate_y),
                    scale=1.0, shear=0
                )
                
                augmented_slices.append(slice_img)
            
            return augmented_slices

    # ------------------- 自定义数据集 -------------------
    class CorrectOrderDataset(Dataset):
        """正确的数据增强顺序数据集"""
        def __init__(self, root_dir):
            self.original_dataset = datasets.ImageFolder(root_dir)
            self.transform = transforms.Compose([
                SliceFirstTransform(),  # 先切片
                transforms.Lambda(lambda x: [  # 对每个切片单独处理
                    transforms.Compose([  # 切片级增强组合
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])(img) for img in x
                ])
            ])
            
        def __getitem__(self, index):
            original_img, label = self.original_dataset[index]
            processed_slices = self.transform(original_img)
            return torch.stack(processed_slices), label
        
        def __len__(self):
            return len(self.original_dataset)

    # ------------------- 数据加载 -------------------
    # 路径配置
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'Rock Data'

    # 创建数据集（训练集使用新增强顺序）
    train_dataset = CorrectOrderDataset(os.path.join(data_dir, 'train'))
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'valid'),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # 数据加载器配置
    batch_size = 16
    num_workers = 0 if os.name == 'nt' else 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ------------------- 模型构建 -------------------
    # 加载纹理数据集预训练模型
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # 加载自定义预训练权重（纹理库）
    '''texture_weights = torch.load('texture_pretrained.pth')
    model.load_state_dict(texture_weights)'''
    
    # 冻结所有层（除全连接层）
    for param in model.parameters():
        param.requires_grad = False
        
    # 重构分类头（使用ReLU激活）
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),  # 原地操作节省内存
        nn.Dropout(0.5),
        nn.Linear(512, len(train_dataset.original_dataset.classes))
    )
    # 参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} | 总参数: {total_params}")

    # ------------------- 训练配置 -------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 优化器设置（Adam算法）
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=0.001,
        betas=(0.9, 0.999)
    )
    # 学习率调度
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练参数
    num_epochs = 60
    best_val_acc = 0.0
    global_step = 0  # 全局训练步数

    # 训练记录
    history = {
        'train_loss': [],
        'val_acc': [],
        'lr': []
    }

    # ------------------- 训练循环 -------------------
    print("\n启动优化后训练流程...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for sliced_imgs, labels in tqdm(train_loader, desc="训练进度"):
            # 处理切片图像维度 [batch, 4, C, H, W] -> [batch*4, C, H, W]
            batch_size = sliced_imgs.size(0)
            inputs = sliced_imgs.view(-1, 3, 224, 224).to(device)
            labels = labels.repeat_interleave(4).to(device)  # 每个切片继承标签
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            

        # 计算epoch指标
        epoch_loss = running_loss / (len(train_dataset)*4)  # 每个样本4个切片
        epoch_acc = running_corrects.double() / (len(train_dataset)*4)
        history['train_loss'].append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        history['val_acc'].append(val_acc)
        
        # 学习率调整
        scheduler.step()
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"验证准确率提升至 {val_acc:.4f}，模型已保存！")
        
        # 打印统计信息
        epoch_time = time.time() - epoch_start
        print(f"训练损失: {epoch_loss:.4f} | 训练准确率: {epoch_acc:.4f}")
        print(f"验证准确率: {val_acc:.4f} | 耗时: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")

    # ------------------- 最终处理 -------------------
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

    # 模型测试与保存
    model.load_state_dict(torch.load('best_model.pth'))
    torch.save(model, 'final_model.pth')
    print("最终模型已保存为final_model.pth")

    # ------------------- 测试评估 -------------------
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print("\n测试分类报告:")
    print(classification_report(test_labels, test_preds, 
                              target_names=test_dataset.classes))
    print("混淆矩阵:")
    print(confusion_matrix(test_labels, test_preds))