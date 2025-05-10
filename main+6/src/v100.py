"""
基于RockS2Net的岩石图像分类实现
双分支网络结构：全局特征分支 + 局部特征分支
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
    from sklearn.metrics import classification_report
    import time
    import os
    import torch.nn.functional as F

    # ------------------- CUDA配置检查 -------------------
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ------------------- 自定义数据集类 -------------------
    class DualInputDataset(Dataset):
        """生成双输入数据：全局图像 + 局部裁剪"""
        def __init__(self, root_dir, transform=None):
            self.dataset = datasets.ImageFolder(root_dir, transform=transform)
            self.transform = transform
            
        def __getitem__(self, index):
            img, label = self.dataset[index]
            
            # 生成局部裁剪（随机裁剪原图的1/4区域）
            _, h, w = img.shape
            crop_size = int(h * 0.5)
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)
            local_img = img[:, top:top+crop_size, left:left+crop_size]
            local_img = F.interpolate(local_img.unsqueeze(0), size=(112, 112)).squeeze()
            
            return (img, local_img), label
        
        def __len__(self):
            return len(self.dataset)

    # ------------------- 数据预处理 -------------------
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------- 数据加载 -------------------
    data_dir = 'Rock Data'
    train_dir = os.path.join(data_dir, 'train')
    
    # 创建双输入数据集
    train_dataset = DualInputDataset(train_dir, transform=base_transform)
    val_dataset = DualInputDataset(os.path.join(data_dir, 'valid'), transform=base_transform)
    test_dataset = DualInputDataset(os.path.join(data_dir, 'test'), transform=base_transform)

    # 配置数据加载器
    batch_size = 32
    num_workers = 0 if os.name == 'nt' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ------------------- 网络定义 -------------------
    class GlobalBranch(nn.Module):
        """全局特征提取分支"""
        def __init__(self):
            super().__init__()
            resnet = models.resnet34(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后两层
            
        def forward(self, x):
            x = self.features(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return x.flatten(1)

    class LocalBranch(nn.Module):
        """局部特征提取分支（高分辨率处理）"""
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
        def forward(self, x):
            return self.conv_layers(x).flatten(1)

    class RockS2Net(nn.Module):
        """双分支岩石分类网络"""
        def __init__(self, num_classes):
            super().__init__()
            self.global_branch = GlobalBranch()
            self.local_branch = LocalBranch()
            
            # 特征融合层
            self.fc = nn.Sequential(
                nn.Linear(512 + 256, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, global_x, local_x):
            global_feat = self.global_branch(global_x)
            local_feat = self.local_branch(local_x)
            combined = torch.cat([global_feat, local_feat], dim=1)
            return self.fc(combined)

    # ------------------- 模型初始化 -------------------
    num_classes = len(train_dataset.dataset.classes)
    model = RockS2Net(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ------------------- 训练配置 -------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': model.global_branch.parameters(), 'lr': 1e-4},
        {'params': model.local_branch.parameters(), 'lr': 1e-3},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # ------------------- 训练循环 -------------------
    def train_model(model, train_loader, val_loader, epochs=50):
        best_acc = 0.0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            # 训练阶段
            for (global_imgs, local_imgs), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                global_imgs = global_imgs.to(device)
                local_imgs = local_imgs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(global_imgs, local_imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * global_imgs.size(0)

            # 验证阶段
            model.eval()
            correct = 0
            with torch.no_grad():
                for (global_imgs, local_imgs), labels in val_loader:
                    global_imgs = global_imgs.to(device)
                    local_imgs = local_imgs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(global_imgs, local_imgs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
            
            val_acc = correct / len(val_dataset)
            scheduler.step(val_acc)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'rocks2net_best.pth')
                print(f"验证准确率提升至 {val_acc:.4f}，模型已保存！")

        return model

    # 执行训练
    trained_model = train_model(model, train_loader, val_loader, epochs=50)

    # ------------------- 测试评估 -------------------
    trained_model.load_state_dict(torch.load('rocks2net_best.pth'))
    trained_model.eval()
    
    test_preds = []
    test_labels = []
    with torch.no_grad():
        correct = 0
        for (global_imgs, local_imgs), labels in test_loader:
            global_imgs = global_imgs.to(device)
            local_imgs = local_imgs.to(device)
            labels = labels.to(device)
            
            outputs = trained_model(global_imgs, local_imgs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
    
    print(f"\n测试准确率: {correct / len(test_dataset):.4f}")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.dataset.classes))