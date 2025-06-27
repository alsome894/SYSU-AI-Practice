# 岩石图像分类增强版：EfficientNet-B7 + Triplet Attention + MixUp + FocalLoss + 高级数据增强

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda.amp as amp
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path

# ------------------- Triplet Attention 模块 -------------------
class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.bn(self.conv(cat)))
        return x * attn

# ------------------- MixUp 增强 -------------------
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------- Focal Loss -------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        return self.alpha * (1 - p) ** self.gamma * logp

# ------------------- EarlyStopping -------------------
class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None or val_acc - self.best_score > self.min_delta:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ------------------- 主程序 -------------------
if __name__ == '__main__':
    # 固定随机种子
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 数据路径
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'Rock Data'

    # 数据增强
    IMG_SIZE = 480
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 数据集与采样
    train_dataset = datasets.ImageFolder(data_dir / 'train', transform=transform_train)
    val_dataset = datasets.ImageFolder(data_dir / 'valid', transform=transform_eval)
    test_dataset = datasets.ImageFolder(data_dir / 'test', transform=transform_eval)

    class_counts = np.array([len(os.listdir(data_dir / 'train' / cls)) for cls in train_dataset.classes])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[y] for y in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 加载器
    batch_size = 8
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    # 模型构建：EfficientNet-B7
    model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)

    # 用 Triplet Attention 替换部分 SE 模块
    for idx in [2, 3, 4]:
        try:
            block = model.features[idx][0].block
            if hasattr(block, 'se'):
                block.se = TripletAttention()
        except Exception:
            pass

    # 修改分类头
    num_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_f, len(train_dataset.classes))
    )
    model = model.to(device)

    # 损失、优化器、调度、混合精度
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = amp.GradScaler()
    early_stop = EarlyStopping(patience=12)

    # 训练与验证
    epochs = 50
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            x, y_a, y_b, lam = mixup_data(x, y)
            optimizer.zero_grad()
            with amp.autocast():
                preds = model(x)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out = model(x_val)
                _, pred = out.max(1)
                correct += (pred == y_val).sum().item()
        val_acc = correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")
        '''if early_stop(val_acc):
            print("Early stopping triggered.")
            break'''

    # 测试评估
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t, y_t = x_t.to(device), y_t.to(device)
            out = model(x_t)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_t.cpu().numpy())
    test_acc = np.mean(np.array(all_preds)==np.array(all_labels))
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
    print(confusion_matrix(all_labels, all_preds))
