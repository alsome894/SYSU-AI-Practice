import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import copy
from multiprocessing import freeze_support
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {DEVICE}")

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
data_dir = project_root / 'Rock Data'
TRAIN_DIR = os.path.join(data_dir, 'train')
VALID_DIR = os.path.join(data_dir, 'valid')
TEST_DIR = os.path.join(data_dir, 'test')

LEARNING_RATE = 0.01
EPOCHS = 60
BATCH_SIZE = 2
IMG_SIZE = 600

class AddSaltPepperNoise:
    "给张量图像添加椒盐噪声"
    def __init__(self, amount=0.02, s_vs_p=0.5):
        self.amount = amount
        self.s_vs_p = s_vs_p

    def __call__(self, img_tensor):
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("输入必须是 PyTorch 张量。")
        if img_tensor.ndim != 3:
            raise ValueError("输入张量必须是3维 (C, H, W)。")
        c, h, w = img_tensor.shape
        num_pixels_per_channel = h * w
        out_tensor = img_tensor.clone()
        for i in range(c):
            num_salt = np.ceil(self.amount * num_pixels_per_channel * self.s_vs_p)
            coords_h_salt = torch.randint(0, h, (int(num_salt),))
            coords_w_salt = torch.randint(0, w, (int(num_salt),))
            out_tensor[i, coords_h_salt, coords_w_salt] = 1.0
            num_pepper = np.ceil(self.amount * num_pixels_per_channel * (1. - self.s_vs_p))
            coords_h_pepper = torch.randint(0, h, (int(num_pepper),))
            coords_w_pepper = torch.randint(0, w, (int(num_pepper),))
            out_tensor[i, coords_h_pepper, coords_w_pepper] = 0.0
        return torch.clamp(out_tensor, 0., 1.)

class AddGaussianNoise:
    "给张量图像添加高斯噪声"
    def __init__(self, mean=0., std=0.03):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("输入必须是 PyTorch 张量。")
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0., 1.)

class TripletAttention(nn.Module):
    "Triplet Attention 模块"
    def __init__(self, input_channels_placeholder=None):
        super(TripletAttention, self).__init__()
        self.k_size = 7
        self.conv_bn_sig_perm1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=self.k_size, stride=1, padding=(self.k_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )
        self.conv_bn_sig_perm2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=self.k_size, stride=1, padding=(self.k_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )
        self.conv_bn_sig_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=self.k_size, stride=1, padding=(self.k_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_pool1_max = torch.max(x_perm1, dim=2, keepdim=True)[0]
        x_pool1_avg = torch.mean(x_perm1, dim=2, keepdim=True)
        x_pool1_concat = torch.cat([x_pool1_max, x_pool1_avg], dim=2)
        x_pool1_for_conv = x_pool1_concat.permute(0, 2, 1, 3).contiguous()
        att_map1 = self.conv_bn_sig_perm1(x_pool1_for_conv)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_pool2_max = torch.max(x_perm2, dim=3, keepdim=True)[0]
        x_pool2_avg = torch.mean(x_perm2, dim=3, keepdim=True)
        x_pool2_concat = torch.cat([x_pool2_max, x_pool2_avg], dim=3)
        x_pool2_for_conv = x_pool2_concat.permute(0, 3, 1, 2).contiguous()
        att_map2_raw = self.conv_bn_sig_perm2(x_pool2_for_conv)
        att_map2 = att_map2_raw.permute(0, 1, 3, 2).contiguous()
        x_pool_spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        x_pool_spatial_avg = torch.mean(x, dim=1, keepdim=True)
        x_pool_spatial_concat = torch.cat([x_pool_spatial_max, x_pool_spatial_avg], dim=1)
        att_map_spatial = self.conv_bn_sig_spatial(x_pool_spatial_concat)
        attention_weights = (att_map1 + att_map2 + att_map_spatial) / 3.0
        return x * attention_weights

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    AddSaltPepperNoise(amount=0.02, s_vs_p=0.5),
    AddGaussianNoise(mean=0., std=0.03),
    normalize
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

def replace_se_with_triplet(features_module):
    "递归地将 EfficientNet features 中的 SqueezeExcitation 模块替换为 TripletAttention"
    for name, child_module in features_module.named_children():
        if isinstance(child_module, models.efficientnet.SqueezeExcitation):
            original_se_input_channels = child_module.fc1.in_channels
            triplet_attention_block = TripletAttention(original_se_input_channels)
            setattr(features_module, name, triplet_attention_block)
        elif len(list(child_module.children())) > 0:
            replace_se_with_triplet(child_module)

class TripletEfficientNetModel(nn.Module):
    "集成Triplet Attention的EfficientNet-B7模型"
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b7(weights=weights)
        replace_se_with_triplet(base_model.features)
        self.features = base_model.features
        efficientnet_b7_output_feature_channels = 2560
        self.head = nn.Sequential(
            nn.Conv2d(efficientnet_b7_output_feature_channels,
                      efficientnet_b7_output_feature_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(efficientnet_b7_output_feature_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(efficientnet_b7_output_feature_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    freeze_support()
    print(f"使用的设备: {DEVICE}")
    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 未找到。请确保它与脚本位于同一目录。")
        exit()
    if not os.path.isdir(TRAIN_DIR):
        print(f"错误: 训练数据目录 '{TRAIN_DIR}' 未找到。")
        exit()
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(VALID_DIR, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)
    CLASS_NAMES = train_dataset.classes
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"检测到 {NUM_CLASSES} 个类别: {', '.join(CLASS_NAMES)}")
    if NUM_CLASSES != 7 and NUM_CLASSES != 9:
         print(f"警告: 检测到的类别数量 ({NUM_CLASSES}) 与期望的 7 (指令 9) 或 9 (数据集描述) 不同。")
    print(f"模型将使用 {NUM_CLASSES} 个类别。")
    num_workers = min(os.cpu_count(), 4)
    print(f"为 DataLoader 使用 {num_workers} 个 worker (设置为0进行调试)。")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)
    model = TripletEfficientNetModel(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_valid_accuracy = 0.0
    best_model_weights = None
    print(f"\n开始训练 {EPOCHS} 个周期...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"周期 {epoch+1}/{EPOCHS} [训练]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix({'损失': running_loss/total_train, '准确率': correct_train/total_train})
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        valid_pbar = tqdm(valid_loader, desc=f"周期 {epoch+1}/{EPOCHS} [验证]")
        with torch.no_grad():
            for inputs, labels in valid_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                valid_pbar.set_postfix({'损失': valid_loss/total_valid, '准确率': correct_valid/total_valid})
        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        epoch_valid_acc = correct_valid / total_valid
        print(f"周期 {epoch+1}/{EPOCHS} - "
              f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.4f} - "
              f"验证损失: {epoch_valid_loss:.4f}, 验证准确率: {epoch_valid_acc:.4f}")
        if epoch_valid_acc > best_valid_accuracy:
            best_valid_accuracy = epoch_valid_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"新的最佳验证准确率: {best_valid_accuracy:.4f}。正在保存模型权重。")
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        print("\n已加载最佳模型权重 (基于验证准确率) 进行最终评估。")
    else:
        print("\n训练期间未找到更优模型 (或验证准确率未提升)。使用最后一个周期的模型状态进行评估。")
    print("\n在测试集上评估...")
    model.eval()
    all_preds = []
    all_labels = []
    test_pbar = tqdm(test_loader, desc="[测试评估]")
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\n--- 分类报告 (Classification Report) ---")
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(report)
    print("\n--- 混淆矩阵 (Confusion Matrix) ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plt.switch_backend('agg')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n混淆矩阵已保存至 confusion_matrix.png")
    MODEL_SAVE_PATH = 'triplet_efficientnet_rock_classifier.pth'
    try:
        torch.save(model, MODEL_SAVE_PATH)
        print(f"\n完整模型已保存至 {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"\n保存完整模型时出错: {e}")
        print("尝试改为保存模型 state_dict。")
        MODEL_STATE_DICT_SAVE_PATH = 'triplet_efficientnet_rock_classifier_statedict.pth'
        torch.save(model.state_dict(), MODEL_STATE_DICT_SAVE_PATH)
        print(f"模型 state_dict 已保存至 {MODEL_STATE_DICT_SAVE_PATH}")
    print("\n脚本执行完毕。")
