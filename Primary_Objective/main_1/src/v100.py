import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import numpy as np

from pathlib import Path

def plot_metrics(history, prefix=""):
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{prefix}Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{prefix}Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{prefix.lower().replace(' ', '_')}training_curves.png")
    print(f"训练曲线图已保存为 {prefix.lower().replace(' ', '_')}training_curves.png")
    plt.show() # 取消注释以在运行时显示

def run_primary_goal():
    print("开始初级目标：使用基础CNN进行岩石分类...")

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. Hyperparameters and Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    NUM_CLASSES = 9

    # 3. Dataset Paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    data_dir = project_root / 'Rock Data'
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # 4. Data Preprocessing and Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # 5. Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    class_names = train_dataset.classes
    print(f"岩石类别: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"警告: 找到 {len(class_names)} 个类别，但 NUM_CLASSES 设置为 {NUM_CLASSES}")

    # 7. Define Basic CNN Model
    class BasicCNN(nn.Module):
        def __init__(self, num_classes):
            super(BasicCNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), 

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), 

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512), # Adjusted for IMG_SIZE
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    model = BasicCNN(NUM_CLASSES).to(device)

    # 8. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 9. Train Model
    print("\n开始模型训练...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss_train_epoch = 0.0
        correct_train_epoch = 0
        total_train_epoch = 0
        
        train_pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{NUM_EPOCHS} [训练中]", unit="批次")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train_epoch += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_epoch += labels.size(0)
            correct_train_epoch += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_loss_train = running_loss_train_epoch / len(train_loader.dataset)
        epoch_acc_train = correct_train_epoch / total_train_epoch
        history['train_loss'].append(epoch_loss_train)
        history['train_acc'].append(epoch_acc_train)

        model.eval()
        running_loss_val_epoch = 0.0
        correct_val_epoch = 0
        total_val_epoch = 0
        
        val_pbar = tqdm(valid_loader, desc=f"轮次 {epoch+1}/{NUM_EPOCHS} [验证中]", unit="批次")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss_val_epoch += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_epoch += labels.size(0)
                correct_val_epoch += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': loss.item()})

        epoch_loss_val = running_loss_val_epoch / len(valid_loader.dataset)
        epoch_acc_val = correct_val_epoch / total_val_epoch
        history['val_loss'].append(epoch_loss_val)
        history['val_acc'].append(epoch_acc_val)

        print(f"轮次 {epoch+1}/{NUM_EPOCHS} - "
              f"训练损失: {epoch_loss_train:.4f}, 训练准确率: {epoch_acc_train:.4f} - "
              f"验证损失: {epoch_loss_val:.4f}, 验证准确率: {epoch_acc_val:.4f}")

    print("训练完成！")

    # Plot training history
    plot_metrics(history, prefix="Primary Goal ")

    # 10. Evaluate Model on Test Set
    print("\n在测试集上评估模型...")
    model.eval()
    all_labels = []
    all_preds = []
    correct_test = 0
    total_test = 0

    test_pbar = tqdm(test_loader, desc="测试中", unit="批次")
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_accuracy = correct_test / total_test
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 11. Classification Report and Confusion Matrix
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("\n混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10)) # Create a new figure and axes for the confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    ax.set_title('Primary Goal - Confusion Matrix') # Use ax.set_title
    plt.tight_layout()
    plt.savefig("primary_goal_confusion_matrix.png")
    print("混淆矩阵图已保存为 primary_goal_confusion_matrix.png")
    plt.show()

    # 12. Save Full Model
    model_path = "basic_cnn_rock_classifier.pth"
    torch.save(model, model_path)
    print(f"\n完整模型已保存到: {model_path}")
    
    print("\n初级目标执行完成。")

if __name__ == '__main__':
    if not os.path.isdir("./Rock Data"):
        print("错误: 当前目录中未找到 'Rock Data' 文件夹。请确保数据集已正确放置。")
    else:
        start_time = time.time()
        run_primary_goal()
        end_time = time.time()
        print(f"初级目标总执行时间: {(end_time - start_time)/60:.2f} 分钟")
