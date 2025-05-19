import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms # models is not strictly needed now
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import numpy as np

from pathlib import Path

# (plot_metrics function remains the same as provided in the previous response)
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
    save_path = f"{prefix.lower().replace(' ', '_').replace('-', '_')}training_curves.png"
    plt.savefig(save_path)
    print(f"训练曲线图已保存为 {save_path}")
    # plt.show()

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes, img_size_for_fc_calc, dropout_rate=0.4):
        super(EnhancedCNN, self).__init__()
        self.img_size_for_fc_calc = img_size_for_fc_calc

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # IMG_SIZE -> IMG_SIZE/2

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # IMG_SIZE/2 -> IMG_SIZE/4

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # IMG_SIZE/4 -> IMG_SIZE/8

        # Convolutional Block 4 (Enhanced: More filters, deeper)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # IMG_SIZE/8 -> IMG_SIZE/16
        
        # Calculate flattened size dynamically for FC layers
        with torch.no_grad(): # Avoid tracking gradients for this dummy pass
            # Create a dummy input with the specified image size
            dummy_input = torch.zeros(1, 3, self.img_size_for_fc_calc, self.img_size_for_fc_calc)
            # Pass the dummy input through the convolutional part of the network
            dummy_output = self._forward_conv_part(dummy_input)
            # Calculate the flattened size
            self.flattened_size = dummy_output.view(1, -1).size(1)
            # print(f"Dynamically calculated flattened size: {self.flattened_size}") # For debugging

        self.flatten = nn.Flatten()
        
        # Fully Connected Layers (Enhanced: More layers, dropout)
        self.fc1 = nn.Linear(self.flattened_size, 1024) # Increased neurons
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512) # Additional hidden layer
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, num_classes) # Output layer

    def _forward_conv_part(self, x):
        # Helper method to pass input through convolutional blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_conv_part(x) # Pass through conv blocks
        x = self.flatten(x)           # Flatten the output
        x = self.dropout1(self.relu_fc1(self.fc1(x))) # Pass through first FC layer with ReLU and Dropout
        x = self.dropout2(self.relu_fc2(self.fc2(x))) # Pass through second FC layer with ReLU and Dropout
        x = self.fc3(x)               # Output layer
        return x

def run_intermediate_goal_optimized_cnn():
    print("开始中级目标：使用增强的自定义CNN和数据增强进行岩石分类...")

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. Hyperparameters and Configuration
    IMG_SIZE = 224       # Keep consistent or ensure model adapts
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005 # Potentially smaller LR for a more complex model
    NUM_EPOCHS = 50      # Might need more epochs for a deeper model
    NUM_CLASSES = 9
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    DROPOUT_RATE = 0.4   # Dropout rate for FC layers

    # 3. Dataset Paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    data_dir = project_root / 'Rock Data'
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # 4. Data Preprocessing and Transforms (with Data Augmentation)
    mean = [0.485, 0.456, 0.406] # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std

    train_transforms_augmented = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20), # Increased rotation slightly
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1), # Increased jitter
        transforms.RandomAffine(degrees=0, shear=10), # Added shear as mentioned in dataset description
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # 5. Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms_augmented)
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

    # 7. Define Enhanced Custom CNN Model
    model = EnhancedCNN(num_classes=NUM_CLASSES, img_size_for_fc_calc=IMG_SIZE, dropout_rate=DROPOUT_RATE).to(device)
    print("\n增强的CNN模型架构:")
    print(model)
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数: {total_params:,}")


    # 8. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Optional: Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # 9. Train Model
    print("\n开始模型训练...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = "enhanced_custom_cnn_best_statedict.pth"

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
            train_pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        
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

        # if scheduler: scheduler.step(epoch_loss_val) # For ReduceLROnPlateau

        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            torch.save(model.state_dict(), best_model_path)
            print(f"新的最佳模型状态字典已保存，轮次 {epoch+1}，验证准确率: {best_val_acc:.4f}，保存路径: {best_model_path}")

    print("训练完成！")

    plot_metrics(history, prefix="Intermediate Goal-EnhancedCNN ")

    print(f"\n从 {best_model_path} 加载最佳模型状态字典进行测试...")
    if os.path.exists(best_model_path):
        # Re-initialize model architecture before loading state_dict
        eval_model = EnhancedCNN(num_classes=NUM_CLASSES, img_size_for_fc_calc=IMG_SIZE, dropout_rate=DROPOUT_RATE).to(device)
        eval_model.load_state_dict(torch.load(best_model_path))
        print(f"成功从 {best_model_path} 加载最佳模型状态字典")
    else:
        print(f"警告: 未找到最佳模型状态字典文件 {best_model_path}。将使用训练结束时的模型进行测试。")
        eval_model = model # Use the model from the end of training
    
    eval_model.eval()

    # 10. Evaluate Model on Test Set
    print("\n在测试集上评估优化后的自定义CNN模型...")
    all_labels_test = []
    all_preds_test = []
    correct_test = 0
    total_test = 0

    test_pbar = tqdm(test_loader, desc="测试中", unit="批次")
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = eval_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_labels_test.extend(labels.cpu().numpy())
            all_preds_test.extend(predicted.cpu().numpy())

    test_accuracy = correct_test / total_test
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 11. Classification Report and Confusion Matrix
    print("\n分类报告（测试集）:")
    print(classification_report(all_labels_test, all_preds_test, target_names=class_names, zero_division=0))

    print("\n混淆矩阵（测试集）:")
    cm_test = confusion_matrix(all_labels_test, all_preds_test)
    print(cm_test)

    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
    fig_test, ax_test = plt.subplots(figsize=(10,10))
    disp_test.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax_test)
    ax_test.set_title('Intermediate Goal - EnhancedCNN Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig("intermediate_goal_enhanced_cnn_confusion_matrix.png")
    print("混淆矩阵图已保存为 intermediate_goal_enhanced_cnn_confusion_matrix.png")
    # plt.show()

    # 12. Save Full Model (final state of the best loaded model or last epoch model)
    final_model_path = "enhanced_custom_cnn_final_full_model.pth"
    torch.save(eval_model, final_model_path)
    print(f"\n最终完整优化自定义CNN模型已保存到: {final_model_path}")
    
    print("\n中级目标（优化自定义CNN）执行完成。")

if __name__ == '__main__':
    if not os.path.isdir("./Rock Data"):
        print("错误: 当前目录中未找到 'Rock Data' 文件夹。请确保数据集已正确放置。")
    else:
        start_time = time.time()
        run_intermediate_goal_optimized_cnn()
        end_time = time.time()
        print(f"中级目标（优化自定义CNN）总执行时间: {(end_time - start_time)/60:.2f} 分钟")

