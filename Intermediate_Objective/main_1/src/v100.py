import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import numpy as np # 引入numpy用于绘图时的x轴

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
    # plt.show() # 取消注释以在运行时显示

def run_intermediate_goal():
    print("开始中级目标：使用优化的ResNet18和数据增强进行岩石分类...")

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. Hyperparameters and Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 25
    NUM_CLASSES = 9
    WEIGHT_DECAY = 1e-4

    # 3. Dataset Paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    data_dir = project_root / 'Rock Data'
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # 4. Data Preprocessing and Transforms (with Data Augmentation)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms_augmented = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Consider adding transforms.RandomAffine(degrees=0, shear=10) if shear is important
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Or Resize(256) then CenterCrop(224)
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

    # 7. Define Optimized Model (Pre-trained ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Fine-tuning: Unfreeze all layers or a subset of layers
    # For full fine-tuning, all params require_grad should be True (default for new layers)
    # For partial fine-tuning, freeze earlier layers:
    # for name, param in model.named_parameters():
    #     if "fc" not in name and "layer4" not in name: # Example: unfreeze only last block and fc
    #         param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # Replace classifier
    model = model.to(device)

    # 8. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Optional

    # 9. Train Model
    print("\n开始模型训练...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = "optimized_rock_classifier_best_statedict.pth" # Saving state_dict

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

        # if scheduler: scheduler.step()

        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            torch.save(model.state_dict(), best_model_path)
            print(f"新的最佳模型状态字典已保存，轮次 {epoch+1}，验证准确率: {best_val_acc:.4f}，保存路径: {best_model_path}")

    print("训练完成！")

    # Plot training history
    plot_metrics(history, prefix="Intermediate Goal ")

    # Load the best model state_dict for evaluation
    print(f"\n从 {best_model_path} 加载最佳模型状态字典进行测试...")
    if os.path.exists(best_model_path):
        # Re-define the model structure before loading state_dict
        # model_for_eval = models.resnet18(weights=None) # No pre-trained weights when loading state_dict for a modified model
        # num_ftrs_eval = model_for_eval.fc.in_features
        # model_for_eval.fc = nn.Linear(num_ftrs_eval, NUM_CLASSES)
        # model_for_eval.load_state_dict(torch.load(best_model_path))
        # model_for_eval = model_for_eval.to(device)
        # model_for_eval.eval()
        # current_model = model_for_eval # Use this model for evaluation
        model.load_state_dict(torch.load(best_model_path)) # simpler if model structure hasn't changed since saving
        print(f"成功从 {best_model_path} 加载最佳模型状态字典")
    else:
        print(f"警告: 未找到最佳模型状态字典文件 {best_model_path}。将使用训练结束时的模型进行测试。")
        # current_model = model # Use the model from the end of training

    current_model = model # Ensure current_model is defined for evaluation
    current_model.eval() # Set to evaluation mode

    # 10. Evaluate Model on Test Set
    print("\n在测试集上评估优化后的模型...")
    all_labels = []
    all_preds = []
    correct_test = 0
    total_test = 0

    test_pbar = tqdm(test_loader, desc="测试中", unit="批次")
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = current_model(inputs) # Use current_model
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
    fig, ax = plt.subplots(figsize=(10, 10)) # Create a new figure and axes
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    ax.set_title('Intermediate Goal - Optimized Model Confusion Matrix') # Use ax.set_title
    plt.tight_layout()
    plt.savefig("intermediate_goal_confusion_matrix.png")
    print("混淆矩阵图已保存为 intermediate_goal_confusion_matrix.png")
    # plt.show()

    # 12. Save Full Model (final state)
    final_model_path = "optimized_cnn_rock_classifier_final_full_model.pth"
    torch.save(current_model, final_model_path) # Save the full model (best one if loaded, or final if not)
    print(f"\n最终完整优化模型已保存到: {final_model_path}")
    
    print("\n中级目标执行完成。")

if __name__ == '__main__':
    if not os.path.isdir("./Rock Data"):
        print("错误: 当前目录中未找到 'Rock Data' 文件夹。请确保数据集已正确放置。")
    else:
        # You can choose which goal to run
        # run_primary_goal()
        
        start_time = time.time()
        run_intermediate_goal()
        end_time = time.time()
        print(f"中级目标总执行时间: {(end_time - start_time)/60:.2f} 分钟")
