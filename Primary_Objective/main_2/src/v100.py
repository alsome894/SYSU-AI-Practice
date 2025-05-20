import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
import joblib

def extract_features_labels(loader: DataLoader, img_size: int):
    """
    Extracts flattened image features and labels from a PyTorch DataLoader.
    Images are expected to be C, H, W.
    """
    all_features_list = []
    all_labels_list = []
    
    # Calculate expected feature dimension (C * H * W)
    # Assuming 3 color channels (RGB)
    expected_feature_dim = 3 * img_size * img_size
    
    print(f"Extracting features from {len(loader.dataset)} images (expected dim: {expected_feature_dim})...")

    for inputs, labels in tqdm(loader, desc="Extracting features"):
        # Inputs are (batch_size, C, H, W)
        # Ensure inputs are on CPU for numpy conversion
        inputs_cpu = inputs.cpu()
        
        # Flatten: (batch_size, C, H, W) -> (batch_size, C*H*W)
        # The ToTensor transform already puts it in C, H, W order.
        # inputs_cpu.shape will be [batch_size, 3, img_size, img_size]
        current_batch_size = inputs_cpu.shape[0]
        features = inputs_cpu.view(current_batch_size, -1).numpy() # Flatten
        
        all_features_list.append(features)
        all_labels_list.append(labels.cpu().numpy())

    if not all_features_list: # Handle empty loader case
        return np.array([]).reshape(0, expected_feature_dim), np.array([])

    all_features_np = np.concatenate(all_features_list, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    
    return all_features_np, all_labels_np

def run_primary_goal_svm():
    print("开始初级目标：使用SVM进行岩石分类...")
    print("警告：使用高维度原始像素作为特征 ({}x{}x3) 可能导致训练缓慢和高内存使用。".format(IMG_SIZE, IMG_SIZE))

    # 1. Configuration (No GPU needed for sklearn SVM directly, but DataLoader might use it)
    # Device for DataLoader if transformations are heavy, though features will be on CPU for sklearn
    # data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"数据加载将尝试使用: {data_device}") # PyTorch DataLoader might use this

    # 2. Hyperparameters and Configuration
    # Global IMG_SIZE for this script
    # BATCH_SIZE for DataLoader
    
    # 3. Dataset Paths (using the structure from your provided CNN code)
    try:
        current_file = Path(__file__).resolve()
        # Assuming 'Rock Data' is four levels up from the script's location as per your example.
        # Adjust this if your project structure is different.
        # A common structure might be: project_root/scripts/goals/primary_goal.py
        # and project_root/Rock Data
        # If script is in project_root, then project_root = current_file.parent
        project_root = current_file.parent.parent.parent.parent 
        data_dir = project_root / 'Rock Data'
        if not data_dir.is_dir(): # Fallback for simpler structures or if __file__ is not ideal
            print(f"警告: 路径 {data_dir} 未找到。尝试备用路径 './Rock Data'")
            data_dir = Path("./Rock Data")
    except NameError: # __file__ is not defined (e.g. in some interactive environments)
         print("警告: __file__ 未定义。使用 './Rock Data' 作为数据目录。")
         data_dir = Path("./Rock Data")

    train_dir = data_dir / "train"
    # valid_dir = data_dir / "valid" # Validation set not explicitly used for basic SVM training here
    test_dir = data_dir / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        print(f"错误: 训练目录 ({train_dir}) 或测试目录 ({test_dir}) 未找到。请检查路径。")
        return

    print(f"使用训练数据目录: {train_dir}")
    print(f"使用测试数据目录: {test_dir}")

    # 4. Data Preprocessing and Transforms
    # Normalization values from your CNN code
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transforms for loading images
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), # Converts to [C, H, W] and scales to [0, 1]
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # 5. Load Datasets using ImageFolder
    train_dataset = datasets.ImageFolder(str(train_dir), transform=data_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=data_transforms)
    # valid_dataset = datasets.ImageFolder(str(valid_dir), transform=data_transforms)


    # 6. Create DataLoaders (to efficiently load and transform data in batches)
    # SVM will process all data at once, so shuffle for feature extraction DataLoaders isn't critical
    # but batch_size helps manage memory during loading.
    # num_workers and pin_memory are PyTorch DataLoader optimizations.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)


    class_names = train_dataset.classes
    print(f"岩石类别: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"警告: 找到 {len(class_names)} 个类别，但 NUM_CLASSES 设置为 {NUM_CLASSES}")

    # 7. Extract Features and Labels
    print("为训练集提取特征...")
    X_train, y_train = extract_features_labels(train_loader, IMG_SIZE)
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")

    print("\n为测试集提取特征...")
    X_test, y_test = extract_features_labels(test_loader, IMG_SIZE)
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")

    if X_train.size == 0 or X_test.size == 0:
        print("错误：未能从一个或多个数据集中提取特征。请检查数据集和路径。")
        return

    # 8. Train SVM Model
    print("\n开始训练SVM模型...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=False) # Basic SVM
    # For large datasets, probability=True can be very slow.
    # Consider 'linear' kernel if 'rbf' is too slow or for very high-dim data.
    
    start_train_time = time.time()
    svm_model.fit(X_train, y_train)
    end_train_time = time.time()
    print(f"SVM模型训练完成。耗时: {(end_train_time - start_train_time):.2f} 秒")

    # 9. Evaluate Model on Test Set
    print("\n在测试集上评估SVM模型...")
    start_eval_time = time.time()
    y_pred_test = svm_model.predict(X_test)
    end_eval_time = time.time()
    print(f"SVM模型评估完成。耗时: {(end_eval_time - start_eval_time):.2f} 秒")

    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 10. Classification Report and Confusion Matrix
    print("\n分类报告 (测试集):")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

    print("\n混淆矩阵 (测试集):")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)

    # Plotting Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    ax.set_title('Primary Goal - SVM Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig("primary_goal_svm_confusion_matrix.png")
    print("SVM混淆矩阵图已保存为 primary_goal_svm_confusion_matrix.png")
    # plt.show()

    # 11. Save SVM Model
    model_path = "svm_rock_classifier.joblib"
    joblib.dump(svm_model, model_path)
    print(f"\nSVM模型已保存到: {model_path}")
    
    print("\n基于SVM的初级目标执行完成。")

# Global configurations (can be overridden or moved into function if preferred)
IMG_SIZE = 224 # REDUCED for SVM/KNN on raw pixels to manage computation. 224 is too large.
BATCH_SIZE = 32 # For DataLoader feature extraction phase
NUM_CLASSES = 9 # Number of rock categories

if __name__ == '__main__':
    # Note: The Path(__file__) logic for data_dir might need adjustment
    # based on your actual project directory structure.
    # If 'Rock Data' is in the same directory as this script, 
    # then data_dir = Path("./Rock Data") is simpler.
    
    # Simplified check for Rock Data in current directory for broader compatibility
    if not os.path.isdir("./Rock Data") and not (Path(__file__).resolve().parent.parent.parent.parent / 'Rock Data').is_dir() :
         print("错误: 'Rock Data' 文件夹在预期路径未找到。请确保数据集已正确放置。")
    else:
        overall_start_time = time.time()
        run_primary_goal_svm()
        overall_end_time = time.time()
        print(f"SVM初级目标总执行时间: {(overall_end_time - overall_start_time)/60:.2f} 分钟")

