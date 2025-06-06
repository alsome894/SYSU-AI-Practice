import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # KNN is sensitive to feature scaling
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
import joblib

# Helper function to extract features and labels (same as in SVM version)
def extract_features_labels(loader: DataLoader, img_size: int):
    all_features_list = []
    all_labels_list = []
    expected_feature_dim = 3 * img_size * img_size
    print(f"Extracting features from {len(loader.dataset)} images (expected dim: {expected_feature_dim})...")
    for inputs, labels in tqdm(loader, desc="Extracting features"):
        inputs_cpu = inputs.cpu()
        current_batch_size = inputs_cpu.shape[0]
        features = inputs_cpu.view(current_batch_size, -1).numpy()
        all_features_list.append(features)
        all_labels_list.append(labels.cpu().numpy())
    if not all_features_list:
        return np.array([]).reshape(0, expected_feature_dim), np.array([])
    all_features_np = np.concatenate(all_features_list, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    return all_features_np, all_labels_np

def run_primary_goal_knn():
    print("开始初级目标：使用KNN进行岩石分类...")
    print("警告：使用高维度原始像素作为特征 ({}x{}x3) 可能导致训练缓慢、高内存使用和KNN性能下降（维度灾难）。".format(IMG_SIZE, IMG_SIZE))
    print("注意：KNN对特征缩放敏感，将使用StandardScaler。")

    # 1. Configuration
    # (Similar path logic as SVM version)
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent 
        data_dir = project_root / 'Rock Data'
        if not data_dir.is_dir():
            print(f"警告: 路径 {data_dir} 未找到。尝试备用路径 './Rock Data'")
            data_dir = Path("./Rock Data")
    except NameError:
         print("警告: __file__ 未定义。使用 './Rock Data' 作为数据目录。")
         data_dir = Path("./Rock Data")

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        print(f"错误: 训练目录 ({train_dir}) 或测试目录 ({test_dir}) 未找到。请检查路径。")
        return
        
    print(f"使用训练数据目录: {train_dir}")
    print(f"使用测试数据目录: {test_dir}")

    # 2. Data Preprocessing and Transforms (Normalization included in PyTorch transforms)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # 3. Load Datasets
    train_dataset = datasets.ImageFolder(str(train_dir), transform=data_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=data_transforms)

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    class_names = train_dataset.classes
    print(f"岩石类别: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"警告: 找到 {len(class_names)} 个类别，但 NUM_CLASSES 设置为 {NUM_CLASSES}")

    # 5. Extract Features and Labels
    print("为训练集提取特征...")
    X_train, y_train = extract_features_labels(train_loader, IMG_SIZE)
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")

    print("\n为测试集提取特征...")
    X_test, y_test = extract_features_labels(test_loader, IMG_SIZE)
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")

    if X_train.size == 0 or X_test.size == 0:
        print("错误：未能从一个或多个数据集中提取特征。请检查数据集和路径。")
        return

    # 6. Train KNN Model (with feature scaling)
    # KNN is sensitive to feature scaling, so we use StandardScaler.
    # The ToTensor already scales to [0,1] and Normalize standardizes based on ImageNet stats.
    # For raw pixel values, an additional StandardScaler on the flattened features is often beneficial for KNN.
    print("\n开始训练KNN模型 (带StandardScaler)...")
    
    # Create a pipeline for scaling and KNN
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()), # Scale features before KNN
        ('knn', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)) # p=2 is Euclidean distance
    ])
    
    start_train_time = time.time()
    knn_pipeline.fit(X_train, y_train)
    end_train_time = time.time()
    print(f"KNN模型训练完成。耗时: {(end_train_time - start_train_time):.2f} 秒")

    # 7. Evaluate Model on Test Set
    print("\n在测试集上评估KNN模型...")
    start_eval_time = time.time()
    y_pred_test = knn_pipeline.predict(X_test)
    end_eval_time = time.time()
    print(f"KNN模型评估完成。耗时: {(end_eval_time - start_eval_time):.2f} 秒")

    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 8. Classification Report and Confusion Matrix
    print("\n分类报告 (测试集):")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

    print("\n混淆矩阵 (测试集):")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    ax.set_title('Primary Goal - KNN Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig("primary_goal_knn_confusion_matrix.png")
    print("KNN混淆矩阵图已保存为 primary_goal_knn_confusion_matrix.png")
    # plt.show()

    # 9. Save KNN Model (the entire pipeline)
    model_path = "knn_rock_classifier_pipeline.joblib"
    joblib.dump(knn_pipeline, model_path)
    print(f"\nKNN模型 (带scaler的pipeline) 已保存到: {model_path}")
    
    print("\n基于KNN的初级目标执行完成。")

# Global configurations (consistent with SVM version)
IMG_SIZE = 224 # REDUCED for SVM/KNN on raw pixels.
BATCH_SIZE = 32
NUM_CLASSES = 9

if __name__ == '__main__':
    if not os.path.isdir("./Rock Data") and not (Path(__file__).resolve().parent.parent.parent.parent / 'Rock Data').is_dir() :
         print("错误: 'Rock Data' 文件夹在预期路径未找到。请确保数据集已正确放置。")
    else:
        overall_start_time = time.time()
        run_primary_goal_knn()
        overall_end_time = time.time()
        print(f"KNN初级目标总执行时间: {(overall_end_time - overall_start_time)/60:.2f} 分钟")
