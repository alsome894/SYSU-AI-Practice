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
    "提取DataLoader中的特征和标签，返回展平后的特征和标签数组"
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

def run_primary_goal_svm():
    "主流程：SVM岩石分类"
    print("开始初级目标：使用SVM进行岩石分类...")
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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    train_dataset = datasets.ImageFolder(str(train_dir), transform=data_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    class_names = train_dataset.classes
    print(f"岩石类别: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"警告: 找到 {len(class_names)} 个类别，但 NUM_CLASSES 设置为 {NUM_CLASSES}")
    print("为训练集提取特征...")
    X_train, y_train = extract_features_labels(train_loader, IMG_SIZE)
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print("\n为测试集提取特征...")
    X_test, y_test = extract_features_labels(test_loader, IMG_SIZE)
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")
    if X_train.size == 0 or X_test.size == 0:
        print("错误：未能从一个或多个数据集中提取特征。请检查数据集和路径。")
        return
    print("\n开始训练SVM模型...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=False)
    start_train_time = time.time()
    svm_model.fit(X_train, y_train)
    end_train_time = time.time()
    print(f"SVM模型训练完成。耗时: {(end_train_time - start_train_time):.2f} 秒")
    print("\n在测试集上评估SVM模型...")
    start_eval_time = time.time()
    y_pred_test = svm_model.predict(X_test)
    end_eval_time = time.time()
    print(f"SVM模型评估完成。耗时: {(end_eval_time - start_eval_time):.2f} 秒")
    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    print("\n分类报告 (测试集):")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))
    print("\n混淆矩阵 (测试集):")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    ax.set_title('Primary Goal - SVM Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig("primary_goal_svm_confusion_matrix.png")
    print("SVM混淆矩阵图已保存为 primary_goal_svm_confusion_matrix.png")
    model_path = "svm_rock_classifier.joblib"
    joblib.dump(svm_model, model_path)
    print(f"\nSVM模型已保存到: {model_path}")
    print("\n基于SVM的初级目标执行完成。")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 9

if __name__ == '__main__':
    if not os.path.isdir("./Rock Data") and not (Path(__file__).resolve().parent.parent.parent.parent / 'Rock Data').is_dir() :
         print("错误: 'Rock Data' 文件夹在预期路径未找到。请确保数据集已正确放置。")
    else:
        overall_start_time = time.time()
        run_primary_goal_svm()
        overall_end_time = time.time()
        print(f"SVM初级目标总执行时间: {(overall_end_time - overall_start_time)/60:.2f} 分钟")

