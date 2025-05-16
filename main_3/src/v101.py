import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import pathlib

from pathlib import Path

# --- 配置 ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
data_dir = project_root / 'Rock Data'
IMG_WIDTH = 600  # 图像宽度
IMG_HEIGHT = 600  # 图像高度
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)  # 图像尺寸
BATCH_SIZE = 16  # 批次大小
EPOCHS = 60  # 训练轮数
LEARNING_RATE = 0.01  # 学习率
# 类别数量从文件夹结构推断：Basalt、Chert、Clay、Conglomerate、Diatomite、Gypsum、Olivine-Basalt、Shale-(Mudstone)、Siliceous-Sinter
# 与说明中的7个类别冲突，此处根据数据使用9个类别
NUM_CLASSES = 9  # 类别数量
TRIPLET_KERNEL_SIZE = 7  # 三元组注意力卷积核大小

# --- 1. 三元组注意力模块 ---
class TripletAttention(layers.Layer):
    """
    三元组注意力模块，用于捕获特征图的空间和通道维度之间的交互作用。
    包含三个分支：空间注意力、通道-高度注意力和通道-宽度注意力。
    """
    def __init__(self, kernel_size=7, **kwargs):
        super(TripletAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        # 定义用于生成注意力图的卷积块（为简单起见，权重共享）
        self.conv_block = tf.keras.Sequential([
            layers.Conv2D(filters=1,
                          kernel_size=self.kernel_size,
                          padding='same',
                          activation='sigmoid',
                          use_bias=False),
            # 可选：在Conv2D和Sigmoid之间添加BatchNormalization
        ])

    def build(self, input_shape):
        # 为每个分支定义独立的卷积层
        self.spatial_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False, name=self.name + '_spatial_conv')
        self.ch_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False, name=self.name + '_ch_conv')
        self.cw_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False, name=self.name + '_cw_conv')
        super(TripletAttention, self).build(input_shape)

    def call(self, x):
        """
        计算三元组注意力并返回加权后的特征图。
        """
        # 空间注意力分支 (H-W)
        max_pool_c = tf.reduce_max(x, axis=-1, keepdims=True)  # 通道轴上的最大池化
        avg_pool_c = tf.reduce_mean(x, axis=-1, keepdims=True)  # 通道轴上的平均池化
        spatial_z_pool = tf.concat([max_pool_c, avg_pool_c], axis=-1)  # 拼接池化结果
        attn_spatial = self.spatial_conv(spatial_z_pool)  # 生成空间注意力图
        out_spatial = x * attn_spatial  # 加权特征图

        # 通道-高度注意力分支 (C-H)
        x_ch_perm = tf.transpose(x, perm=[0, 3, 1, 2])  # 调整维度顺序以沿宽度轴池化
        max_p_ch = tf.reduce_max(x_ch_perm, axis=3, keepdims=True)  # 宽度轴最大池化
        avg_p_ch = tf.reduce_mean(x_ch_perm, axis=3, keepdims=True)  # 宽度轴平均池化
        z_p_ch = tf.concat([max_p_ch, avg_p_ch], axis=3)  # 拼接池化结果
        attn_ch_map_intermediate = self.ch_conv(z_p_ch)  # 生成通道-高度注意力图
        attn_ch_map_perm = tf.transpose(attn_ch_map_intermediate, perm=[0, 2, 3, 1])  # 调整维度以匹配输入
        out_ch = x * attn_ch_map_perm  # 加权特征图

        # 通道-宽度注意力分支 (C-W)
        x_cw_perm = tf.transpose(x, perm=[0, 3, 2, 1])  # 调整维度顺序以沿高度轴池化
        max_p_cw = tf.reduce_max(x_cw_perm, axis=3, keepdims=True)  # 高度轴最大池化
        avg_p_cw = tf.reduce_mean(x_cw_perm, axis=3, keepdims=True)  # 高度轴平均池化
        z_p_cw = tf.concat([max_p_cw, avg_p_cw], axis=3)  # 拼接池化结果
        attn_cw_map_intermediate = self.cw_conv(z_p_cw)  # 生成通道-宽度注意力图
        attn_cw_map_perm = tf.transpose(attn_cw_map_intermediate, perm=[0, 3, 2, 1])  # 调整维度以匹配输入
        out_cw = x * attn_cw_map_perm  # 加权特征图
        
        return (out_spatial + out_ch + out_cw) / 3.0  # 平均三个分支的输出

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

# --- 2. 构建修改后的EfficientNet-B7 ---
def build_triplet_efficientnet_b7(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, triplet_kernel_size=TRIPLET_KERNEL_SIZE):
    """
    构建一个修改版的EfficientNet-B7模型，将SE模块替换为三元组注意力模块。
    """
    print("正在构建三元组EfficientNet-B7...")
    # 加载预训练的EfficientNet-B7模型
    base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
    
    model_input = base_model.input
    tensors_map = {model_input.name: model_input}  # 存储每层的输出张量

    # 遍历基础模型的层进行修改
    for layer_idx, layer in enumerate(base_model.layers[1:]):  # 跳过输入层
        layer_input_node_names = []
        for node in layer._inbound_nodes:
            for i in range(len(node.inbound_layers)):
                inbound_layer_name = node.inbound_layers[i].name
                layer_input_node_names.append(inbound_layer_name)
        
        # 获取当前层的输入张量
        current_input_tensors = [tensors_map[name] for name in layer_input_node_names]
        if len(current_input_tensors) == 1:
            current_input_tensor_for_layer = current_input_tensors[0]
        else:
            current_input_tensor_for_layer = current_input_tensors  # 处理如Add、Multiply等层

        # 识别SE块的最终乘法层
        if isinstance(layer, tf.keras.layers.Multiply) and "se_excite" in layer.name:
            if not isinstance(current_input_tensor_for_layer, list) or len(current_input_tensor_for_layer) != 2:
                print(f"警告：SE层 {layer.name} 未按预期具有2个输入，跳过修改。")
                x = layer(current_input_tensor_for_layer)
            else:
                feature_map_tensor = current_input_tensor_for_layer[0]  # 假设第一个输入是特征图
                
                block_prefix = layer.name.split("_se_excite")[0]  # 提取块前缀用于命名
                
                print(f"  将 {layer.name} 中的SE替换为三元组注意力。")
                triplet_attn_layer = TripletAttention(kernel_size=triplet_kernel_size, name=f"{block_prefix}_triplet_attention")
                attention_output = triplet_attn_layer(feature_map_tensor)
                
                # 使用三元组注意力输出替换SE乘法
                x = tf.keras.layers.Multiply(name=layer.name.replace("_se_excite", "_triplet_excite"))([feature_map_tensor, attention_output])
        else:
            # 非SE层保持不变
            x = layer(current_input_tensor_for_layer)
        
        tensors_map[layer.name] = x  # 存储当前层输出

    # 获取修改后的基础模型输出
    modified_base_output = tensors_map[base_model.layers[-1].name]

    # 添加分类头部
    head = layers.Conv2D(filters=1024, kernel_size=(1,1), padding="same", activation=tf.nn.swish, name="head_conv1x1")(modified_base_output)
    head = layers.GlobalAveragePooling2D(name="head_gap")(head)
    head = layers.Dense(512, activation=tf.nn.swish, name="head_fc")(head)
    output_softmax = layers.Dense(num_classes, activation="softmax", name="head_softmax")(head)
    
    model = tf.keras.Model(inputs=model_input, outputs=output_softmax, name="Triplet_EfficientNetB7_RockClassifier")
    print("三元组EfficientNet-B7模型构建完成。")
    return model

# --- 3. 数据增强和加载 ---
def parse_image(filename):
    """
    解析图像文件并调整大小。
    """
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # 调整为600x600
    return image

def augment_image(image, label):
    """
    对图像应用九种数据增强技术。
    """
    # 1. 旋转（随机角度，例如+/-30度）
    image = tfa.image.rotate(image, angles=tf.random.uniform(shape=[], minval=-np.pi/6, maxval=np.pi/6, dtype=tf.float32), interpolation='BILINEAR')
    
    # 2. 椒盐噪声（应用于部分像素）
    if tf.random.uniform(()) > 0.5:  # 50%概率应用
        noise_ratio = 0.02  # 影响2%的像素
        h, w, c = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
        num_noise_pixels = tf.cast(tf.cast(h * w, tf.float32) * noise_ratio, tf.int32)
        
        # 盐噪声（白色像素）
        salt_indices_h = tf.random.uniform(shape=[num_noise_pixels // 2], minval=0, maxval=h, dtype=tf.int32)
        salt_indices_w = tf.random.uniform(shape=[num_noise_pixels // 2], minval=0, maxval=w, dtype=tf.int32)
        salt_updates = tf.ones_like(salt_indices_h, dtype=tf.uint8) * 255
        
        # 椒噪声（黑色像素）
        pepper_indices_h = tf.random.uniform(shape=[num_noise_pixels // 2], minval=0, maxval=h, dtype=tf.int32)
        pepper_indices_w = tf.random.uniform(shape=[num_noise_pixels // 2], minval=0, maxval=w, dtype=tf.int32)
        pepper_updates = tf.zeros_like(pepper_indices_h, dtype=tf.uint8)

        for i in range(c):  # 对每个通道应用
            channel_slice = image[..., i]
            salt_coords = tf.stack([salt_indices_h, salt_indices_w], axis=1)
            channel_slice = tf.tensor_scatter_nd_update(channel_slice, salt_coords, salt_updates)
            pepper_coords = tf.stack([pepper_indices_h, pepper_indices_w], axis=1)
            channel_slice = tf.tensor_scatter_nd_update(channel_slice, pepper_coords, pepper_updates)
            image = tf.tensor_scatter_nd_update(image, [[...,i]], tf.expand_dims(channel_slice, axis=-1))

    # 3 & 4. 增亮/变暗（随机调整亮度）
    image = tf.image.random_brightness(image, max_delta=0.2)  # 亮度变化范围-0.2到0.2
    
    # 5. 放大（随机裁剪和调整大小）
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform((), 0.8, 1.0)
        new_h = tf.cast(tf.cast(IMG_HEIGHT, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(IMG_WIDTH, tf.float32) * scale, tf.int32)
        image_resized_for_crop = tf.image.resize(image, [new_h, new_w])
        image = tf.image.random_crop(image_resized_for_crop, size=[IMG_HEIGHT, IMG_WIDTH, 3])
        image = tf.image.central_crop(image, central_fraction=tf.random.uniform((), 0.7, 0.95))  # 裁剪70-95%
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # 6. 垂直翻转
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        
    # 7. 水平翻转
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        
    # 8. 高斯噪声
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=10.0, dtype=tf.float32)  # 高斯噪声标准差10
        image = tf.cast(image, tf.float32) + noise
        image = tf.clip_by_value(image, 0.0, 255.0)  # 限制像素值范围
        
    # 9. 平移（随机水平和垂直移动）
    max_translate_x = IMG_WIDTH // 10  # 最大10%水平移动
    max_translate_y = IMG_HEIGHT // 10  # 最大10%垂直移动
    translations = [tf.random.uniform((), -max_translate_x, max_translate_x, dtype=tf.float32),
                    tf.random.uniform((), -max_translate_y, max_translate_y, dtype=tf.float32)]
    image = tfa.image.translate(image, translations=translations, interpolation='BILINEAR')

    image = tf.cast(image, tf.uint8)  # 确保类型为uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # 归一化到[0,1]
    return image, label

def load_dataset(subset_dir, class_names):
    """
    加载数据集并创建tf.data.Dataset。
    """
    image_paths = list(subset_dir.glob('*/*.jpg')) + list(subset_dir.glob('*/*.jpeg')) + list(subset_dir.glob('*/*.png'))
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        raise ValueError(f"在 {subset_dir} 中未找到图像，请检查数据集结构。")

    labels = [class_names.index(pathlib.Path(p).parent.name) for p in image_paths]
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset

# --- 主脚本 ---
if __name__ == '__main__':

    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    test_dir = data_dir / 'test'

    if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
        print("错误：未找到train、valid或test子目录。")
        exit()
        
    # 从训练目录推断类别名称
    class_names_path = sorted([item.name for item in train_dir.glob('*') if item.is_dir()])
    if len(class_names_path) != NUM_CLASSES:
         print(f"警告：检测到 {len(class_names_path)} 个类别，但NUM_CLASSES设置为 {NUM_CLASSES}。")
         print(f"检测到的类别：{class_names_path}")
         if len(class_names_path) == 0:
            print("错误：训练目录中未找到类别文件夹。")
            exit()
    CLASS_NAMES = class_names_path
    print(f"类别名称：{CLASS_NAMES}（共 {len(CLASS_NAMES)} 个）")

    print("正在加载数据集...")
    try:
        train_dataset = load_dataset(train_dir, CLASS_NAMES)
        valid_dataset = load_dataset(valid_dir, CLASS_NAMES)
        test_dataset = load_dataset(test_dir, CLASS_NAMES)
    except ValueError as e:
        print(e)
        exit()

    # 准备训练数据集
    train_dataset = train_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)  # 预归一化
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)  # 应用增强
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)  # 打乱、批处理、预取

    # 准备验证和测试数据集
    valid_dataset = valid_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    test_dataset_for_eval = test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset_for_eval = test_dataset_for_eval.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # 构建模型
    model = build_triplet_efficientnet_b7(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES)
    model.summary()

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',  # 交叉熵损失
                  metrics=['accuracy'])  # 准确率指标

    # 训练模型
    print("\n开始模型训练...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ]
    )
    print("\n训练完成。")

    # 评估模型
    print("\n在测试集上评估模型...")
    loss, accuracy = model.evaluate(test_dataset_for_eval)
    print(f"测试损失：{loss:.4f}")
    print(f"测试准确率：{accuracy:.4f}")

    # 生成分类报告和混淆矩阵
    print("\n正在生成分类报告和混淆矩阵...")
    y_pred_probs = model.predict(test_dataset_for_eval)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # 获取真实标签
    y_true_classes = []
    raw_test_labels = []
    for _, labels_batch in test_dataset:
        raw_test_labels.extend(np.argmax(labels_batch.numpy(), axis=1))
    y_true_classes = np.array(raw_test_labels[:len(y_pred_classes)])  # 确保长度匹配

    if len(y_true_classes) != len(y_pred_classes):
         print(f"警告：真实标签（{len(y_true_classes)}）和预测标签（{len(y_pred_classes)}）长度不匹配。")
         if len(y_true_classes) > len(y_pred_classes):
             y_true_classes = y_true_classes[:len(y_pred_classes)]
         else:
             print("由于标签严重不匹配，无法生成报告。")

    print("\n分类报告：")
    if CLASS_NAMES and len(CLASS_NAMES) == NUM_CLASSES:
        print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, zero_division=0))
    else:
        print(classification_report(y_true_classes, y_pred_classes, zero_division=0))

    print("\n混淆矩阵：")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES if CLASS_NAMES and len(CLASS_NAMES) == NUM_CLASSES else range(NUM_CLASSES), 
                yticklabels=CLASS_NAMES if CLASS_NAMES and len(CLASS_NAMES) == NUM_CLASSES else range(NUM_CLASSES))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    print(f"混淆矩阵已保存到 {confusion_matrix_path}")
    # plt.show() # Uncomment to display plot if running in an environment that supports it

    # 保存模型
    model_save_path = "rock_classification_triplet_efficientnet_b7_model.keras"
    model.save(model_save_path)
    print(f"\n完整模型已保存到 {model_save_path}")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    training_history_path = "training_history.png"
    plt.savefig(training_history_path)
    print(f"训练历史图像已保存到 {training_history_path}")
    plt.show() # Uncomment to display plots

    print("\n脚本成功完成。")
