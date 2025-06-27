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
IMG_WIDTH = 600
IMG_HEIGHT = 600
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 0.01
NUM_CLASSES = 9
TRIPLET_KERNEL_SIZE = 7

class TripletAttention(layers.Layer):
    "三元组注意力模块，用于捕获特征图的空间和通道维度之间的交互作用"
    def __init__(self, kernel_size=7, **kwargs):
        super(TripletAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.spatial_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False, name=self.name + '_spatial_conv')
        self.ch_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False, name=self.name + '_ch_conv')
        self.cw_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid', use_bias=False, name=self.name + '_cw_conv')
        super(TripletAttention, self).build(input_shape)

    def call(self, x):
        "计算三元组注意力并返回加权后的特征图"
        max_pool_c = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool_c = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_z_pool = tf.concat([max_pool_c, avg_pool_c], axis=-1)
        attn_spatial = self.spatial_conv(spatial_z_pool)
        out_spatial = x * attn_spatial

        x_ch_perm = tf.transpose(x, perm=[0, 3, 1, 2])
        max_p_ch = tf.reduce_max(x_ch_perm, axis=3, keepdims=True)
        avg_p_ch = tf.reduce_mean(x_ch_perm, axis=3, keepdims=True)
        z_p_ch = tf.concat([max_p_ch, avg_p_ch], axis=3)
        attn_ch_map_intermediate = self.ch_conv(z_p_ch)
        attn_ch_map_perm = tf.transpose(attn_ch_map_intermediate, perm=[0, 2, 3, 1])
        out_ch = x * attn_ch_map_perm

        x_cw_perm = tf.transpose(x, perm=[0, 3, 2, 1])
        max_p_cw = tf.reduce_max(x_cw_perm, axis=3, keepdims=True)
        avg_p_cw = tf.reduce_mean(x_cw_perm, axis=3, keepdims=True)
        z_p_cw = tf.concat([max_p_cw, avg_p_cw], axis=3)
        attn_cw_map_intermediate = self.cw_conv(z_p_cw)
        attn_cw_map_perm = tf.transpose(attn_cw_map_intermediate, perm=[0, 3, 2, 1])
        out_cw = x * attn_cw_map_perm

        return (out_spatial + out_ch + out_cw) / 3.0

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

def build_triplet_efficientnet_b7(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, triplet_kernel_size=TRIPLET_KERNEL_SIZE):
    "构建一个修改版的EfficientNet-B7模型，将SE模块替换为三元组注意力模块"
    print("正在构建三元组EfficientNet-B7...")
    base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
    model_input = base_model.input
    tensors_map = {model_input.name: model_input}
    for layer_idx, layer in enumerate(base_model.layers[1:]):
        layer_input_node_names = []
        for node_obj in layer._inbound_nodes:
            current_inbound_layers_or_layer = node_obj.inbound_layers
            actual_inbound_layers_list = []
            if isinstance(current_inbound_layers_or_layer, list):
                actual_inbound_layers_list = current_inbound_layers_or_layer
            else:
                actual_inbound_layers_list = [current_inbound_layers_or_layer]
            for inbound_layer in actual_inbound_layers_list:
                if inbound_layer.name not in layer_input_node_names:
                     layer_input_node_names.append(inbound_layer.name)
        current_input_tensors = [tensors_map[name] for name in layer_input_node_names]
        if len(current_input_tensors) == 1:
            current_input_tensor_for_layer = current_input_tensors[0]
        else:
            current_input_tensor_for_layer = current_input_tensors
        if isinstance(layer, tf.keras.layers.Multiply) and "se_excite" in layer.name:
            if not isinstance(current_input_tensor_for_layer, list) or len(current_input_tensor_for_layer) != 2:
                print(f"警告：SE层 {layer.name} 未按预期具有2个输入，跳过修改。输入数量: {len(current_input_tensor_for_layer) if isinstance(current_input_tensor_for_layer, list) else '非列表'}")
                x = layer(current_input_tensor_for_layer)
            else:
                feature_map_tensor = current_input_tensor_for_layer[0]
                block_prefix = layer.name.split("_se_excite")[0]
                triplet_attn_layer = TripletAttention(kernel_size=triplet_kernel_size, name=f"{block_prefix}_triplet_attention")
                attention_output = triplet_attn_layer(feature_map_tensor)
                x = attention_output
        else:
            x = layer(current_input_tensor_for_layer)
        tensors_map[layer.name] = x
    modified_base_output = tensors_map[base_model.layers[-1].name]
    head = layers.Conv2D(filters=1024, kernel_size=(1,1), padding="same", activation=tf.nn.swish, name="head_conv1x1")(modified_base_output)
    head = layers.GlobalAveragePooling2D(name="head_gap")(head)
    head = layers.Dense(512, activation=tf.nn.swish, name="head_fc")(head)
    output_softmax = layers.Dense(num_classes, activation="softmax", name="head_softmax")(head)
    model = tf.keras.Model(inputs=model_input, outputs=output_softmax, name="Triplet_EfficientNetB7_RockClassifier")
    print("三元组EfficientNet-B7模型构建完成。")
    return model

def parse_image(filename):
    "解析图像文件并调整大小"
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image

def augment_image(image, label):
    "对图像应用九种数据增强技术"
    image = tfa.image.rotate(image, angles=tf.random.uniform(shape=[], minval=-np.pi/6, maxval=np.pi/6, dtype=tf.float32), interpolation='BILINEAR')
    if tf.random.uniform(()) > 0.5:
        img_shape = tf.shape(image)
        h, w = img_shape[0], img_shape[1]
        num_image_channels = image.shape[-1]
        if num_image_channels is None:
            num_image_channels = tf.shape(image)[-1]
        noise_ratio = 0.02
        total_pixels_per_channel = tf.cast(h * w, tf.float32)
        num_noise_pixels_on_channel = tf.cast(total_pixels_per_channel * noise_ratio, tf.int32)
        num_salt_pixels = num_noise_pixels_on_channel // 2
        num_pepper_pixels = num_noise_pixels_on_channel - num_salt_pixels
        processed_channels = []
        channels_list = tf.unstack(image, axis=-1)
        for channel_idx in range(num_image_channels):
            current_channel = channels_list[channel_idx]
            if num_salt_pixels > 0:
                salt_h_indices = tf.random.uniform(shape=[num_salt_pixels], minval=0, maxval=h, dtype=tf.int32)
                salt_w_indices = tf.random.uniform(shape=[num_salt_pixels], minval=0, maxval=w, dtype=tf.int32)
                salt_coords = tf.stack([salt_h_indices, salt_w_indices], axis=1)
                salt_updates = tf.ones(shape=[num_salt_pixels], dtype=image.dtype)
                current_channel = tf.tensor_scatter_nd_update(current_channel, salt_coords, salt_updates)
            if num_pepper_pixels > 0:
                pepper_h_indices = tf.random.uniform(shape=[num_pepper_pixels], minval=0, maxval=h, dtype=tf.int32)
                pepper_w_indices = tf.random.uniform(shape=[num_pepper_pixels], minval=0, maxval=w, dtype=tf.int32)
                pepper_coords = tf.stack([pepper_h_indices, pepper_w_indices], axis=1)
                pepper_updates = tf.zeros(shape=[num_pepper_pixels], dtype=image.dtype)
                current_channel = tf.tensor_scatter_nd_update(current_channel, pepper_coords, pepper_updates)
            processed_channels.append(current_channel)
        if processed_channels:
             image = tf.stack(processed_channels, axis=-1)
    image = tf.image.random_brightness(image, max_delta=0.2)
    if tf.random.uniform(()) > 0.5:
        crop_fraction = tf.random.uniform((), 0.7, 0.95)
        image = tf.image.central_crop(image, central_fraction=crop_fraction)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        noise_stddev_normalized = 10.0 / 255.0
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev_normalized, dtype=image.dtype)
        image = image + noise
        image = tf.clip_by_value(image, 0.0, 1.0)
    max_translate_x = IMG_WIDTH // 10
    max_translate_y = IMG_HEIGHT // 10
    translations_x = tf.random.uniform((), -tf.cast(max_translate_x, tf.float32), tf.cast(max_translate_x, tf.float32))
    translations_y = tf.random.uniform((), -tf.cast(max_translate_y, tf.float32), tf.cast(max_translate_y, tf.float32))
    image = tfa.image.translate(image, translations=[translations_x, translations_y], interpolation='BILINEAR')
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def load_dataset(subset_dir, class_names_list_arg):
    "加载数据集并创建tf.data.Dataset"
    image_paths = list(subset_dir.glob('*/*.jpg')) + list(subset_dir.glob('*/*.jpeg')) + list(subset_dir.glob('*/*.png'))
    image_paths_str = [str(p) for p in image_paths]
    if not image_paths_str:
        raise ValueError(f"在 {subset_dir} 中未找到图像，请检查数据集结构。")
    labels = [class_names_list_arg.index(pathlib.Path(p).parent.name) for p in image_paths_str]
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(class_names_list_arg))
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths_str)
    image_ds = path_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels_one_hot)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset

# --- 主脚本 ---
if __name__ == '__main__':
    "主程序入口，配置GPU、加载数据、训练与评估模型"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"找到并配置了 {len(gpus)} 个物理GPU, {len(logical_gpus)} 个逻辑GPU。")
            print("TensorFlow 将尝试使用 GPU 进行训练。")
        except RuntimeError as e:
            print(f"设置GPU内存按需增长时发生错误: {e}")
    else:
        print("未找到可用的GPU。TensorFlow 将使用 CPU 进行训练。")
    if not data_dir.exists():
        print(f"错误：未找到数据目录 '{data_dir}'，请确保其与脚本在同一目录下或路径正确。")
        exit()
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    test_dir = data_dir / 'test'
    if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
        print(f"错误：在 '{data_dir}' 中未找到train、valid或test子目录。")
        exit()
    class_names_from_data = sorted([item.name for item in train_dir.glob('*') if item.is_dir()])
    if not class_names_from_data:
        print(f"错误：在训练目录 '{train_dir}' 中未找到任何类别文件夹。")
        exit()
    if len(class_names_from_data) != NUM_CLASSES:
         print(f"警告：从数据目录检测到 {len(class_names_from_data)} 个类别 ({class_names_from_data})，但初始 NUM_CLASSES 设置为 {NUM_CLASSES}。")
         print(f"脚本将使用从数据目录检测到的类别数量 ({len(class_names_from_data)}) 和名称进行后续操作。")
         NUM_CLASSES = len(class_names_from_data)
    CLASS_NAMES_LIST = class_names_from_data
    print(f"类别名称：{CLASS_NAMES_LIST}（共 {NUM_CLASSES} 个）")
    print("正在加载数据集...")
    try:
        train_dataset_raw = load_dataset(train_dir, CLASS_NAMES_LIST)
        valid_dataset_raw = load_dataset(valid_dir, CLASS_NAMES_LIST)
        test_dataset_raw = load_dataset(test_dir, CLASS_NAMES_LIST)
    except ValueError as e:
        print(e)
        exit()
    def preprocess_and_augment_train(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return augment_image(image, label)
    def preprocess_eval(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label
    train_dataset_cardinality = tf.data.experimental.cardinality(train_dataset_raw).numpy()
    shuffle_buffer_size = 1000
    if train_dataset_cardinality > 0 and train_dataset_cardinality < 1000:
        shuffle_buffer_size = train_dataset_cardinality
    train_dataset = train_dataset_raw.map(preprocess_and_augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset_raw.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset_for_eval = test_dataset_raw.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset_for_eval = test_dataset_for_eval.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    model = build_triplet_efficientnet_b7(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, triplet_kernel_size=TRIPLET_KERNEL_SIZE)
    model.summary(line_length=150)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("\n开始模型训练...")
    if tf.data.experimental.cardinality(train_dataset).numpy() == 0:
        print("错误：训练数据集为空。请检查数据加载和路径。")
        exit()
    if tf.data.experimental.cardinality(valid_dataset).numpy() == 0:
        print("错误：验证数据集为空。请检查数据加载和路径。")
        exit()
    tensorboard_log_dir = current_file.parent / 'logs_rock_classifier'
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard 日志将保存在: {tensorboard_log_dir.resolve()}")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), histogram_freq=1)
        ]
    )
    print("\n训练完成。")
    print("\n在测试集上评估模型...")
    if tf.data.experimental.cardinality(test_dataset_for_eval).numpy() == 0:
        print("错误：测试数据集为空。请检查数据加载和路径。")
        exit()
    loss, accuracy = model.evaluate(test_dataset_for_eval)
    print(f"测试损失：{loss:.4f}")
    print(f"测试准确率：{accuracy:.4f}")
    print("\n正在生成分类报告和混淆矩阵...")
    y_pred_probs = model.predict(test_dataset_for_eval)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_list_from_batched_eval = []
    for _, labels_batch in test_dataset_for_eval:
        y_true_list_from_batched_eval.extend(np.argmax(labels_batch.numpy(), axis=1))
    final_y_true = np.array(y_true_list_from_batched_eval)
    if len(final_y_true) != len(y_pred_classes):
        print(f"警告：真实标签 ({len(final_y_true)}) 和预测标签 ({len(y_pred_classes)}) 长度不匹配。")
        min_len = min(len(final_y_true), len(y_pred_classes))
        if min_len > 0:
            final_y_true = final_y_true[:min_len]
            y_pred_classes = y_pred_classes[:min_len]
            print(f"报告将基于前 {min_len} 个样本生成。")
        else:
            final_y_true = None
            print("错误：无法匹配真实标签和预测标签的数量以生成报告。")
    if final_y_true is not None and len(final_y_true) > 0:
        print("\n分类报告：")
        print(classification_report(final_y_true, y_pred_classes, target_names=CLASS_NAMES_LIST, zero_division=0))
        print("\n混淆矩阵：")
        cm = confusion_matrix(final_y_true, y_pred_classes)
        plt.figure(figsize=(max(10, NUM_CLASSES), max(8, int(NUM_CLASSES*0.8))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES_LIST,
                    yticklabels=CLASS_NAMES_LIST)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        confusion_matrix_path = current_file.parent / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        print(f"混淆矩阵已保存到 {confusion_matrix_path}")
    else:
        print("由于真实标签和预测标签的长度不匹配或为空，无法生成分类报告和/或混淆矩阵。")
    model_save_path = current_file.parent / "rock_classification_triplet_efficientnet_b7_model.keras"
    model.save(model_save_path)
    print(f"\n完整模型已保存到 {model_save_path}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    training_history_path = current_file.parent / "training_history.png"
    plt.savefig(training_history_path)
    print(f"训练历史图像已保存到 {training_history_path}")
    plt.show()
    
    print("\n脚本成功完成。")
