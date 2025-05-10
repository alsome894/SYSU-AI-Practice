
Rock clasfication - v1 2024-02-04 12:45pm
==============================

This dataset was exported via roboflow.com on February 4, 2024 at 6:49 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 4212 images.
Rocks-life-detection are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Salt and pepper noise was applied to 0.1 percent of pixels

岩石分类 - v1 2024-02-04 12:45pm
本数据集于2024年2月4日18:49 GMT通过roboflow.com导出

Roboflow是一个端到端的计算机视觉平台，可帮助您：

与团队合作开发计算机视觉项目

收集和组织图像

理解和搜索非结构化图像数据

标注和创建数据集

导出、训练和部署计算机视觉模型

使用主动学习持续改进数据集

该数据集包含4212张图像，岩石类型通过文件夹结构进行标注。

预处理流程：

自动调整图像方向（去除EXIF方向信息）

统一缩放到640x640（非等比例拉伸）

数据增强策略（为每张原始图像创建3个增强版本）：

50%概率水平翻转

随机裁剪0-20%的图像区域

随机水平/垂直剪切（±10度）

随机亮度调整（±15%）

随机曝光调整（±10%）

添加椒盐噪声（影响0.1%的像素）



