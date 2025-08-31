YOLO 系列目标检测/分割模型（YOLOv5/YOLOv8/Ultralytics）常用的数据集格式是一种 简化版的 COCO 格式，主要包括：

📦 YOLO 数据集目录结构

常见的目录结构如下：
```bash
datasets/
└── my_dataset/
    ├── images/                 # 图像文件
    │   ├── train/              # 训练集图像
    │   ├── val/                # 验证集图像
    │   └── test/               # （可选）测试集图像
    ├── labels/                 # 标签文件（与 images 同目录结构）
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml               # 数据配置文件
```
🖼️ 图像文件

- 格式：.jpg / .png / .jpeg 都可。

- 存放在 images/{train,val,test}/ 中。

🏷️ 标签文件（YOLO txt 格式）

- 存放在 labels/{train,val,test}/ 中。

- 每张图像对应一个同名 .txt 文件，例如：
    ```bash
    images/train/0001.jpg
    labels/train/0001.txt
    ```

- 标签文件内容（每一行一个目标）：
    ```bash
    class_id x_center y_center width height
    ```

    - class_id：类别索引（从 0 开始编号）
    - x_center：目标中心点 x 坐标（归一化到 [0,1]）
    - y_center：目标中心点 y 坐标（归一化到 [0,1]）
    - width：目标宽度（归一化到 [0,1]）
    - height：目标高度（归一化到 [0,1]）


    ⚠️ 注意：YOLO 使用的是 相对坐标（相对于图像宽高），不是像素坐标。

- 示例（640×480 图像中，一个类别 2 的目标框占图像中心 50% 宽和 30% 高）：
    ```bash
    2 0.5 0.5 0.5 0.3
    ```
# 📑 数据配置文件（data.yaml）

数据集的说明文件，通常位于数据集根目录，例如 my_dataset/data.yaml：
```bash
# train/val/test 路径
train: datasets/my_dataset/images/train
val: datasets/my_dataset/images/val
test: datasets/my_dataset/images/test   # 可选

# 类别数
nc: 3

# 类别名称（顺序对应 class_id）
names: ['cat', 'dog', 'person']
```

# 📊 与 COCO 的区别

- YOLO：简单 txt + yaml，轻量易读。

- COCO：采用 JSON 格式（instances_train2017.json 等），功能更复杂（支持 segmentation、keypoint、caption 等）。

- Ultralytics YOLO 可以自动从 COCO 格式转换到 YOLO 格式。

# ⚡ 总结：
YOLO 数据集由 图像目录 + 标签目录（txt）+ 配置文件（yaml） 组成。
核心是 YOLO txt 标签文件，每行 class x_center y_center w h（相对坐标）。