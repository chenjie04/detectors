# ----------------------------------------------------------------------------------
# SOD-DETR配置文件。

# 此配置文件用于COCO数据集上的SOD-DETR模型训练和评估。
# 包含数据集设置、模型架构、训练策略等完整配置信息。

# 配置定义规则：
#     1. 所有配置均为键值对形式，键为配置名称，值为配置值，以便能够将整个配置解析为字典。
#     2. 值可以为字符串，如果值需要包含多个项且每个项具有键值对的形式，则可以使用字典表示。
#        如果值需要包含多个项且每个项没有对应的键，则可以使用列表表示。
#        如train_pipeline表示训练数据的增强过程，包含加载图像、加载标注、resize、随机翻转、打包输入等多个步骤。
#        不方便为每个增强步骤指定一个键，因为每个模型需要的增强步骤可能不同。
#        因此，为了方便配置，将train_pipeline表示为一个列表，每个元素为一个字典，字典包含增强步骤的类型和参数。
#        如train_pipeline[0]表示加载图像的步骤，包含type为LoadImageFromFile的键值对，用于指定加载图像的方式。
#     3. 注释应当采用单行注释，方便解析过程中将其屏蔽。
# -----------------------------------------------------------------------------------


# dataset settings

pre_transform = [
    dict(
        name="Mosaic",
        config=dict(
            dataset=None,  # 在数据集对象内实例化时，将其替换为self
            imgsz=640,
            p=1.0,  # 应用概率
            n=4,  # 拼接图片数量，如果为了增加单张图片内正样本的数量，还可以设为9
        ),
    ),
    # 复制粘贴增强, 有两种策略
    # 1. flip 水平翻转后和原图叠加，然后删去重叠超过30%的目标
    # 2. mixup 随机选择一张图片，将其和原图叠加，然后删去重叠超过30%的目标，此时pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine])
    dict(
        name="CopyPaste",
        config=dict(
            dataset=None,
            pre_transform=None,
            p=0.5,
            mode="flip",
        ),
    ),
    dict(
        name="RandomPerspective",
        config=dict(
            degrees=0.0,  # 0.0-180.0 在指定的角度范围内随机旋转图像，提高模型识别各种方向物体的能力。
            translate=0.1,  # 0.0-1.0 通过图像尺寸的一小部分在水平和垂直方向上平移图像，帮助学习检测部分可见的物体。
            scale=0.5,  # 0.0-1.0 通过增益因子缩放图像，模拟物体与相机的不同距离。
            shear=0.0,  # -180.0-180.0 按指定的角度错切图像，模仿从不同角度观察物体的效果。
            perspective=0.0,  # 0.0-1.0 对图像应用随机透视变换，增强模型理解 3D 空间中物体的能力。
            border=(0, 0),  # Tuple specifying mosaic border (top/bottom, left/right).
            pre_transform=dict(
                name="LetterBox",
                config=dict(
                    new_shape=(640, 640),
                ),
            ),
        ),
    ),
]
training_pipeline = [
    *pre_transform,
    dict(name="MixUp", config=dict(dataset=None, pre_transform=pre_transform, p=0.0)),
    dict(
        name="CutMix",
        config=dict(
            dataset=None,
            pre_transform=pre_transform,
            p=0.0,
            beta=1.0,
            num_areas=3,
        ),
    ),
    dict(name="Albumentations", config=dict(p=1.0)),
    dict(name="RandomHSV", config=dict(hgain=0.5, sgain=0.5, vgain=0.5)),
    dict(name="RandomFlip", config=dict(direction="vertical", p=0.0, flip_idx=None)),
    dict(name="RandomFlip", config=dict(direction="horizontal", p=0.5, flip_idx=None)),
    dict(
        name="Format",
        config=dict(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
            mask_ratio=4,
            mask_overlap=True,
            batch_idx=True,
            bgr=0.0,
        ),
    ),
]

test_pipeline = [
    dict(
        name="LetterBox",
        config=dict(
            new_shape=(640, 640),
        ),
    ),
    dict(
        name="Format",
        config=dict(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
            mask_ratio=4,
            mask_overlap=True,
            batch_idx=True,
            bgr=0.0,
        ),
    ),
]

training_dataset = dict(
    name="YOLODataset",
    config=dict(
        img_path="/datasets/coco/train2017.txt",
        # metainfo=dict(classes=default_classes),  # coco 数据集不必提供
        imgsz=640,
        cache="disk",
        augment=True,
        augment_cfg=training_pipeline,
        rect=False,
        batch_size=16,
    ),
)

test_dataset = dict(
    name="YOLODataset",
    config=dict(
        img_path="/datasets/coco/val2017.txt",
        # metainfo=dict(classes=default_classes),  # coco 数据集不必提供
        imgsz=640,
        cache="disk",
        augment=False,
        augment_cfg=test_pipeline,
        rect=True,
        batch_size=16,
    ),
)

# model setting
deepen_factor = 0.5
widen_factor = 0.25

model = dict(
    name="YOLO11",
    config=dict(
        backbone=dict(
            name="YOLO11Backbone",
            config=dict(
                arch="P5",
                last_stage_out_channels=1024,
                deepen_factor=deepen_factor,
                widen_factor=widen_factor,
                out_indices=(2, 3, 4),
            ),
        ),  # [(128, 80, 80), (128, 40, 40), (256, 20, 20)]
        neck=dict(
            name="YOLO11Neck",
            config=dict(
                in_channels=[128, 128, 256],
                num_blocks=2,
                deepen_factor=deepen_factor,
                c3k=False,
            ),
        ),  # [(128, 80, 80)]
        bbox_head=dict(
            name="YOLO11Head",
            config=dict(
                num_classes=80,
                channels=[128, 128, 256],
                stride=[8, 16, 32],
            ),
        ),
    ),
)

losses = dict(
    loss_cls=dict(
        type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
    ),
    loss_bbox=dict(type="L1Loss", loss_weight=5.0),
    loss_iou=dict(type="GIoULoss", loss_weight=2.0),
)

max_epochs = 500
base_lr = 0.004
interval = 10

trainer = dict(
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    param_scheduler=[
        dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
        dict(
            # use cosine lr from 250 to 500 epoch
            type="CosineAnnealingLR",
            eta_min=base_lr * 0.05,
            begin=max_epochs // 2,
            end=max_epochs,
            T_max=max_epochs // 2,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
    ],
)
