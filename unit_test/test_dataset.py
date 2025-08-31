import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from engine.config import Config
from engine.parsers.dataset_parser import parse_dataset

config_path = "configs/sod_detr/sod_detr_n_500e_coco.py"
cfg = Config.fromfile(config_path)

training_dataset_cfg = cfg.training_dataset
training_dataset = parse_dataset(training_dataset_cfg.name, training_dataset_cfg.config)

test_dataset_cfg = cfg.test_dataset
test_dataset = parse_dataset(test_dataset_cfg.name, test_dataset_cfg.config)

# from engine.config import Config
# from engine.dataset.yolo_dataset import YOLODataset
# from engine.utils.plotting import plot_images

# path = "/datasets/coco/train2017.txt"

# dataset = YOLODataset(path, augment=True, batch_size=16)

# no_augment = [
#     dict(
#         name="LetterBox",
#         config=dict(
#             new_shape=(640, 640),
#         ),
#     ),
#     dict(
#         name="Format",
#         config=dict(
#             bbox_format="xywh",
#             normalize=True,
#             return_mask=False,
#             return_keypoint=False,
#             return_obb=False,
#             mask_ratio=4,
#             mask_overlap=True,
#             batch_idx=True,
#             bgr=0.0,
#         ),
#     ),
# ]

# test_pipeline = Config(dict(augmentlist = no_augment))
# path = "/datasets/coco/val2017.txt"
# dataset = YOLODataset(path, augment=False, augment_cfg=test_pipeline.augmentlist, rect=True)

