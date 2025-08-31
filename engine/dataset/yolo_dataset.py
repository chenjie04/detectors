"""
YOLO格式数据集类
"""

import os
import glob
import math
from pathlib import Path
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Union, Tuple
from itertools import repeat
import random

import cv2
from torch.utils.data import Dataset
import numpy as np

from engine.config import Config
from engine.dataset.augment_parser import parse_augment
from engine.dataset.utils import (
    FORMATS_HELP_MSG,
    HELP_URL,
    IMG_FORMATS,
    check_file_speeds,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    get_hash,
    verify_image_label,
)
from engine.dataset.augment import Compose
from engine.utils import LOGGER, LOCAL_RANK, TQDM, NUM_THREADS
from engine.utils.instance import Instances
from engine.utils.ops import resample_segments
from engine.utils.patches import imread

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"

default_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
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
default_augment_cfg = [
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

no_augment = [
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

default_augment = Config(dict(augmentlist=default_augment_cfg))


class YOLODataset(Dataset):
    def __init__(
        self,
        img_path: Union[str, List[str]],
        metainfo: Dict[str, Any] = dict(classes=default_classes),
        imgsz: int = 640,
        cache: Union[bool, str] = "disk",
        augment: bool = True,
        augment_cfg: List[Dict] = default_augment.augmentlist,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        sub_classes: Optional[List[int]] = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """
        Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | List[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            augment_cfg (Dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used. 多GPU训练不支持，所以只在验证的时候设为True
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            sub_classes (List[int], optional): List of included sub_classes. 意思是只选择特定的类别进行训练
            fraction (float): Fraction of dataset to utilize.
            channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.img_path = img_path
        self.metainfo = metainfo
        self.imgsz = imgsz
        self.augment = augment
        self.augment_cfg = augment_cfg
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.single_cls = single_cls # 单类训练，有一些应用只在乎位置不在乎类别
        self.im_files = self.get_img_files(self.img_path)  # 图片路径列表
        self.labels = self.get_labels()
        self.update_labels(include_class=sub_classes)
        self.num_img = len(self.labels)  # number of images
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad

        self.rect = rect
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = (
            min((self.num_img, self.batch_size * 8, 1000)) if self.augment else 0
        )

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.num_img,
            [None] * self.num_img,
            [None] * self.num_img,
        )
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = (
            cache.lower()
            if isinstance(cache, str)
            else "ram" if cache is True else None
        )
        if self.cache == "ram" and self.check_cache_ram():
            LOGGER.warning(
                "cache='ram' may produce non-deterministic training results. "
                "Consider cache='disk' as a deterministic alternative if your disk space allows."
            )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(augment_cfg=augment_cfg)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.labels)

    def get_labels(self) -> List[Dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(
                self.label_files + self.im_files
            )  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = (
            (len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels
        )
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(
                f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        return labels

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(len(self.metainfo["classes"])),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_img_files(self, img_path: Union[str, List[str]]) -> List[str]:
        """
        Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x
                            for x in t
                        ]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(
                x.replace("/", os.sep)
                for x in f
                if x.rpartition(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert (
                im_files
            ), f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e
        if self.fraction < 1:
            im_files = im_files[
                : round(len(im_files) * self.fraction)
            ]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: Optional[List[int]]) -> None:
        """
        Update labels to include only specified classes.

        Args:
            include_class (List[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [
                        segments[si] for si, idx in enumerate(j) if idx
                    ]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """
        Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.num_img, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = (
            b * self.num_img / n * (1 + safety_margin)
        )  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """
        Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.num_img, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(
                    f"{self.prefix}Skipping caching images to disk, directory not writeable"
                )
                return False
        disk_required = (
            b * self.num_img / n * (1 + safety_margin)
        )  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (
            (self.cache_images_to_disk, "Disk")
            if self.cache == "disk"
            else (self.load_image, "RAM")
        )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.num_img))
            pbar = TQDM(enumerate(results), total=self.num_img, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                        x  # im, hw_orig, hw_resized = load_image(self, i)
                    )
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def load_image(
        self, i: int, rect_mode: bool = True
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (Tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (Tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(
                        f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}"
                    )
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (
                h0 == w0 == self.imgsz
            ):  # resize by stretching image to square imgsz
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.buffer.append(i)
                if (
                    1 < len(self.buffer) >= self.max_buffer_length
                ):  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def get_image_and_label(self, index: int) -> Dict[str, Any]:
        """
        Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (Dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(
            self.labels[index]
        )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(
            index
        )
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (
                (max_len + 1) if segment_resamples < max_len else segment_resamples
            )
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(
                resample_segments(segments, n=segment_resamples), axis=0
            )
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(
            bboxes,
            segments,
            keypoints=None,
            bbox_format=bbox_format,
            normalized=normalized,
        )
        return label

    def build_transforms(self, augment_cfg: Dict) -> Compose:
        """
        Build and return a Compose object for image transformations.

        Args:
            augment_cfg (Dict): Augmentation configuration dictionary.

        Returns:
            Compose: Composed image transformations.
        """
        augments = []
        for augment in augment_cfg:
            if "dataset" in augment.config:
                augment.config["dataset"] = self
            if "pre_transform" in augment.config and isinstance(
                augment.config["pre_transform"], list
            ):
                for i in range(len(augment.config["pre_transform"])):
                    if "dataset" in augment.config["pre_transform"][i].config:
                        augment.config["pre_transform"][i].config["dataset"] = self
            augments.append(parse_augment(augment.name, augment.config))
        return Compose(augments)

    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.num_img) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image