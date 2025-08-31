"""
将配置解析成实例
"""

from engine.dataset.augment import (
    Compose,
    Mosaic,
    RandomPerspective,
    CopyPaste,
    MixUp,
    CutMix,
    Albumentations,
    LetterBox,
    RandomHSV,
    RandomFlip,
    Format,
)



def parse_augment(augment_name: str, augment_cfg: dict):
    """将增强配置解析成实例"""
    if augment_name == "Mosaic":
        return Mosaic(**augment_cfg)
    elif augment_name == "RandomPerspective" and augment_cfg.pre_transform is None:
        return RandomPerspective(**augment_cfg)
    elif augment_name == "RandomPerspective" and isinstance(
        augment_cfg.pre_transform, dict
    ):
        augment_cfg.pre_transform = parse_augment(
            augment_cfg.pre_transform["name"], augment_cfg.pre_transform["config"]
        )
        return RandomPerspective(**augment_cfg)
    elif augment_name == "CopyPaste" and augment_cfg.pre_transform is None:
        return CopyPaste(**augment_cfg)
    elif augment_name == "CopyPaste" and isinstance(augment_cfg.pre_transform, list):
        transforms = []
        for _, transform in enumerate(augment_cfg.pre_transform):
            transforms.append(parse_augment(transform["name"], transform["config"]))
        augment_cfg.pre_transform = Compose(transforms)
        return CopyPaste(**augment_cfg)
    elif augment_name == "MixUp" and augment_cfg.pre_transform is None:
        return MixUp(**augment_cfg)
    elif augment_name == "MixUp" and isinstance(augment_cfg.pre_transform, list):
        transforms = []
        for _, transform in enumerate(augment_cfg.pre_transform):
            transforms.append(parse_augment(transform["name"], transform["config"]))
        augment_cfg.pre_transform = Compose(transforms)
        return MixUp(**augment_cfg)
    elif augment_name == "CutMix" and augment_cfg.pre_transform is None:
        return CutMix(**augment_cfg)
    elif augment_name == "CutMix" and isinstance(augment_cfg.pre_transform, list):
        transforms = []
        for _, transform in enumerate(augment_cfg.pre_transform):
            transforms.append(parse_augment(transform["name"], transform["config"]))
        augment_cfg.pre_transform = Compose(transforms)
        return CutMix(**augment_cfg)
    elif augment_name == "Albumentations":
        return Albumentations(**augment_cfg)
    elif augment_name == "LetterBox":
        return LetterBox(**augment_cfg)
    elif augment_name == "RandomHSV":
        return RandomHSV(**augment_cfg)
    elif augment_name == "RandomFlip":
        return RandomFlip(**augment_cfg)
    elif augment_name == "Format":
        return Format(**augment_cfg)
    else:
        raise ValueError(f"Unknown augment name: {augment_name}")


