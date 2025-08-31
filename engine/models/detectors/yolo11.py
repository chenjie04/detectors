"""YOLO11 模型的基础模块导入.

本模块实现了YOLO11目标检测模型,包含backbone、neck和head三个主要组件.
主要依赖PyTorch的nn模块来构建神经网络层.
"""
from typing import List, Union, Tuple
import torch
import torch.nn as nn


from engine.models.module_parser import parse_module
from engine.config import Config

class YOLO11(nn.Module):
    """YOLO11目标检测模型类.

    该类实现了YOLO11目标检测模型的主体架构,包含以下三个主要组件:
        1. backbone: 主干网络,用于提取图像特征
        2. neck: 特征融合网络,用于融合不同尺度的特征
        3. head: 检测头,用于生成最终的检测结果
    """
    def __init__(self, backbone: Config, neck: Config, bbox_head: Config):
        super().__init__()

        # 解析backbone
        self.backbone = parse_module(backbone.name, backbone.config)

        # 解析neck
        self.neck = parse_module(neck.name, neck.config)

        # 解析head
        self.head = parse_module(bbox_head.name, bbox_head.config)

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], Tuple]:
        """Forward function"""
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)
