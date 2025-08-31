from typing import List, Tuple, Union, Sequence

import math
import torch
import torch.nn as nn

from engine.models.building_blocks import Conv, DWConv, DFL
from engine.models.utils.decode_utils import make_anchors, dist2bbox


class YOLO11Head(nn.Module):
    """YOLO Detect head for object detection models."""
    export = False  # export mode
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(
        self,
        num_classes: int = 80,
        channels: Sequence[int] = (128, 128, 256),
        stride: Sequence[int] = (8, 16, 32),
    ):
        """
        Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            num_classes (int): Number of classes.
            channels (Sequence[int]): List of channel sizes from backbone feature maps.
            stride (Sequence[int]): Strides of the feature maps.
        """
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.num_levels = len(
            channels
        )  # number of feature levels used to detect objects
        self.reg_max = (
            16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x) ??
        )
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        # self.stride = torch.zeros(self.num_levels)  # strides computed during build
        self.stride = stride  # 直接指定吧
        c2, c3 = max((16, channels[0] // 4, self.reg_max * 4)), max(
            channels[0], min(self.num_classes, 100)
        )  # channels

        self.cls_branch = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.num_classes, 1),
            )
            for x in channels
        )
        self.reg_branch = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in channels
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        """Forward pass of the YOLO detection head.

        Args:
            x (List[torch.Tensor]): List of feature maps from neck network,
                                   each with shape (batch_size, channels[i], Hi, Wi)

        Returns:
            Union[List[torch.Tensor], Tuple]:
                - If training: Returns list of raw predictions from each detection layer
                - If inference: Returns tuple of (decoded predictions, raw predictions) if not exporting,
                              or just decoded predictions if exporting
        """
        raw_pred = []
        for i in range(self.num_levels):
            raw_pred.append(torch.cat((self.reg_branch[i](x[i]), self.cls_branch[i](x[i])), 1)) 
        if self.training:  # Training path
            return raw_pred
        decoded_pred = self._inference(raw_pred)
        return decoded_pred if self.export else (decoded_pred, raw_pred)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.reg_branch, m.cls_branch, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(
                5 / m.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)

    def _inference(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        self.anchors, self.strides = (
            x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
        )

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = (
            self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        )

        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True
    ) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)
