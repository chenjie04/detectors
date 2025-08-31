from typing import Tuple
import torch
import torch.nn as nn


from engine.models.building_blocks import Conv, C3k2, C2PSA, SPPF


class YOLO11Backbone(nn.Module):
    """The YOLOv11 backbone.

    scales: # model compound scaling constants
        # [depth, width, max_channels]
        n: [0.50, 0.25, 1024]
        s: [0.50, 0.50, 1024]
        m: [0.50, 1.00, 512]
        l: [1.00, 1.00, 512]
        x: [1.00, 1.50, 512]
    """

    # From left to right:
    # in_channels, out_channels, num_blocks, use_c3k2, use_psa
    arch_settings = {
        "P5": [
            [128, 256, 2, False, False],
            [256, 512, 2, False, False],
            [512, 512, 2, True, False],
            [512, None, 2, True, True],
        ],
    }

    def __init__(
        self,
        arch="P5",
        last_stage_out_channels: int = 512,
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
    ):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__()
        arch_setting = self.arch_settings[arch]
        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        self.out_indices = out_indices
        self.last_stage_out_channels = last_stage_out_channels

        self.stem = Conv(c1=3, c2=int(64 * widen_factor), k=3, s=2)
        outs_channel = [int(64 * widen_factor)]
        self.layers = ["stem"]

        for i, (in_channels, out_channels, num_blocks, use_c3k2, use_psa) in enumerate(
            arch_setting
        ):
            # 下采样
            down_in_channels = outs_channel[-1]
            down_out_channels = int(in_channels * widen_factor)
            down_sample = Conv(down_in_channels, down_out_channels, 3, 2)

            # 阶段
            stage = [down_sample]
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            module = C3k2(in_channels, out_channels, num_blocks, use_c3k2)
            stage.append(module)
            if use_psa:
                sppf = SPPF(out_channels, out_channels, 5)
                stage.append(sppf)
                c2psa = C2PSA(out_channels, out_channels, n=2)
                stage.append(c2psa)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")
            outs_channel.append(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
