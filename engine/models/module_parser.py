"""
将配置解析成实例
"""

from engine.models.backbones import YOLO11Backbone
from engine.models.necks import YOLO11Neck
from engine.models.heads import YOLO11Head

def parse_module(module_name: str, module_cfg: dict):
    """将模型配置解析成实例"""
    if module_name == "YOLO11Backbone":
        return YOLO11Backbone(**module_cfg)
    elif module_name == "YOLO11Neck":
        return YOLO11Neck(**module_cfg)
    elif module_name == "YOLO11Head":
        return YOLO11Head(**module_cfg)
    else:
        raise ValueError(f"Unknown module name: {module_name}")

