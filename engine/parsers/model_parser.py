from engine.models.detectors import YOLO11

def parse_model(model_name: str, model_cfg: dict):
    """将模型配置解析成实例"""
    if model_name == "YOLO11":
        return YOLO11(**model_cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
