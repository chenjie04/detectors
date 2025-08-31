from engine.dataset import YOLODataset
from engine.config import Config

def parse_dataset(dataset_name: str, dataset_cfg: Config) -> YOLODataset:
    if dataset_name == "YOLODataset":
        return YOLODataset(**dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")