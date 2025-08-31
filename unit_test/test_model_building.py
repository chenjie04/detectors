import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from engine.config import Config
from engine.parsers.model_parser import parse_model

config_path = "configs/sod_detr/sod_detr_n_500e_coco.py"
cfg = Config.fromfile(config_path)
model_cfg = cfg.model
model = parse_model(model_cfg.name, model_cfg.config)
print(model)
