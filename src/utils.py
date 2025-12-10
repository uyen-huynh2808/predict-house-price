import os
import logging
import random
import json
from typing import Any, Dict

import numpy as np
import configparser

try:
    import yaml  # dùng cho config.yaml
except ImportError:
    yaml = None


def set_seed(seed: int = 42) -> None:
    """
    Cố định random seed để đảm bảo reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logger(log_path: str, name: str = "MainLogger") -> logging.Logger:
    """
    Thiết lập logger ghi ra file và console.
    Thêm tham số 'name' để linh hoạt đặt tên logger.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Tránh bị add handler trùng nếu gọi nhiều lần
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load config từ file .yaml/.yml, .ini hoặc .json.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()

    # YAML
    if ext in [".yaml", ".yml"]:
        if yaml is None:
            raise ImportError(
                "PyYAML chưa được cài. Hãy thêm 'pyyaml' vào requirements.txt."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # INI
    if ext == ".ini":
        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="utf-8")
        config_dict: Dict[str, Any] = {}
        for section in parser.sections():
            config_dict[section] = dict(parser.items(section))
        return config_dict

    # Fallback: JSON
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Không đọc được config từ {config_path}: {e}")