from .base import BaseEnhancementModel
from .gtcrn.model import GTCRN
from .lisennet.model import LiSenNet
from .tfgridnet.model import TFGridNet

MODEL_REGISTRY = {"gtcrn": GTCRN, "lisennet": LiSenNet, "tfgridnet": TFGridNet}


def build_model(name: str) -> BaseEnhancementModel:
    try:
        model_class = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc
    return model_class()


__all__ = ["BaseEnhancementModel", "build_model", "GTCRN", "LiSenNet", "TFGridNet"]
