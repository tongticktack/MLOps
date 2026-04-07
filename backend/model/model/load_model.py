"""
Model loader — single source of truth for where the model artifact lives.
"""

from pathlib import Path

import joblib
import xgboost as xgb

_MODEL_PATH    = Path(__file__).parent / "window_model.pkl"
_PRETRAIN_PATH = Path(__file__).parent / "window_model_pretrain.pkl"


def load_pretrain_model() -> xgb.XGBRegressor:
    """Load the fixed pretrain model. Never overwritten by retrain."""
    if not _PRETRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Pretrain model not found at {_PRETRAIN_PATH}.\n"
            "Copy window_model.pkl → window_model_pretrain.pkl first."
        )
    return joblib.load(_PRETRAIN_PATH)


def load_model(path: Path | None = None) -> xgb.XGBRegressor:
    """
    Load and return the pretrained XGBRegressor.

    Parameters
    ----------
    path : optional override for the model file location.
    """
    target = path or _MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Model not found at {target}.\n"
            "Run  python model/train.py  first to train and save the model."
        )
    return joblib.load(target)
