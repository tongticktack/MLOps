"""
Model loader — single source of truth for where the model artifact lives.
"""

from pathlib import Path

import joblib
import xgboost as xgb

_MODEL_PATH = Path(__file__).parent / "window_model.pkl"


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
