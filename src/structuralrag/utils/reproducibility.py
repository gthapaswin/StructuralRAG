from __future__ import annotations

import os
import platform
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def environment_info() -> dict[str, str]:
    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = str(torch.backends.mps.is_available())
    except ImportError:  # pragma: no cover
        pass
    return info


def best_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"
