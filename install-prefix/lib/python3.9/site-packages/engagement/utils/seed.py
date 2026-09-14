"""Deterministic seed setup for Python, NumPy, and PyTorch."""

import os
import random
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def worker_seed(worker_id: int) -> None:
    try:
        import numpy as np
        import torch

        seed: Optional[int] = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        random.seed(seed)
    except ImportError:
        random.seed(worker_id)

