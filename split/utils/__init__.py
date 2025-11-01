"""
split.utils

Provides default PyTorch imports, device configuration,
and commonly used helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, Any
from scipy.ndimage import gaussian_filter, map_coordinates
from dataclasses import dataclass
from torch import Tensor 

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "torch", "nn", "F", "optim", "DataLoader", "Dataset", "transforms", "models",
    "device", "set_seed", "Image", "np", "Optional", "Tuple",
    "gaussian_filter", "map_coordinates", "Union", "dataclass", "Tensor", "Any"
]
