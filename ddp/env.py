# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

import os
import torch
import numpy as np
import random
def setup_env():
    """Configure runtime environment variables dynamically."""
 
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("BATCH_SIZE","64")
    os.environ.setdefault("NUMBER_NODE","1")
    
    # Optional Hugging Face cache
    os.environ.setdefault("HF_HOME", "/tank/hf_tank")
    

    # CUDA setup
    if torch.cuda.is_available():
        print(f"✅ CUDA detected: {torch.cuda.device_count()} device(s)")
    else:
        print("⚠️ CUDA not detected — training will run on CPU.")


def seed_everything(seed: int = 42):
    """Ensure deterministic behavior across workers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")
