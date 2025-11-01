#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from typing import Optional
import yaml
import os
from pathlib import Path
import importlib.resources as pkg_resources
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_config_path(config_name: str = "disco.yaml") -> str:
    try:
        # Try package-based access
        return str(pkg_resources.files("split.disco.config").joinpath(config_name))
    except Exception:
        # Fallback to relative file location
        here = Path(__file__).resolve().parent
        candidate = here / config_name
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"Could not find {config_name}. Checked {candidate} and package resources."
        )

def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = get_config_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)