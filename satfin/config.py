"""YAML config loading."""
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
