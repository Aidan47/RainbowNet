from copy import deepcopy
from pathlib import Path
from sysconfig import get_path
from typing import Any, Dict
import yaml


HERE = Path(__file__).parent


def load(model: str):
    path = HERE / f"configs/{model}.yaml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data