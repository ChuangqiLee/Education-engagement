"""Small YAML configuration loader with path resolution."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping: %s" % path)
    return data


def save_yaml(data: Dict[str, Any], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def require(config: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError("missing required configuration key: %s" % dotted_key)
        value = value[key]
    return value

