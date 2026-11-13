"""Configuration helpers for curriculum learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None


@dataclass
class CurriculumConfig:
    total_epochs: int = 50
    initial_mag: float = 0.1
    final_mag: float = 0.678342
    schedule: str = "linear"
    t_min: int = 5
    t_max: int = 50
    num_augmentations: int = 2
    random_seed: int = 0
    enable_curriculum: bool = True
    dataset_dir: str = ""
    dataset_path: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 32
    learning_rate: float = 1e-4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumConfig":
        values = {field.name: getattr(cls, field.name) for field in fields(cls)}
        values.update({key: value for key, value in data.items() if key in values})
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CurriculumConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        if yaml is None:
            raise ImportError("PyYAML is required to load curriculum YAML configs")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
