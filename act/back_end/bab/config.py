# ===- act/back_end/bab/config.py - BaB Configuration --------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Configuration dataclass for Branch-and-Bound verification.
#   Supports both programmatic construction and YAML serialisation,
#   following the same pattern as FuzzingConfig in act/pipeline/fuzzing.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Union

import yaml

from act.util.path_config import get_project_root

_DEFAULT_YAML = Path(__file__).parent / "config.yaml"


@dataclass
class BaBConfig:
    """Configuration for Branch-and-Bound verification.

    Supports three construction modes:

    1. ``BaBConfig()``                — programmatic with defaults
    2. ``BaBConfig.from_yaml()``      — load from YAML (+ optional overrides)
    3. ``config.to_yaml(path)``       — persist current config to YAML
    """

    max_depth: int = 20
    max_nodes: int = 2000
    time_budget_s: float = 300.0

    branching_method: str = "random"
    scheduling_method: str = "random"

    verbose: bool = False

    # -- YAML I/O -----------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BaBConfig:
        """Load config from YAML with optional keyword overrides.

        Args:
            config_path: Path to YAML file.
                          Default: ``act/back_end/bab/config.yaml``.
            **overrides:  Any field name ⟶ value that takes precedence
                          over the YAML contents.

        Returns:
            A ``BaBConfig`` instance.
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML

        if not path.exists():
            raise FileNotFoundError(
                f"BaB config not found: {path}\nExpected: act/back_end/bab/config.yaml"
            )

        with open(path) as f:
            yaml_data = yaml.safe_load(f)

        yaml_config: dict = yaml_data.get("bab", {})

        valid_keys = {fld.name for fld in fields(cls)}
        merged = {k: v for k, v in yaml_config.items() if k in valid_keys}
        merged.update({k: v for k, v in overrides.items() if k in valid_keys})

        return cls(**merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write current config to a YAML file.

        Args:
            path: Destination file path.

        Returns:
            Resolved ``Path`` that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                {"bab": asdict(self)}, f, default_flow_style=False, sort_keys=False
            )

        return path
