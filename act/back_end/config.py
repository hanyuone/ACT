# ===- act/back_end/config.py - Backend Configuration ---------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union

import yaml

_DEFAULT_YAML = Path(__file__).parent / "config.yaml"

_VALID_SOLVERS = {"auto", "gurobi", "torchlp", "dual"}
_VALID_DEVICES = {"cpu", "cuda", "gpu"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_REGISTRY_MODES = {"intersection", "union"}
_VALID_COVERAGE_MODES = {"basic", "full"}


# ---------------------------------------------------------------------------
# BaBConfig — Branch-and-Bound algorithm parameters
# ---------------------------------------------------------------------------


@dataclass
class BaBConfig:
    """Configuration for Branch-and-Bound verification algorithm.

    Construction::

        BaBConfig()                     # programmatic defaults
        BaBConfig.from_yaml()           # load from act/back_end/config.yaml
        BaBConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    max_depth: int = 20
    max_nodes: int = 2000

    branching_method: str = "random"
    bounding_method: str = "random"

    # Dual-tier solver knobs — support solver_tier="dual_alpha_eta" with
    # Iterative slope + Lagrange-multiplier optimization for the dual backward pass.
    solver_tier: str = "lp"
    """Solver tier for BaB bound computation. One of:

    - ``"lp"``             (default): LP/MILP backend.
    - ``"dual"``           : DualSolver (linear-relaxation dual bound, no iterative optimization).
    - ``"dual_alpha"``     : DualSolver with Lagrange-relaxed lower-slope optimization (Adam on α ∈ [0, 1]).
    - ``"dual_alpha_eta"`` : DualSolver with joint slope + split-constraint KKT-multiplier optimization (α, η).
    """

    dual_n_iters: int = 50
    """Number of Adam iterations for α/η optimization (only used in ``dual_alpha`` / ``dual_alpha_eta`` tiers)."""

    lr_alpha: float = 0.1
    """Adam learning rate for α (slope) variables."""

    lr_beta: float = 0.1
    """Adam learning rate for η (split-constraint KKT multipliers). 0.1 default; tune per network."""

    lr_decay: float = 0.98
    """Multiplicative learning-rate decay applied each Adam iteration."""

    warm_start_enabled: bool = True
    """Reuse α/η tensors from the parent subproblem as the initial point for child optimization."""

    per_class_alpha: bool = True
    """Allocate separate α tensors per output class (tighter bounds) rather than sharing one α."""

    verbose: bool = False

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BaBConfig:
        """Load BaB settings from YAML with optional keyword overrides.

        Reads from ``backend.bab`` in the unified backend config, falling
        back to a top-level ``bab`` key for standalone BaB YAML files.
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML

        if not path.exists():
            raise FileNotFoundError(
                f"Backend config not found: {path}\nExpected: act/back_end/config.yaml"
            )

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Support both nested (backend.bab) and flat (bab) YAML layouts.
        backend_section = yaml_data.get("backend", {})
        yaml_config: dict = backend_section.get("bab", yaml_data.get("bab", {}))

        valid_keys = {fld.name for fld in fields(cls)}
        merged = {k: v for k, v in yaml_config.items() if k in valid_keys}
        merged.update({k: v for k, v in overrides.items() if k in valid_keys})

        return cls(**merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write BaB settings to a standalone YAML file (top-level ``bab`` key)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                {"bab": asdict(self)}, f, default_flow_style=False, sort_keys=False
            )

        return path


# ---------------------------------------------------------------------------
# GenerationConfig — network generation (net_factory) parameters
# ---------------------------------------------------------------------------

_DEFAULT_GEN_CONFIG = str(
    Path(__file__).parent / "examples" / "config_gen_act_net.yaml"
)


@dataclass
class GenerationConfig:
    """Configuration for network generation via ``NetFactory``.

    Controls the simple knobs (how many, where, seed, TF filtering).
    The architecture sampling DSL lives in a separate file referenced
    by ``gen_config_path``.
    """

    gen_config_path: str = _DEFAULT_GEN_CONFIG
    output_dir: str = "act/back_end/examples/nets"
    num_instances: int = 15
    base_seed: int = 42
    name_prefix: str = "cfg_seed"
    tf_targets: Optional[List[str]] = None
    registry_mode: str = "intersection"
    coverage_mode: str = "basic"
    coverage_max_attempts: int = 1000
    coverage_report: bool = True
    write_manifest: bool = True

    def __post_init__(self) -> None:
        if self.registry_mode not in _VALID_REGISTRY_MODES:
            raise ValueError(
                f"Invalid registry_mode {self.registry_mode!r}; "
                f"expected one of {_VALID_REGISTRY_MODES}"
            )
        if self.coverage_mode not in _VALID_COVERAGE_MODES:
            raise ValueError(
                f"Invalid coverage_mode {self.coverage_mode!r}; "
                f"expected one of {_VALID_COVERAGE_MODES}"
            )


# ---------------------------------------------------------------------------
# BackendConfig — unified back-end configuration
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """Unified configuration for the ACT back-end.

    Covers runtime selectors (solver / device / dtype), verification timeout,
    and nested BaB settings.  The canonical source is ``act/back_end/config.yaml``;
    CLI flags and environment variables override it at load time.

    Construction::

        BackendConfig()                     # programmatic defaults
        BackendConfig.from_yaml()           # load from default YAML
        BackendConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    solver: str = "auto"
    device: str = "cpu"
    dtype: str = "float64"
    verbose: bool = False
    timeout: float = 300.0

    bab_enabled: bool = False
    bab: BaBConfig = field(default_factory=BaBConfig)

    # -- batched-API knobs (C11) --------------------------------------------
    lp_enabled: bool = True
    """Enable the LP-batched tier (tier 2) in the 3-tier cascade.

    Set to False to skip verify_lp_batched and fall through directly to BaB.
    Must be False when solver='gurobi' (Gurobi solve_batch is N=1 only;
    see commit af797ff / C6).
    """

    bab_max_batch_size: int = 8
    """Maximum K for BaB sub-problem batching (tier 3).

    BaB dispatches up to K sub-problems per solve_batch call.  Set to 1 to
    disable batching inside BaB (equivalent to the legacy sequential loop).
    Must be 1 when solver='gurobi' (same N=1 restriction as lp_enabled).
    """

    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(
                f"Invalid solver {self.solver!r}; expected one of {_VALID_SOLVERS}"
            )
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"Invalid device {self.device!r}; expected one of {_VALID_DEVICES}"
            )
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype {self.dtype!r}; expected one of {_VALID_DTYPES}"
            )
        # Gurobi solve_batch is restricted to N=1 (commit af797ff / C6).
        # Fail loud at config-load time rather than at the first batched call.
        if self.solver == "gurobi":
            if self.lp_enabled:
                raise ValueError(
                    "BackendConfig: solver='gurobi' is incompatible with "
                    "lp_enabled=True.  GurobiSolver.solve_batch raises for N>1 "
                    "(Gurobi does not expose a truly parallel multi-LP API for "
                    "varying constraint matrices; see commit af797ff).  "
                    "Either set lp_enabled=False or switch to solver='torchlp'."
                )
            if self.bab_max_batch_size > 1:
                raise ValueError(
                    f"BackendConfig: solver='gurobi' is incompatible with "
                    f"bab_max_batch_size={self.bab_max_batch_size} > 1.  "
                    f"GurobiSolver.solve_batch raises for N>1.  "
                    f"Either set bab_max_batch_size=1 or switch to solver='torchlp'."
                )

    # -- YAML I/O -----------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BackendConfig:
        """Load config from YAML with optional keyword overrides.

        YAML layout::

            backend:
              solver: "torchlp"
              ...
              bab:
                enabled: true
                ...
              generation:
                num_instances: 15
                ...

        Override naming:
          - ``bab_<field>`` → ``BaBConfig.<field>``
          - ``gen_<field>`` → ``GenerationConfig.<field>``
          - ``bab_enabled`` → top-level ``bab_enabled``
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML
        if not path.exists():
            raise FileNotFoundError(f"Backend config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        backend_raw: dict = raw.get("backend", {})
        bab_raw: dict = backend_raw.pop("bab", {})
        gen_raw: dict = backend_raw.pop("generation", {})

        # Extract "enabled" from bab section → top-level bab_enabled
        bab_enabled = bab_raw.pop("enabled", None)

        # Route prefixed overrides to the right sub-config
        bab_fields = {fld.name for fld in fields(BaBConfig)}
        gen_fields = {fld.name for fld in fields(GenerationConfig)}
        bab_overrides: dict = {}
        gen_overrides: dict = {}
        top_overrides: dict = {}
        for k, v in overrides.items():
            if k.startswith("bab_") and k[4:] in bab_fields:
                bab_overrides[k[4:]] = v
            elif k.startswith("gen_") and k[4:] in gen_fields:
                gen_overrides[k[4:]] = v
            else:
                top_overrides[k] = v

        # Build BaBConfig
        bab_merged = {k: v for k, v in bab_raw.items() if k in bab_fields}
        bab_merged.update(bab_overrides)
        bab_config = BaBConfig(**bab_merged)

        # Build GenerationConfig
        gen_merged = {k: v for k, v in gen_raw.items() if k in gen_fields}
        gen_merged.update(gen_overrides)
        gen_config = GenerationConfig(**gen_merged)

        # Build top-level config
        top_fields = {fld.name for fld in fields(cls)} - {"bab", "generation"}
        top_merged: dict = {}
        for k, v in backend_raw.items():
            if k in top_fields:
                top_merged[k] = v

        if bab_enabled is not None:
            top_merged["bab_enabled"] = bab_enabled

        top_merged.update({k: v for k, v in top_overrides.items() if k in top_fields})

        return cls(bab=bab_config, generation=gen_config, **top_merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = asdict(self)
        bab_d = d.pop("bab")
        gen_d = d.pop("generation")
        bab_enabled = d.pop("bab_enabled")
        bab_d["enabled"] = bab_enabled

        with open(path, "w") as f:
            yaml.dump(
                {"backend": {**d, "bab": bab_d, "generation": gen_d}},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        return path


if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def _check(label: str, fn) -> None:  # pragma: no cover
        global passed, failed
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {label}: {exc}")
            failed += 1

    print("BackendConfig.__post_init__ rejection tests")

    def _t1():  # pragma: no cover
        try:
            BackendConfig(solver="gurobi", lp_enabled=True)
            raise AssertionError("expected ValueError not raised")
        except ValueError as e:
            assert "lp_enabled" in str(e), f"wrong message: {e}"

    def _t2():  # pragma: no cover
        try:
            BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size=2)
            raise AssertionError("expected ValueError not raised")
        except ValueError as e:
            assert "bab_max_batch_size" in str(e), f"wrong message: {e}"

    def _t3():  # pragma: no cover
        cfg = BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size=1)
        assert cfg.solver == "gurobi"
        assert not cfg.lp_enabled
        assert cfg.bab_max_batch_size == 1

    def _t4():  # pragma: no cover
        cfg = BackendConfig()
        assert cfg.lp_enabled is True
        assert cfg.bab_max_batch_size == 8

    _check("gurobi + lp_enabled=True raises ValueError", _t1)
    _check("gurobi + bab_max_batch_size=2 raises ValueError", _t2)
    _check("gurobi + lp_enabled=False + bab_max_batch_size=1 succeeds", _t3)
    _check("default config has lp_enabled=True, bab_max_batch_size=8", _t4)

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
