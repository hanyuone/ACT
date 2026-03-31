#===- act/back_end/net_factory/factory.py - NetFactory Orchestration ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ConfigSampler (YAML rule → random config) + NetFactory (orchestration,
#   weight generation, variable assignment, coverage tracking, serialization).
#
# Optimizations vs UCU NetFactory:
#   - Registry-driven variable generation (_VAR_STRATEGIES replaces ~180-line if-elif)
#   - Data-driven minimal templates (in layer_builder.py, replaces ~340 lines)
#   - Coverage-guided sampling in full mode
#   - No redundant wrapper functions
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import json
import logging
import random
import secrets
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import torch
import yaml

from act.back_end.core import Layer, Net
from act.back_end.serialization.serialization import NetSerializer
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_dtype

from .layer_builder import build_cnn_layers, build_mlp_layers, generate_minimal_template
from .tf_capabilities import get_allowed_layers

logger = logging.getLogger(__name__)

# ============================================================================
# Default coverage layer list (fallback when TF registry unavailable)
# ============================================================================

_DEFAULT_COVERAGE_LAYERS = [
    "DENSE", "BIAS", "SCALE", "BN",
    "RELU", "LRELU", "ABS", "CLIP", "SQUARE", "POWER",
    "SIGMOID", "TANH", "SOFTPLUS", "SILU", "RELU6",
    "HARDTANH", "HARDSIGMOID", "HARDSWISH", "MISH", "SOFTSIGN",
    "ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN", "MATMUL", "CONCAT",
    "CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D",
    "MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D", "AVGPOOL1D", "AVGPOOL2D",
    "PAD", "UPSAMPLE", "FLATTEN",
    "RESHAPE", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE",
    "TILE", "EXPAND", "SLICE", "GATHER", "INDEX_SELECT",
]

# ============================================================================
# Internal utilities
# ============================================================================

def _derive_seed(base_seed: int, idx: int, instance_id: str) -> int:
    payload = f"{base_seed}|{idx}|{instance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


# ============================================================================
# Registry-driven variable generation (optimization: replaces if-elif chain)
# ============================================================================

def _vars_input(layers, i, vc, meta):
    n = torch.Size(meta["shape"]).numel()
    return [], list(range(vc, vc + n)), vc + n

def _vars_passthrough(layers, i, vc, meta):
    pv = layers[i - 1].out_vars
    return list(pv), list(pv), vc

def _vars_same_size(layers, i, vc, meta):
    preds_idx = meta.get("preds_indices", [None])[0] if isinstance(meta.get("preds_indices"), list) else meta.get("preds_indices")
    in_vars = list(layers[preds_idx].out_vars) if preds_idx is not None else list(layers[i - 1].out_vars)
    out_vars = list(range(vc, vc + len(in_vars)))
    return in_vars, out_vars, vc + len(in_vars)

def _vars_dense(layers, i, vc, meta):
    in_vars = list(layers[i - 1].out_vars)
    n_out = int(meta["out_features"])
    out_vars = list(range(vc, vc + n_out))
    return in_vars, out_vars, vc + n_out

def _vars_spatial(layers, i, vc, meta):
    preds_idx = meta.get("preds_indices", [None])[0] if isinstance(meta.get("preds_indices"), list) else meta.get("preds_indices")
    in_vars = list(layers[preds_idx].out_vars) if preds_idx is not None else list(layers[i - 1].out_vars)
    n_out = torch.Size(meta["output_shape"]).numel()
    out_vars = list(range(vc, vc + n_out))
    return in_vars, out_vars, vc + n_out

def _vars_binary(layers, i, vc, meta):
    x_vars = meta.get("x_vars", [])
    y_vars = meta.get("y_vars", [])
    if not x_vars or not y_vars:
        if len(layers) >= 2:
            x_vars = list(layers[i - 2].out_vars)
            y_vars = list(layers[i - 1].out_vars)
            meta["x_vars"] = x_vars
            meta["y_vars"] = y_vars
    in_vars = list(x_vars) + list(y_vars)
    out_vars = list(range(vc, vc + len(x_vars)))
    return in_vars, out_vars, vc + len(x_vars)

def _vars_concat(layers, i, vc, meta):
    preds = meta.get("preds_indices", [])
    if not preds and i >= 2:
        preds = [i - 2, i - 1]
    in_vars = []
    for pidx in preds:
        if pidx < len(layers):
            in_vars.extend(layers[pidx].out_vars)
    out_vars = list(range(vc, vc + len(in_vars)))
    return in_vars, out_vars, vc + len(in_vars)

_ELEMENTWISE_KINDS = frozenset({
    "RELU", "SIGMOID", "TANH", "LRELU", "RELU6", "HARDTANH", "HARDSIGMOID",
    "HARDSWISH", "SILU", "SOFTPLUS", "MISH", "SOFTSIGN", "ABS",
    "CLIP", "SQUARE", "POWER", "GELU",
    "BIAS", "SCALE", "BN",
})
_SPATIAL_KINDS = frozenset({
    "CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D",
    "MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D", "AVGPOOL1D", "AVGPOOL2D",
    "UPSAMPLE", "PAD",
})
_BINARY_KINDS = frozenset({"ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN", "MATMUL"})
_SHAPE_PRESERVE_KINDS = frozenset({"RESHAPE", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE", "FLATTEN"})
_SHAPE_EXPAND_KINDS = frozenset({"TILE", "EXPAND"})
_SHAPE_REDUCE_KINDS = frozenset({"SLICE", "GATHER", "INDEX_SELECT"})


def _generate_layer_variables(kind, i, vc, meta, layers):
    """Registry-driven variable generation."""
    if kind == "INPUT":
        return _vars_input(layers, i, vc, meta)
    if kind in ("INPUT_SPEC", "ASSERT"):
        return _vars_passthrough(layers, i, vc, meta)
    if kind == "DENSE":
        return _vars_dense(layers, i, vc, meta)
    if kind in _ELEMENTWISE_KINDS:
        return _vars_same_size(layers, i, vc, meta)
    if kind in _SPATIAL_KINDS:
        return _vars_spatial(layers, i, vc, meta)
    if kind in _BINARY_KINDS:
        return _vars_binary(layers, i, vc, meta)
    if kind == "CONCAT":
        return _vars_concat(layers, i, vc, meta)
    if kind in _SHAPE_PRESERVE_KINDS:
        return _vars_same_size(layers, i, vc, meta)
    if kind in _SHAPE_EXPAND_KINDS or kind in _SHAPE_REDUCE_KINDS:
        if "output_shape" in meta:
            return _vars_spatial(layers, i, vc, meta)
        return _vars_same_size(layers, i, vc, meta)
    raise NotImplementedError(f"No variable strategy for layer kind '{kind}'")


# ============================================================================
# ConfigSampler
# ============================================================================

class ConfigSampler:
    """
    Generic sampler using YAML-defined sampling rules.

    TF-driven design: activation choices and injectable operators are
    computed dynamically from ``allowed_layers`` (which comes from
    ``tf_capabilities``), NOT hardcoded in YAML.  The YAML only controls
    structural parameters (depth, width, variant, etc.).
    """

    _FAMILY_REQUIRED_LAYERS = {
        "mlp": {"DENSE", "RELU"},
        "cnn2d": {"CONV2D", "DENSE", "RELU"},
    }

    # Master list of activation-like layers (element-wise, no required tensor params).
    # The actual available set is computed as _ALL_ACTIVATIONS & allowed_layers.
    # Only activations that have torch_module in REGISTRY (so act2torch can restore them).
    # ABS and CLIP are excluded: no nn.Module equivalent (torch.abs / torch.clamp are functions).
    _ALL_ACTIVATIONS = frozenset({
        "RELU", "LRELU", "SIGMOID", "TANH", "RELU6",
        "HARDTANH", "HARDSIGMOID", "HARDSWISH", "SILU",
        "SOFTPLUS", "MISH", "SOFTSIGN", "GELU",
    })

    # Operators that can be injected into any network as fork-merge pairs.
    # Each group maps op kinds to the structural pattern they need.
    _INJECTABLE_BINARY_OPS = frozenset({
        "ADD", "SUB", "MUL", "DIV", "MAX", "MIN",
    })
    _INJECTABLE_SHAPE_OPS = frozenset({
        "RESHAPE", "SQUEEZE", "UNSQUEEZE", "TRANSPOSE",
    })
    _INJECTABLE_NORM_OPS = frozenset({
        "BIAS", "SCALE",
    })

    # Maps YAML pool_kind / downsample values to the LayerKind they produce.
    _POOL_KIND_TO_LAYER = {"maxpool": "MAXPOOL2D", "avgpool": "AVGPOOL2D"}

    def __init__(self, config: Dict[str, Any], allowed_layers: Optional[FrozenSet[str]] = None):
        self.config = config
        self.allowed_layers = allowed_layers or frozenset(_DEFAULT_COVERAGE_LAYERS)

        # Precompute available layer sets from allowed_layers (same pattern for all).
        # Activations: which activation functions can we use?
        self.available_activations = self._ALL_ACTIVATIONS & self.allowed_layers
        if not self.available_activations:
            self.available_activations = frozenset({"RELU"})
        self.available_activations_list = sorted(self.available_activations)

        # Pooling: which pool kinds can we use?
        self.available_pool_kinds = [
            k for k, v in self._POOL_KIND_TO_LAYER.items() if v in self.allowed_layers
        ]
        # Downsample: which downsample methods can we use?
        self.available_downsamples = ["stride2_conv"] + [
            k for k, v in self._POOL_KIND_TO_LAYER.items() if v in self.allowed_layers
        ]
        # Head pooling: can we use AVGPOOL2D for global pooling?
        self.can_head_pool = "AVGPOOL2D" in self.allowed_layers

        self.available_families = self._compute_available_families()

    def _compute_available_families(self) -> List[str]:
        available = []
        for family, required in self._FAMILY_REQUIRED_LAYERS.items():
            if family not in self.config.get("families", {}):
                continue
            if required <= self.allowed_layers:
                available.append(family)
        return available

    # ---- Rule engine ----

    def _sample_value(self, rng: random.Random, rule: Any) -> Any:
        if not isinstance(rule, dict):
            return rule
        if "const" in rule:
            return rule["const"]
        if "choice" in rule:
            return rng.choice(rule["choice"])
        if "range" in rule:
            lo, hi = int(rule["range"][0]), int(rule["range"][1])
            if hi < lo:
                lo, hi = hi, lo
            return rng.randint(lo, hi)
        if "weighted" in rule:
            items = list(rule["weighted"].keys())
            weights = list(rule["weighted"].values())
            total = sum(weights)
            return rng.choices(items, weights=[w / total for w in weights])[0]
        if "repeat" in rule:
            r = rule["repeat"]
            count = self._sample_value(rng, r["count"])
            return [self._sample_value(rng, r["value"]) for _ in range(int(count))]
        if "probability" in rule:
            return rng.random() < float(rule["probability"])
        raise ValueError(f"Unknown sampling rule: {rule}")

    def _sample_dict(self, rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in spec.items():
            if isinstance(value, dict):
                is_rule = any(k in value for k in ("choice", "range", "weighted", "repeat", "probability", "const"))
                result[key] = self._sample_value(rng, value) if is_rule else self._sample_dict(rng, value)
            else:
                result[key] = value
        return result

    # ---- Public sampling API ----

    def sample_family(self, rng: random.Random) -> Tuple[str, Dict[str, Any]]:
        if not self.available_families:
            raise ValueError("No families available for current allowed_layers")
        selection = self.config["family_selection"]
        if "weighted" in selection:
            filtered = {k: v for k, v in selection["weighted"].items() if k in self.available_families}
            if not filtered:
                raise ValueError(f"No families match: {self.available_families}")
            names = list(filtered.keys())
            weights = list(filtered.values())
            total = sum(weights)
            family = rng.choices(names, weights=[w / total for w in weights])[0]
        else:
            raise ValueError("family_selection must have 'weighted' strategy")
        params = self._sample_dict(rng, self.config["families"][family])

        # Type normalization
        for k in ("input_shape", "hidden_sizes", "conv_channels"):
            if k in params:
                params[k] = tuple(int(x) for x in params[k])

        # Override YAML choices with TF-aware precomputed sets.
        # Same pattern for every variable layer type: precompute in __init__,
        # override here.  The builder never sees unsupported layers.
        params["activation"] = rng.choice(self.available_activations_list).lower()

        if "use_pooling" in params:
            if self.available_pool_kinds:
                params["pool_kind"] = rng.choice(self.available_pool_kinds)
            else:
                params["use_pooling"] = False

        if "downsample" in params:
            params["downsample"] = rng.choice(self.available_downsamples)

        if "head_pool_to_1x1" in params and not self.can_head_pool:
            params["head_pool_to_1x1"] = False

        return family, params

    def sample_input_spec(self, rng: random.Random) -> Dict[str, Any]:
        sc = self.config["input_spec"]
        kind = self._sample_value(rng, sc["kind"])
        vr = self._sample_value(rng, sc["value_range"])
        lo, hi = float(vr[0]), float(vr[1])
        if hi < lo:
            lo, hi = hi, lo
        if kind == "BOX":
            shrink = sc.get("box_shrink_range", [0.0, 0.2])
            span = hi - lo
            sa, sb = rng.random() * shrink[1], rng.random() * shrink[1]
            lb_val, ub_val = lo + span * sa, hi - span * sb
            if ub_val < lb_val:
                lb_val, ub_val = lo, hi
            return {"kind": "BOX", "value_range": (lo, hi), "lb_val": lb_val, "ub_val": ub_val}
        if kind == "LINF_BALL":
            center = lo + (hi - lo) * rng.random()
            eps = self._sample_value(rng, sc["eps"])
            eps = min(float(eps), 0.5 * (hi - lo)) if hi > lo else 0.0
            return {"kind": "LINF_BALL", "value_range": (lo, hi), "center_val": center, "eps": eps}
        raise ValueError(f"Unsupported input_spec kind '{kind}'")

    def sample_output_spec(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        sc = self.config["output_spec"]
        kind = self._sample_value(rng, sc["kind"])
        y_true = rng.randrange(num_classes)
        if kind == "TOP1_ROBUST":
            return {"kind": "TOP1_ROBUST", "y_true": y_true}
        if kind == "MARGIN_ROBUST":
            margin = self._sample_value(rng, sc["margin"])
            return {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}
        if kind == "LINEAR_LE":
            cr = sc["linear_le_c_range"]
            dr = sc["linear_le_d_range"]
            c = [cr[0] + (cr[1] - cr[0]) * rng.random() for _ in range(num_classes)]
            d = dr[0] + (dr[1] - dr[0]) * rng.random()
            return {"kind": "LINEAR_LE", "c": c, "d": d}
        if kind == "RANGE":
            br = self._sample_value(rng, sc["range_bounds"])
            lo, hi = br[0], br[1]
            lb = [min(lo + (hi - lo) * rng.random(), lo + (hi - lo) * rng.random()) for _ in range(num_classes)]
            ub = [max(lo + (hi - lo) * rng.random(), lo + (hi - lo) * rng.random()) for _ in range(num_classes)]
            return {"kind": "RANGE", "lb": lb, "ub": ub}
        raise ValueError(f"Unsupported output_spec kind '{kind}'")


# ============================================================================
# NetFactory
# ============================================================================

class NetFactory:
    """
    Generator-driven factory for ACT Nets with TF-aware layer support.

    Args:
        gen_config_path: Path to YAML generation config.
        tf_targets: Target TFs for layer filtering (None = all).
        registry_mode: ``"intersection"`` or ``"union"`` for combining TF layer sets.
    """

    def __init__(
        self,
        gen_config_path: Optional[str] = None, *,
        output_dir: Optional[str] = None,
        base_seed: Optional[int] = None,
        num_instances: Optional[int] = None,
        name_prefix: Optional[str] = None,
        write_manifest: Optional[bool] = None,
        tf_targets: Optional[List[str]] = None,
        registry_mode: str = "intersection",
    ):
        if gen_config_path is None:
            gen_config_path = str(Path(__file__).parent.parent / "examples" / "config_gen_act_net.yaml")
        self.config_path = str(gen_config_path)
        self.config = self._load_config(self.config_path)
        common = self.config["common"]

        self.tf_targets = tf_targets
        self.registry_mode = registry_mode
        self.allowed_layers = self._compute_allowed_layers(tf_targets, registry_mode)
        self.sampler = ConfigSampler(self.config, allowed_layers=self.allowed_layers)

        self.base_seed = int(base_seed) if base_seed is not None else (int(common["base_seed"]) if common.get("base_seed") else int(secrets.randbits(32)))
        self.num_instances = int(num_instances) if num_instances is not None else int(common["num_instances"])
        self.name_prefix = str(name_prefix) if name_prefix is not None else str(common["name_prefix"])

        od = output_dir or common["output_dir"]
        self.output_dir = Path(od)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_manifest = bool(write_manifest) if write_manifest is not None else bool(common.get("write_manifest", True))
        mp = common.get("manifest_path")
        self.manifest_path = Path(mp) if mp else (self.output_dir / "_meta" / "manifest.json")

        self.coverage_mode = common.get("coverage_mode", "basic")
        self.coverage_max_attempts = int(common.get("coverage_max_attempts", 1000))
        self.coverage_report = bool(common.get("coverage_report", True))
        self._init_coverage()
        self.total_generated = 0

    # ---- Config loading ----

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {p}")
        return data

    def _compute_allowed_layers(self, tf_targets, mode):
        try:
            return get_allowed_layers(tf_targets, mode)
        except Exception as e:
            logger.warning("TF capabilities unavailable: %s. Using defaults.", e)
            return frozenset(_DEFAULT_COVERAGE_LAYERS)

    # ---- Coverage ----

    def _init_coverage(self):
        skip = {"INPUT", "INPUT_SPEC", "OUTPUT_SPEC", "ASSERT", "DROPOUT", "IDENTITY"}
        self.coverage_stats = {l: 0 for l in sorted(self.allowed_layers - skip)}

    def _record(self, net: Net):
        for layer in net.layers:
            k = layer.kind.upper()
            if k in self.coverage_stats:
                self.coverage_stats[k] += 1

    def _uncovered(self) -> List[str]:
        return [l for l, c in self.coverage_stats.items() if c == 0]

    def _coverage_rate(self) -> float:
        total = len(self.coverage_stats)
        covered = sum(1 for c in self.coverage_stats.values() if c > 0)
        return (covered / total * 100) if total else 100.0

    # ---- Weight generation ----

    def _gen_weight(self, kind: str, meta: Dict[str, Any]) -> Optional[torch.Tensor]:
        if kind == "DENSE":
            return torch.randn(int(meta.get("out_features", 1)), int(meta.get("in_features", 1))) * 0.1
        if kind in ("CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D"):
            ic = int(meta.get("in_channels", 1))
            oc = int(meta.get("out_channels", 1))
            ks = meta.get("kernel_size", 3)
            ks = int(ks) if isinstance(ks, int) else int(ks[0])
            ndim = {"CONV1D": 1, "CONV2D": 2, "CONV3D": 3, "CONVTRANSPOSE2D": 2}[kind]
            # ConvTranspose2D: [in_ch, out_ch, k, ...]
            if kind == "CONVTRANSPOSE2D":
                shape = (ic, oc) + (ks,) * ndim
            else:
                shape = (oc, ic) + (ks,) * ndim
            return torch.randn(*shape) * 0.1
        return None

    # ---- INPUT_SPEC / ASSERT param generation ----

    def _input_spec_params(self, meta, input_shape, dtype):
        if meta["kind"] == InKind.BOX:
            return {
                "lb": torch.full(input_shape, float(meta.get("lb_val", 0.0)), dtype=dtype),
                "ub": torch.full(input_shape, float(meta.get("ub_val", 1.0)), dtype=dtype),
            }
        if meta["kind"] == InKind.LINF_BALL:
            center = torch.full(input_shape, float(meta.get("center_val", 0.5)), dtype=dtype)
            eps = float(meta.get("eps", 0.0))
            return {"center": center, "lb": center - eps, "ub": center + eps}
        raise ValueError(f"Unsupported INPUT_SPEC kind '{meta.get('kind')}'")

    def _assert_params(self, params, meta, dtype):
        kind = meta.get("kind")
        if kind == OutKind.LINEAR_LE and isinstance(params.get("c"), list):
            params["c"] = torch.as_tensor(params["c"], dtype=dtype)
        elif kind == OutKind.RANGE:
            for k in ("lb", "ub"):
                if isinstance(params.get(k), list):
                    params[k] = torch.as_tensor(params[k], dtype=dtype)
        return params

    # ---- Sampling ----

    def _sample_instance(self, idx: int) -> Dict[str, Any]:
        temp_id = f"{self.name_prefix}{self.base_seed}_idx{idx:05d}"
        seed = _derive_seed(self.base_seed, idx, temp_id)
        rng = random.Random(seed)
        family, model_cfg = self.sampler.sample_family(rng)
        nc = int(model_cfg["num_classes"])
        instance_id = self._semantic_name(family, model_cfg, seed)
        return {
            "instance_id": instance_id, "seed": seed, "family": family,
            "model_cfg": model_cfg,
            "input_spec": self.sampler.sample_input_spec(rng),
            "output_spec": self.sampler.sample_output_spec(rng, num_classes=nc),
        }

    # ---- Name generation ----

    def _semantic_name(self, family: str, cfg: Dict[str, Any], seed: int) -> str:
        variant = cfg.get("variant", "plain")
        family_tag = f"{family}_{variant}" if family != "cnn2d" or variant != "stage" else "resnet"
        dims = cfg["input_shape"][1:] if cfg["input_shape"][0] == 1 else cfg["input_shape"]
        input_str = "x".join(str(d) for d in dims)
        # Structure summary
        if family == "mlp":
            if variant == "plain":
                struct = "x".join(str(h) for h in cfg.get("hidden_sizes", ()))
            elif variant == "block":
                struct = f"{cfg.get('block_width', 64)}x{cfg.get('num_blocks', 3)}"
            else:
                struct = f"{cfg.get('residual_width', 128)}x{cfg.get('num_residual_blocks', 2)}"
        elif family == "cnn2d":
            if variant == "plain":
                struct = "x".join(str(c) for c in cfg.get("conv_channels", ()))
            elif variant == "residual":
                struct = f"{cfg.get('residual_channels', 32)}x{cfg.get('num_residual_blocks', 3)}"
            else:
                struct = f"{cfg.get('base_channels', 16)}x{cfg.get('stages', 3)}x{cfg.get('blocks_per_stage', 2)}"
        else:
            struct = "default"
        return f"{family_tag}_{input_str}_{struct}_{seed}"

    # ---- Build network spec ----

    def _build_spec(self, instance: Dict[str, Any], dtype: str) -> Dict[str, Any]:
        cfg = instance["model_cfg"]
        input_shape = list(cfg["input_shape"])
        nc = int(cfg["num_classes"])
        layers: List[Dict[str, Any]] = []

        # INPUT
        layers.append({"kind": "INPUT", "params": {}, "meta": {
            "shape": input_shape, "dtype": dtype, "num_classes": nc,
            "value_range": list(instance["input_spec"]["value_range"]),
        }})

        # INPUT_SPEC
        ik = str(instance["input_spec"]["kind"])
        sm: Dict[str, Any] = {"kind": ik}
        if ik == "BOX":
            sm["lb_val"] = float(instance["input_spec"]["lb_val"])
            sm["ub_val"] = float(instance["input_spec"]["ub_val"])
        elif ik == "LINF_BALL":
            sm["center_val"] = float(instance["input_spec"]["center_val"])
            sm["eps"] = float(instance["input_spec"]["eps"])
        layers.append({"kind": "INPUT_SPEC", "params": {}, "meta": sm})

        # Model layers
        if instance["family"] == "mlp":
            build_mlp_layers(layers, cfg=cfg)
        elif instance["family"] == "cnn2d":
            build_cnn_layers(layers, cfg=cfg, rng=random.Random(int(instance["seed"])))
        else:
            raise ValueError(f"Unsupported family: {instance['family']}")

        # ASSERT
        ok = str(instance["output_spec"]["kind"])
        om: Dict[str, Any] = {"kind": ok}
        op: Dict[str, Any] = {}
        if ok == "TOP1_ROBUST":
            om["y_true"] = int(instance["output_spec"]["y_true"])
        elif ok == "MARGIN_ROBUST":
            om["y_true"] = int(instance["output_spec"]["y_true"])
            om["margin"] = float(instance["output_spec"]["margin"])
        elif ok == "LINEAR_LE":
            op["c"] = list(instance["output_spec"]["c"])
            om["d"] = float(instance["output_spec"]["d"])
        elif ok == "RANGE":
            op["lb"] = list(instance["output_spec"]["lb"])
            op["ub"] = list(instance["output_spec"]["ub"])
        layers.append({"kind": "ASSERT", "params": op, "meta": om})

        return {"layers": layers}

    # ---- Create Net object ----

    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        """Create Net from spec. ACT Layer has no ``meta`` field — everything
        goes into ``params`` as a flat ``Dict[str, ParamValue]``."""
        dtype = get_default_dtype()
        dtype_str = str(dtype)
        layers: List[Layer] = []
        vc = 0

        for i, ls in enumerate(spec["layers"]):
            # In the spec dicts produced by _build_spec / layer_builder, metadata
            # lives under "meta" and tensor params under "params".  ACT's Layer
            # stores everything in a single ``params`` dict, so we merge them.
            raw_params = dict(ls.get("params", {}))
            meta = dict(ls.get("meta", {}))
            kind = ls["kind"]

            # Multi-input: resolve x_vars / y_vars
            if kind in _BINARY_KINDS:
                inputs = ls.get("inputs") or {}
                xs, ys = inputs.get("x"), inputs.get("y")
                if xs is not None and ys is not None:
                    meta["x_vars"] = list(layers[xs].out_vars)
                    meta["y_vars"] = list(layers[ys].out_vars)

            # CONCAT: resolve preds
            if kind == "CONCAT":
                preds = ls.get("preds", [])
                if not preds and len(layers) >= 2:
                    preds = [len(layers) - 2, len(layers) - 1]
                meta["preds_indices"] = preds

            # Store explicit preds in meta
            if "preds" in ls and "preds_indices" not in meta:
                meta["preds_indices"] = ls["preds"]

            # MAX/MIN: y_vars_list
            if kind in ("MAX", "MIN"):
                preds = ls.get("preds", [])
                if preds:
                    meta["y_vars_list"] = [list(layers[p].out_vars) for p in preds if p < len(layers)]

            # Generate variables (uses meta for shape lookups)
            in_vars, out_vars, vc = _generate_layer_variables(kind, i, vc, meta, layers)

            # Fill tensor parameters — ACT REGISTRY conventions:
            #   DENSE: "weight" (not W), "bias" (not b), requires in_features/out_features
            #   CONV*:  "weight", optional "bias", requires in/out_channels, kernel_size
            #   BIAS:  "bias" param
            #   SCALE: "weight" param
            #   BN:    "weight", "bias", "running_mean", "running_var"
            if kind == "INPUT":
                meta["dtype"] = dtype_str
            elif kind == "INPUT_SPEC":
                raw_params.update(self._input_spec_params(meta, layers[0].params["shape"], dtype))
            elif kind == "ASSERT":
                raw_params = self._assert_params(raw_params, meta, dtype)
            elif kind == "DENSE" and "weight" not in raw_params:
                inf = int(meta.get("in_features", 1))
                outf = int(meta.get("out_features", 1))
                raw_params["weight"] = torch.randn(outf, inf, dtype=dtype) * 0.1
                raw_params["in_features"] = inf
                raw_params["out_features"] = outf
                if meta.get("bias_enabled", meta.get("use_bias", True)):
                    raw_params["bias"] = torch.zeros(outf, dtype=dtype)
                # Remove builder-only keys not in REGISTRY
                meta.pop("bias_enabled", None)
            elif kind in ("CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D") and "weight" not in raw_params:
                w = self._gen_weight(kind, meta)
                if w is not None:
                    raw_params["weight"] = w
            elif kind == "BIAS" and "c" not in raw_params:
                raw_params["c"] = torch.zeros(len(in_vars), dtype=dtype)
            elif kind == "SCALE" and "a" not in raw_params:
                raw_params["a"] = torch.ones(len(in_vars), dtype=dtype)

            # Merge meta into params (ACT Layer has no separate meta field)
            # Strip builder-only keys that are not in ACT REGISTRY
            _BUILDER_ONLY_KEYS = {
                "preds_indices", "bias_enabled",
                "inject_binary_op", "inject_norm_op", "inject_shape_op",
            }
            for bk in _BUILDER_ONLY_KEYS:
                meta.pop(bk, None)

            # LRELU: TF code reads params["alpha"], REGISTRY has "negative_slope".
            # Store both so REGISTRY validation passes AND TF can find it.
            if kind == "LRELU" and "negative_slope" in meta:
                meta["alpha"] = meta["negative_slope"]

            params = {**meta, **raw_params}  # raw_params takes precedence

            layer = Layer(id=i, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars)
            layers.append(layer)

        # Build graph
        preds: Dict[int, List[int]] = {}
        for i, ls in enumerate(spec["layers"]):
            sp = ls.get("preds")
            preds[i] = list(sp) if sp is not None else ([i - 1] if i > 0 else [])

        succs: Dict[int, List[int]] = {i: [] for i in range(len(layers))}
        for i, pl in preds.items():
            for p in pl:
                succs[p].append(i)

        net = Net(layers=layers, preds=preds, succs=succs)
        net.meta = {"name": name}
        return net

    # ---- Save / manifest ----

    def save_network(self, net: Net, name: str) -> None:
        path = self.output_dir / f"{name}.json"
        d = NetSerializer.serialize_net(net)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  Saved: {path}")

    def _write_manifest(self, names: List[str]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_seed": self.base_seed,
            "num_instances": self.num_instances,
            "name_prefix": self.name_prefix,
            "nets": names,
            "tf_targets": self.tf_targets,
            "registry_mode": self.registry_mode,
            "allowed_layers_count": len(self.allowed_layers),
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- Coverage report ----

    def _print_coverage_report(self):
        if not self.coverage_report:
            return
        covered = sum(1 for c in self.coverage_stats.values() if c > 0)
        total = len(self.coverage_stats)
        rate = self._coverage_rate()
        print(f"\n{'=' * 60}")
        print("Layer Coverage Report")
        print(f"{'=' * 60}")
        if self.tf_targets:
            print(f"TF Targets: {self.tf_targets} (mode: {self.registry_mode})")
        print(f"Allowed: {len(self.allowed_layers)}  Trackable: {total}")
        print(f"Coverage: {covered}/{total} ({rate:.1f}%)  Networks: {self.total_generated}")
        uncov = self._uncovered()
        if uncov:
            print(f"\nUncovered ({len(uncov)}):")
            for l in sorted(uncov):
                print(f"  - {l}")
        else:
            print("\nAll target layers covered!")
        print(f"{'=' * 60}\n")

    # ---- Main generate entry point ----

    def generate(self) -> List[str]:
        # Clean output directory before generating (avoids stale nets from
        # previous runs with a different dtype or seed).
        for old in self.output_dir.glob("*.json"):
            old.unlink()
        # Also clean manifest
        if self.manifest_path.exists():
            self.manifest_path.unlink()

        dtype = str(self.config["common"]["dtype"])
        names: List[str] = []

        if self.coverage_mode == "full":
            print(f"Generating networks in FULL coverage mode (max {self.coverage_max_attempts} attempts)...")
            for idx in range(self.coverage_max_attempts):
                inst = self._sample_instance(idx)
                spec = self._build_spec(inst, dtype=dtype)
                net = self.create_network(inst["instance_id"], spec)
                self.save_network(net, inst["instance_id"])
                names.append(inst["instance_id"])
                self.total_generated += 1
                self._record(net)
                if (idx + 1) % 50 == 0:
                    print(f"  {idx + 1} generated, coverage: {self._coverage_rate():.1f}%, uncovered: {len(self._uncovered())}")
                if not self._uncovered():
                    print(f"\n  All layers covered after {idx + 1} networks!")
                    break

            # Fill remaining uncovered with minimal templates
            uncov = [l for l in self._uncovered() if l in self.allowed_layers]
            if uncov:
                print(f"\nGenerating minimal templates for {len(uncov)} uncovered layers...")
                for lk in sorted(uncov):
                    tmpl = generate_minimal_template(lk, dtype)
                    if tmpl:
                        try:
                            tname = f"template_{lk.lower()}_minimal"
                            tnet = self.create_network(tname, tmpl)
                            self.save_network(tnet, tname)
                            names.append(tname)
                            self.total_generated += 1
                            self._record(tnet)
                        except Exception as e:
                            print(f"  Failed template for {lk}: {e}")
                    else:
                        print(f"  Skipped {lk} (no template available)")
        else:
            print(f"Generating {self.num_instances} networks in BASIC mode...")
            for idx in range(self.num_instances):
                inst = self._sample_instance(idx)
                spec = self._build_spec(inst, dtype=dtype)
                net = self.create_network(inst["instance_id"], spec)
                self.save_network(net, inst["instance_id"])
                names.append(inst["instance_id"])
                self.total_generated += 1
                self._record(net)

        if self.write_manifest:
            self._write_manifest(names)

        print(f"\nAll networks saved to {self.output_dir}")
        self._print_coverage_report()
        return names
