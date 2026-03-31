#===- act/back_end/net_factory/layer_builder.py - Layer Construction ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Layer construction helpers for NetFactory.
#   - Build layer specs in-place (list of dicts).
#   - No randomness except the caller-provided rng for CNN variants.
#   - Does not create Layer/Net objects or serialize output.
#
# Optimization vs UCU:
#   - Unified N-D conv/pool (no 6 redundant wrappers)
#   - Data-driven minimal templates (replaces ~340 lines of hand-written specs)
#   - Cleaner validation (single _validate_spatial_params)
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Optional

# ============================================================================
# Internal Utility Functions
# ============================================================================

def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _ensure_batch1(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) < 2:
        raise ValueError(f"input_shape must include batch dim, got {shape}")
    if int(shape[0]) != 1:
        raise ValueError(f"Generator assumes batch=1, got {shape}")
    return tuple(int(x) for x in shape)


_ACTIVATION_MAP = {
    "relu": "RELU", "tanh": "TANH", "sigmoid": "SIGMOID",
    "lrelu": "LRELU", "relu6": "RELU6", "silu": "SILU",
    "gelu": "GELU", "abs": "ABS", "clip": "CLIP",
    "hardtanh": "HARDTANH", "hardsigmoid": "HARDSIGMOID",
    "hardswish": "HARDSWISH", "softplus": "SOFTPLUS",
    "mish": "MISH", "softsign": "SOFTSIGN",
}


def _activation_kind(name: str) -> str:
    name = (name or "relu").lower()
    if name not in _ACTIVATION_MAP:
        raise ValueError(f"Unsupported activation '{name}'. Available: {list(_ACTIVATION_MAP)}")
    return _ACTIVATION_MAP[name]


def _out_dim(x: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)


def _as_block_param(v: Any, i: int, n_blocks: int, name: str) -> int:
    if isinstance(v, int):
        return int(v)
    t = tuple(int(x) for x in v)
    if len(t) == 1:
        return int(t[0])
    if len(t) == n_blocks:
        return int(t[i])
    raise ValueError(f"{name} must be int or tuple of len 1 or len {n_blocks}, got len={len(t)}")


# ============================================================================
# Validation
# ============================================================================

def _validate_spatial_params(
    kind_label: str,
    ndim: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    in_ch: Optional[int] = None,
    groups: int = 1,
) -> None:
    """Validate conv/pool spatial parameters. Raises ValueError on invalid input."""
    if kernel <= 0:
        raise ValueError(f"{kind_label}: kernel must be positive, got {kernel}")
    if stride <= 0:
        raise ValueError(f"{kind_label}: stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"{kind_label}: padding must be non-negative, got {padding}")
    if dilation <= 0:
        raise ValueError(f"{kind_label}: dilation must be positive, got {dilation}")
    if groups <= 0:
        raise ValueError(f"{kind_label}: groups must be positive, got {groups}")
    if in_ch is not None and groups > 1 and in_ch % groups != 0:
        raise ValueError(f"{kind_label}: in_channels ({in_ch}) not divisible by groups ({groups})")

    dim_names = ["W", "H", "D"][:ndim][::-1]  # 1D→[W], 2D→[H,W], 3D→[D,H,W]
    eff_kernel = dilation * (kernel - 1) + 1
    for name, dim in zip(dim_names, spatial_dims):
        if dim <= 0:
            raise ValueError(f"{kind_label}: spatial dim {name} must be positive, got {dim}")
        if eff_kernel > dim + 2 * padding:
            raise ValueError(
                f"{kind_label}: effective kernel {eff_kernel} exceeds dim {name} "
                f"({dim} + 2*{padding} = {dim + 2 * padding})"
            )


# ============================================================================
# Unified N-D Convolution & Pooling
# ============================================================================

def append_conv_nd(
    layers: List[Dict[str, Any]], *,
    ndim: int, in_ch: int, out_ch: int,
    spatial_dims: Tuple[int, ...],
    kernel: int, stride: int, padding: int,
    dilation: int = 1, groups: int = 1,
) -> Tuple[int, ...]:
    """Append CONV{ndim}D layer. Returns output spatial dims."""
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
    if len(spatial_dims) != ndim:
        raise ValueError(f"spatial_dims length ({len(spatial_dims)}) != ndim ({ndim})")

    _validate_spatial_params(
        f"CONV{ndim}D", ndim, spatial_dims, kernel, stride, padding,
        dilation=dilation, in_ch=in_ch, groups=groups,
    )
    if groups > 1 and out_ch % groups != 0:
        raise ValueError(f"CONV{ndim}D: out_channels ({out_ch}) not divisible by groups ({groups})")

    output_spatial = tuple(_out_dim(d, kernel, stride, padding, dilation) for d in spatial_dims)
    for i, od in enumerate(output_spatial):
        if od <= 0:
            raise ValueError(f"CONV{ndim}D: output dim {i} = {od} (input={spatial_dims[i]})")

    layers.append({
        "kind": f"CONV{ndim}D",
        "params": {},
        "meta": {
            "in_channels": int(in_ch), "out_channels": int(out_ch),
            "kernel_size": int(kernel), "stride": int(stride),
            "padding": int(padding), "dilation": int(dilation), "groups": int(groups),
            "input_shape": [1, int(in_ch)] + [int(d) for d in spatial_dims],
            "output_shape": [1, int(out_ch)] + [int(d) for d in output_spatial],
        },
    })
    return output_spatial


def append_pool_nd(
    layers: List[Dict[str, Any]], *,
    ndim: int, kind: str, in_ch: int,
    spatial_dims: Tuple[int, ...],
    kernel: int, stride: int, padding: int = 0,
) -> Tuple[int, ...]:
    """Append MAXPOOL/AVGPOOL{ndim}D layer. Returns output spatial dims."""
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
    if len(spatial_dims) != ndim:
        raise ValueError(f"spatial_dims length ({len(spatial_dims)}) != ndim ({ndim})")

    _validate_spatial_params(kind, ndim, spatial_dims, kernel, stride, padding)

    output_spatial = tuple(_out_dim(d, kernel, stride, padding) for d in spatial_dims)
    for i, od in enumerate(output_spatial):
        if od <= 0:
            raise ValueError(f"{kind}: output dim {i} = {od} (input={spatial_dims[i]})")

    layers.append({
        "kind": kind, "params": {},
        "meta": {
            "kernel_size": int(kernel), "stride": int(stride), "padding": int(padding),
            "input_shape": [1, int(in_ch)] + [int(d) for d in spatial_dims],
            "output_shape": [1, int(in_ch)] + [int(d) for d in output_spatial],
        },
    })
    return output_spatial


# ============================================================================
# Single-Layer Appenders
# ============================================================================

def append_dense(layers, *, in_features: int, out_features: int, use_bias: bool) -> None:
    layers.append({
        "kind": "DENSE", "params": {},
        "meta": {"in_features": int(in_features), "out_features": int(out_features),
                 "bias_enabled": bool(use_bias)},
    })

def append_bias(layers, **meta_kw) -> None:
    layers.append({"kind": "BIAS", "params": {}, "meta": {k: v for k, v in meta_kw.items() if v is not None}})

def append_scale(layers, **meta_kw) -> None:
    layers.append({"kind": "SCALE", "params": {}, "meta": {k: v for k, v in meta_kw.items() if v is not None}})

def append_bn(layers, **meta_kw) -> None:
    """Emit SCALE + BIAS pair (ACT decomposes BatchNorm into these two layers)."""
    filtered = {k: v for k, v in meta_kw.items() if v is not None}
    append_scale(layers, **filtered)
    append_bias(layers, **filtered)

def append_act(layers, act_kind: str, *, act_params: Optional[Dict[str, Any]] = None) -> None:
    meta: Dict[str, Any] = {}
    if act_params:
        if act_kind == "LRELU" and "lrelu_alpha" in act_params:
            meta["negative_slope"] = float(act_params["lrelu_alpha"])
        elif act_kind == "POWER" and "power_exponent" in act_params:
            meta["exponent"] = float(act_params["power_exponent"])
    layers.append({"kind": act_kind, "params": {}, "meta": meta})

def append_add(layers, *, skip_idx: int, main_idx: int) -> None:
    layers.append({
        "kind": "ADD", "params": {}, "meta": {},
        "inputs": {"x": skip_idx, "y": main_idx},
        "preds": [skip_idx, main_idx],
    })

def append_binary_op(layers, *, op_kind: str, x_idx: int, y_idx: int) -> None:
    layers.append({
        "kind": op_kind, "params": {}, "meta": {},
        "inputs": {"x": x_idx, "y": y_idx},
        "preds": [x_idx, y_idx],
    })

def append_concat(layers, *, input_indices: List[int], concat_dim: int = 0) -> None:
    layers.append({
        "kind": "CONCAT", "params": {},
        "meta": {"concat_dim": concat_dim},
        "preds": input_indices,
    })

def append_flatten(layers) -> None:
    layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})


# ============================================================================
# TF-Driven Operator Injection
# ============================================================================

def _inject_extra_ops(
    layers: List[Dict[str, Any]], cfg: Dict[str, Any], feat_size: int,
    *, allow_dag: bool = True,
) -> None:
    """
    Inject extra operator layers into the network based on TF capabilities.

    Called by build_mlp_layers / build_cnn_layers just before the final
    classifier head.  The ``cfg`` dict may contain keys set by ConfigSampler:
      - inject_binary_op:  e.g. "MUL", "SUB" — fork-merge with previous layer
      - inject_norm_op:    e.g. "BIAS", "SCALE" — element-wise normalization
      - inject_shape_op:   e.g. "RESHAPE", "UNSQUEEZE" — shape transform pair

    Args:
        allow_dag: If False, skip binary ops (which create DAG fork-merge).
            Set to False for residual and CNN variants where act2torch
            cannot reliably restore DAG structure.
    """
    # Norm op: element-wise, inserted in-line (no shape change, no DAG)
    norm_op = cfg.get("inject_norm_op")
    if norm_op:
        layers.append({"kind": norm_op, "params": {}, "meta": {}})

    # Binary op: fork-merge creates a DAG — only inject when safe
    binary_op = cfg.get("inject_binary_op")
    if binary_op and allow_dag:
        branch1_idx = len(layers) - 1
        layers.append({"kind": "RELU", "params": {}, "meta": {}, "preds": [branch1_idx]})
        branch2_idx = len(layers) - 1
        append_binary_op(layers, op_kind=binary_op, x_idx=branch1_idx, y_idx=branch2_idx)

    # Shape op: identity-preserving pair (no DAG)
    shape_op = cfg.get("inject_shape_op")
    if shape_op in ("UNSQUEEZE", "SQUEEZE"):
        layers.append({"kind": "UNSQUEEZE", "params": {}, "meta": {"dims": [2]}})
        layers.append({"kind": "SQUEEZE", "params": {}, "meta": {"dims": [2]}})
    elif shape_op == "RESHAPE":
        layers.append({"kind": "RESHAPE", "params": {}, "meta": {"target_shape": [1, feat_size]}})
    elif shape_op == "TRANSPOSE":
        layers.append({"kind": "UNSQUEEZE", "params": {}, "meta": {"dims": [2]}})
        layers.append({"kind": "TRANSPOSE", "params": {}, "meta": {"perm": [0, 2, 1]}})
        layers.append({"kind": "TRANSPOSE", "params": {}, "meta": {"perm": [0, 2, 1]}})
        layers.append({"kind": "SQUEEZE", "params": {}, "meta": {"dims": [2]}})


# ============================================================================
# Network Builders
# ============================================================================

def build_mlp_layers(layers: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> None:
    """Build MLP layers. Supports *plain*, *block*, and *residual* variants."""
    shape = _ensure_batch1(tuple(cfg["input_shape"]))
    in_feat = int(shape[1]) if len(shape) == 2 else _prod(shape[1:])

    if len(shape) > 2:
        append_flatten(layers)

    act_kind = _activation_kind(cfg["activation"])
    use_bias = bool(cfg["use_bias"])
    variant = cfg["variant"]

    if variant == "plain":
        for h in cfg["hidden_sizes"]:
            append_dense(layers, in_features=in_feat, out_features=int(h), use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            in_feat = int(h)

    elif variant == "block":
        width = int(cfg["block_width"])
        append_dense(layers, in_features=in_feat, out_features=width, use_bias=use_bias)
        append_act(layers, act_kind, act_params=cfg)
        in_feat = width
        for _ in range(int(cfg["num_blocks"])):
            append_dense(layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            append_dense(layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias)
            if cfg.get("post_block_activation", True):
                append_act(layers, act_kind, act_params=cfg)

    elif variant == "residual":
        width = int(cfg["residual_width"])
        if in_feat != width:
            append_dense(layers, in_features=in_feat, out_features=width, use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            in_feat = width
        for _ in range(int(cfg["num_residual_blocks"])):
            skip_idx = len(layers) - 1
            append_dense(layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            append_dense(layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias)
            main_idx = len(layers) - 1
            append_add(layers, skip_idx=skip_idx, main_idx=main_idx)
            append_act(layers, act_kind, act_params=cfg)
    else:
        raise ValueError(f"Unsupported MLP variant '{variant}'")

    # Optional rare layers for coverage diversity (YAML-driven)
    if cfg.get("use_bias_layer", False):
        append_bias(layers)
    if cfg.get("use_scale_layer", False):
        append_scale(layers)
    if cfg.get("use_unsqueeze_squeeze", False):
        layers.append({"kind": "UNSQUEEZE", "params": {}, "meta": {"dims": [2]}})
        layers.append({"kind": "SQUEEZE", "params": {}, "meta": {"dims": [2]}})

    # TF-driven injected operators (set by ConfigSampler from allowed_layers)
    # Only plain MLP can safely use DAG injection; block/residual already have DAG
    _inject_extra_ops(layers, cfg, in_feat, allow_dag=(variant == "plain"))

    # Final classifier head
    append_dense(layers, in_features=in_feat, out_features=int(cfg["num_classes"]), use_bias=True)


def build_cnn_layers(
    layers: List[Dict[str, Any]], *,
    cfg: Dict[str, Any], rng: random.Random,
) -> None:
    """Build CNN layers. Supports *plain*, *residual*, and *stage* variants."""
    shape = _ensure_batch1(tuple(cfg["input_shape"]))
    if len(shape) != 4:
        raise ValueError(f"CNN2D expects (1,C,H,W), got {shape}")
    _, in_ch, H, W = (int(x) for x in shape)
    act_kind = _activation_kind(cfg["activation"])
    use_bn = cfg.get("use_batchnorm", False)
    use_transpose = cfg.get("use_transpose", False)

    variant = cfg.get("variant", "plain")

    if variant == "plain":
        n_blocks = len(cfg["conv_channels"])
        for i, out_ch in enumerate(cfg["conv_channels"]):
            out_ch = int(out_ch)
            k = _as_block_param(cfg["kernel_sizes"], i, n_blocks, "kernel_sizes")
            s = _as_block_param(cfg["strides"], i, n_blocks, "strides")
            p = _as_block_param(cfg["paddings"], i, n_blocks, "paddings")
            H, W = append_conv_nd(
                layers, ndim=2, in_ch=in_ch, out_ch=out_ch,
                spatial_dims=(H, W), kernel=k, stride=s, padding=p,
            )
            if use_bn and i == 0:
                append_bn(layers)
            append_act(layers, act_kind, act_params=cfg)
            in_ch = out_ch

            if use_transpose and i == 0 and H == W:
                layers.append({"kind": "TRANSPOSE", "params": {}, "meta": {"perm": [0, 1, 3, 2]}})
                layers.append({"kind": "TRANSPOSE", "params": {}, "meta": {"perm": [0, 1, 3, 2]}})

            use_pooling = cfg.get("use_pooling", cfg.get("use_maxpool", False))
            if use_pooling:
                pool_kind_name = cfg.get("pool_kind", "maxpool")
                pool_type = {"maxpool": "MAXPOOL2D", "avgpool": "AVGPOOL2D"}.get(pool_kind_name)
                if pool_type:
                    pk = int(cfg.get("pool_kernel", 2))
                    ps = int(cfg.get("pool_stride", 2))
                    H, W = append_pool_nd(
                        layers, ndim=2, kind=pool_type, in_ch=in_ch,
                        spatial_dims=(H, W), kernel=pk, stride=ps, padding=0,
                    )

        append_flatten(layers)
        feat = in_ch * H * W
        append_dense(layers, in_features=feat, out_features=int(cfg["fc_hidden"]), use_bias=True)
        append_act(layers, act_kind, act_params=cfg)
        if cfg.get("use_scale_layer", False):
            append_scale(layers)
        # TF-driven injected operators (no DAG in CNN — act2torch can't restore it)
        _inject_extra_ops(layers, cfg, int(cfg["fc_hidden"]), allow_dag=False)
        append_dense(layers, in_features=int(cfg["fc_hidden"]), out_features=int(cfg["num_classes"]), use_bias=True)

    elif variant == "residual":
        ch = int(cfg["residual_channels"])
        H, W = append_conv_nd(layers, ndim=2, in_ch=in_ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
        append_act(layers, act_kind, act_params=cfg)
        for _ in range(int(cfg["num_residual_blocks"])):
            skip_idx = len(layers) - 1
            H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
            append_act(layers, act_kind, act_params=cfg)
            H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
            main_idx = len(layers) - 1
            append_add(layers, skip_idx=skip_idx, main_idx=main_idx)
            append_act(layers, act_kind, act_params=cfg)
        while H > 1 or W > 1:
            H, W = append_pool_nd(layers, ndim=2, kind="AVGPOOL2D", in_ch=ch, spatial_dims=(H, W), kernel=2, stride=2, padding=0)
            if H <= 0 or W <= 0:
                raise ValueError("Invalid spatial dims after head pooling")
        append_flatten(layers)
        append_dense(layers, in_features=ch * H * W, out_features=int(cfg["num_classes"]), use_bias=True)

    elif variant == "stage":
        ch = int(cfg["base_channels"])
        H, W = append_conv_nd(layers, ndim=2, in_ch=in_ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
        append_act(layers, act_kind, act_params=cfg)
        for stage in range(int(cfg["stages"])):
            if stage > 0:
                next_ch = min(64, ch * int(cfg["channel_mult"]))
                ds = cfg.get("downsample", "maxpool")
                if ds == "stride2_conv":
                    H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=next_ch, spatial_dims=(H, W), kernel=3, stride=2, padding=1)
                    append_act(layers, act_kind, act_params=cfg)
                    ch = next_ch
                else:
                    pool_type = "MAXPOOL2D" if ds == "maxpool" else "AVGPOOL2D"
                    H, W = append_pool_nd(layers, ndim=2, kind=pool_type, in_ch=ch, spatial_dims=(H, W), kernel=2, stride=2, padding=0)
                    if next_ch != ch:
                        H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=next_ch, spatial_dims=(H, W), kernel=1, stride=1, padding=0)
                        append_act(layers, act_kind, act_params=cfg)
                        ch = next_ch
            for _ in range(int(cfg["blocks_per_stage"])):
                if rng.random() < float(cfg.get("double_conv_p", 0.5)):
                    H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
                    append_act(layers, act_kind, act_params=cfg)
                    H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
                    append_act(layers, act_kind, act_params=cfg)
                else:
                    H, W = append_conv_nd(layers, ndim=2, in_ch=ch, out_ch=ch, spatial_dims=(H, W), kernel=3, stride=1, padding=1)
                    append_act(layers, act_kind, act_params=cfg)
        if cfg.get("head_pool_to_1x1", True):
            while H > 1 or W > 1:
                H, W = append_pool_nd(layers, ndim=2, kind="AVGPOOL2D", in_ch=ch, spatial_dims=(H, W), kernel=2, stride=2, padding=0)
                if H <= 0 or W <= 0:
                    raise ValueError("Invalid spatial dims after head pooling")
        append_flatten(layers)
        append_dense(layers, in_features=ch * H * W, out_features=int(cfg["num_classes"]), use_bias=True)
    else:
        raise ValueError(f"Unsupported CNN variant '{variant}'")


# ============================================================================
# Data-Driven Minimal Templates (optimization: replaces ~340 lines)
# ============================================================================

# Each entry: (input_shape, layer_specs)
# layer_specs items: dict or str shorthand
# "${LAYER}" is replaced by target layer_kind during expansion
_MINIMAL_TEMPLATE_DEFS = {
    # Element-wise ops: INPUT → INPUT_SPEC → layer → ASSERT
    "elementwise_1d": {
        "layers": [
            "BIAS", "SCALE", "BN", "RELU", "SIGMOID", "TANH", "LRELU",
            "RELU6", "HARDTANH", "HARDSIGMOID", "HARDSWISH", "SILU",
            "SOFTPLUS", "MISH", "SOFTSIGN", "ABS", "CLIP", "SQUARE", "POWER",
        ],
        "input_shape": [1, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            "${LAYER}",
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # DENSE
    "dense": {
        "layers": ["DENSE"],
        "input_shape": [1, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "DENSE", "meta": {"in_features": 4, "out_features": 4, "bias_enabled": True}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # 2D CNN layers
    "cnn_2d": {
        "layers": ["CONV2D", "MAXPOOL2D", "AVGPOOL2D", "PAD", "UPSAMPLE"],
        "input_shape": [1, 1, 4, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            "${LAYER}",
            {"kind": "FLATTEN"},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # 1D CNN layers
    "cnn_1d": {
        "layers": ["CONV1D", "MAXPOOL1D", "AVGPOOL1D"],
        "input_shape": [1, 1, 8],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            "${LAYER}",
            {"kind": "FLATTEN"},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # 3D CNN layers
    "cnn_3d": {
        "layers": ["CONV3D", "MAXPOOL3D"],
        "input_shape": [1, 1, 4, 4, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            "${LAYER}",
            {"kind": "FLATTEN"},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # FLATTEN
    "flatten": {
        "layers": ["FLATTEN"],
        "input_shape": [1, 2, 2],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "FLATTEN"},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # Binary fork ops: INPUT → SPEC → RELU → SIGMOID → op(2,3) → ASSERT
    "binary_fork": {
        "layers": ["ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN"],
        "input_shape": [1, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "RELU"},
            {"kind": "SIGMOID"},
            {"kind": "${LAYER}", "inputs": {"x": 2, "y": 3}, "preds": [2, 3]},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # CONCAT
    "concat": {
        "layers": ["CONCAT"],
        "input_shape": [1, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "RELU"},
            {"kind": "SIGMOID"},
            {"kind": "CONCAT", "meta": {"concat_dim": 0}, "preds": [2, 3]},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    # Shape ops
    "reshape": {
        "layers": ["RESHAPE"],
        "input_shape": [1, 2, 2],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "RESHAPE", "meta": {"target_shape": [1, 4]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "transpose": {
        "layers": ["TRANSPOSE"],
        "input_shape": [1, 2, 3],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "TRANSPOSE", "meta": {"perm": [0, 2, 1]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "squeeze": {
        "layers": ["SQUEEZE"],
        "input_shape": [1, 1, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "SQUEEZE", "meta": {"dims": [1]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "unsqueeze": {
        "layers": ["UNSQUEEZE"],
        "input_shape": [1, 4],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "UNSQUEEZE", "meta": {"dims": [1]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "slice": {
        "layers": ["SLICE"],
        "input_shape": [1, 8],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "SLICE", "meta": {"starts": [0], "ends": [4], "axes": [1], "input_shape": [1, 8], "output_shape": [1, 4]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "gather": {
        "layers": ["GATHER"],
        "input_shape": [1, 8],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "GATHER", "meta": {"indices": [0, 2, 4], "axis": 1, "input_shape": [1, 8], "output_shape": [1, 3]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "index_select": {
        "layers": ["INDEX_SELECT"],
        "input_shape": [1, 8],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "INDEX_SELECT", "meta": {"indices": [0, 2, 4], "dim": 1, "input_shape": [1, 8], "output_shape": [1, 3]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "tile": {
        "layers": ["TILE"],
        "input_shape": [1, 2],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "TILE", "meta": {"repeats": [1, 2], "input_shape": [1, 2], "output_shape": [1, 4]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "expand": {
        "layers": ["EXPAND"],
        "input_shape": [1, 1],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "EXPAND", "meta": {"shape": [1, 1], "input_shape": [1, 1], "output_shape": [1, 1]}},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
    "convtranspose2d": {
        "layers": ["CONVTRANSPOSE2D"],
        "input_shape": [1, 1, 2, 2],
        "spec": [
            {"kind": "INPUT"},
            {"kind": "INPUT_SPEC", "meta": {"kind": "BOX"}},
            {"kind": "CONVTRANSPOSE2D", "meta": {
                "in_channels": 1, "out_channels": 1, "kernel_size": 3,
                "stride": 2, "padding": 1, "output_padding": 1,
                "input_shape": [1, 1, 2, 2], "output_shape": [1, 1, 4, 4],
            }},
            {"kind": "FLATTEN"},
            {"kind": "ASSERT", "meta": {"kind": "TOP1_ROBUST", "y_true": 0}},
        ],
    },
}

# Pre-build reverse index: layer_kind -> (template_key, input_shape)
_LAYER_TO_TEMPLATE: Dict[str, Tuple[str, List[int]]] = {}
for _tkey, _tdef in _MINIMAL_TEMPLATE_DEFS.items():
    for _lk in _tdef["layers"]:
        _LAYER_TO_TEMPLATE[_lk] = (_tkey, _tdef["input_shape"])

# CNN meta defaults for template expansion
_CNN_META_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "CONV2D":    {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1,
                  "input_shape": [1, 1, 4, 4], "output_shape": [1, 1, 4, 4]},
    "CONV1D":    {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1,
                  "input_shape": [1, 1, 8], "output_shape": [1, 1, 8]},
    "CONV3D":    {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1,
                  "input_shape": [1, 1, 4, 4, 4], "output_shape": [1, 1, 4, 4, 4]},
    "MAXPOOL2D": {"kernel_size": 2, "stride": 2, "padding": 0,
                  "input_shape": [1, 1, 4, 4], "output_shape": [1, 1, 2, 2]},
    "MAXPOOL1D": {"kernel_size": 2, "stride": 2, "padding": 0,
                  "input_shape": [1, 1, 8], "output_shape": [1, 1, 4]},
    "MAXPOOL3D": {"kernel_size": 2, "stride": 2, "padding": 0,
                  "input_shape": [1, 1, 4, 4, 4], "output_shape": [1, 1, 2, 2, 2]},
    "AVGPOOL2D": {"kernel_size": 2, "stride": 2, "padding": 0,
                  "input_shape": [1, 1, 4, 4], "output_shape": [1, 1, 2, 2]},
    "AVGPOOL1D": {"kernel_size": 2, "stride": 2, "padding": 0,
                  "input_shape": [1, 1, 8], "output_shape": [1, 1, 4]},
    "PAD":       {"pad": [1, 1, 1, 1], "mode": "constant", "value": 0.0,
                  "input_shape": [1, 1, 4, 4], "output_shape": [1, 1, 6, 6]},
    "UPSAMPLE":  {"scale_factor": 2.0, "mode": "nearest",
                  "input_shape": [1, 1, 4, 4], "output_shape": [1, 1, 8, 8]},
}


def generate_minimal_template(layer_kind: str, dtype: str) -> Optional[Dict[str, Any]]:
    """
    Generate a minimal network spec dict for *layer_kind*.

    Returns None if no template is available for this layer.
    """
    if layer_kind not in _LAYER_TO_TEMPLATE:
        return None

    tkey, input_shape = _LAYER_TO_TEMPLATE[layer_kind]
    tdef = _MINIMAL_TEMPLATE_DEFS[tkey]

    result_layers: List[Dict[str, Any]] = []
    for item in tdef["spec"]:
        if isinstance(item, str) and item == "${LAYER}":
            # Expand placeholder with CNN meta defaults if available
            meta = dict(_CNN_META_DEFAULTS.get(layer_kind, {}))
            result_layers.append({"kind": layer_kind, "params": {}, "meta": meta})
        elif isinstance(item, dict):
            entry = {"kind": item["kind"], "params": dict(item.get("params", {})),
                     "meta": dict(item.get("meta", {}))}
            # Substitute ${LAYER} in kind field (for binary_fork template)
            if entry["kind"] == "${LAYER}":
                entry["kind"] = layer_kind
            # Copy extra keys (inputs, preds)
            for extra in ("inputs", "preds"):
                if extra in item:
                    entry[extra] = item[extra]
            # Set INPUT shape and dtype
            if entry["kind"] == "INPUT":
                entry["meta"]["shape"] = list(input_shape)
                entry["meta"]["dtype"] = dtype
            result_layers.append(entry)

    return {"layers": result_layers}
