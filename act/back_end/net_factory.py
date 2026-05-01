# ===- act/back_end/net_factory.py - NetFactory + Layer Builders ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Generates ACT networks from a YAML config (config_gen_act_net.yaml).
#
# Usage:
#   python -m act.back_end --generate                   # default 15 nets
#   python -m act.back_end --generate --num 50          # custom count
#   python -m act.back_end --generate --base-seed 42    # reproducible
#
# YAML rules: {choice}, {range}, {weighted}, {repeat}, {probability}, {const}
# Families:   mlp (plain/block/residual), cnn2d (plain/residual/stage)
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
import logging
import random
import secrets
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import torch  # pyright: ignore[reportMissingImports]
import yaml

from act.back_end.core import Layer, Net
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.serialization.serialization import NetSerializer
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_dtype

logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions
# ============================================================================


_ACTIVATIONS: FrozenSet[LayerKind] = frozenset(
    {
        LayerKind.RELU,
        LayerKind.TANH,
        LayerKind.SIGMOID,
        LayerKind.LRELU,
        LayerKind.RELU6,
        LayerKind.SILU,
        LayerKind.GELU,
        LayerKind.ABS,
        LayerKind.CLIP,
        LayerKind.HARDTANH,
        LayerKind.HARDSIGMOID,
        LayerKind.HARDSWISH,
        LayerKind.SOFTPLUS,
        LayerKind.MISH,
        LayerKind.SOFTSIGN,
    }
)


def validate_factory_schema_alignment(net: "Net") -> None:
    errors: List[str] = []
    for layer in net.layers:
        kind = layer.kind
        if kind not in REGISTRY:
            errors.append(f"Layer {layer.id}: kind '{kind}' not in REGISTRY")
            continue
        allowed = set(
            REGISTRY[kind]["params_required"] + REGISTRY[kind]["params_optional"]
        )
        for pk in layer.params:
            if pk not in allowed:
                errors.append(
                    f"Layer {layer.id}: param '{pk}' for kind '{kind}' not in REGISTRY. "
                    f"Allowed: {sorted(allowed)}"
                )
    if errors:
        raise AssertionError(
            f"Factory / REGISTRY misalignment ({len(errors)} issue(s)):\n  - "
            + "\n  - ".join(errors)
        )


def _activation_kind(name: str) -> str:
    key = (name or "relu").upper()
    try:
        kind = LayerKind(key)
    except ValueError:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {sorted(lk.value for lk in _ACTIVATIONS)}"
        )
    if kind not in _ACTIVATIONS:
        raise ValueError(
            f"'{name}' is not a supported activation. "
            f"Available: {sorted(lk.value for lk in _ACTIVATIONS)}"
        )
    return kind.value


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
    raise ValueError(
        f"{name} must be int or tuple of len 1 or len {n_blocks}, got len={len(t)}"
    )


# ============================================================================
# Unified N-D Convolution & Pooling
# ============================================================================


def append_conv_nd(
    layers: List[Dict[str, Any]],
    *,
    ndim: int,
    in_ch: int,
    out_ch: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[int, ...]:
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
    if len(spatial_dims) != ndim:
        raise ValueError(f"spatial_dims length ({len(spatial_dims)}) != ndim ({ndim})")

    output_spatial = tuple(
        _out_dim(d, kernel, stride, padding, dilation) for d in spatial_dims
    )
    for i, od in enumerate(output_spatial):
        if od <= 0:
            raise ValueError(
                f"CONV{ndim}D: output dim {i} = {od} (input={spatial_dims[i]})"
            )

    layers.append(
        {
            "kind": f"CONV{ndim}D",
            "params": {
                "in_channels": int(in_ch),
                "out_channels": int(out_ch),
                "kernel_size": int(kernel),
                "stride": int(stride),
                "padding": int(padding),
                "dilation": int(dilation),
                "groups": int(groups),
                "input_shape": [1, int(in_ch)] + [int(d) for d in spatial_dims],
                "output_shape": [1, int(out_ch)] + [int(d) for d in output_spatial],
            },
        }
    )
    return output_spatial


def append_pool_nd(
    layers: List[Dict[str, Any]],
    *,
    ndim: int,
    kind: str,
    in_ch: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, ...]:
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
    if len(spatial_dims) != ndim:
        raise ValueError(f"spatial_dims length ({len(spatial_dims)}) != ndim ({ndim})")

    output_spatial = tuple(_out_dim(d, kernel, stride, padding) for d in spatial_dims)
    for i, od in enumerate(output_spatial):
        if od <= 0:
            raise ValueError(f"{kind}: output dim {i} = {od} (input={spatial_dims[i]})")

    layers.append(
        {
            "kind": kind,
            "params": {
                "kernel_size": int(kernel),
                "stride": int(stride),
                "padding": int(padding),
                "input_shape": [1, int(in_ch)] + [int(d) for d in spatial_dims],
                "output_shape": [1, int(in_ch)] + [int(d) for d in output_spatial],
            },
        }
    )
    return output_spatial


# ============================================================================
# Single-Layer Appenders
# ============================================================================


def append_dense(
    layers, *, in_features: int, out_features: int, use_bias: bool
) -> None:
    layers.append(
        {
            "kind": LayerKind.DENSE.value,
            "params": {
                "in_features": int(in_features),
                "out_features": int(out_features),
                "use_bias": bool(use_bias),
            },
        }
    )


def append_bias(layers, **meta_kw) -> None:
    layers.append(
        {
            "kind": LayerKind.BIAS.value,
            "params": {k: v for k, v in meta_kw.items() if v is not None},
        }
    )


def append_scale(layers, **meta_kw) -> None:
    layers.append(
        {
            "kind": LayerKind.SCALE.value,
            "params": {k: v for k, v in meta_kw.items() if v is not None},
        }
    )


def append_bn(layers, **meta_kw) -> None:
    filtered = {k: v for k, v in meta_kw.items() if v is not None}
    append_scale(layers, **filtered)
    append_bias(layers, **filtered)


def append_act(
    layers, act_kind: str, *, act_params: Optional[Dict[str, Any]] = None
) -> None:
    params: Dict[str, Any] = {}
    if act_params:
        if act_kind == LayerKind.LRELU.value and "lrelu_alpha" in act_params:
            params["negative_slope"] = float(act_params["lrelu_alpha"])
        elif act_kind == LayerKind.POW.value and "power_exponent" in act_params:
            params["exponent"] = float(act_params["power_exponent"])
    layers.append({"kind": act_kind, "params": params})


def append_add(layers, *, skip_idx: int, main_idx: int) -> None:
    append_binary_op(
        layers, op_kind=LayerKind.ADD.value, x_idx=skip_idx, y_idx=main_idx
    )


def append_binary_op(layers, *, op_kind: str, x_idx: int, y_idx: int) -> None:
    layers.append(
        {
            "kind": op_kind,
            "params": {},
            "inputs": {"x": x_idx, "y": y_idx},
            "preds": [x_idx, y_idx],
        }
    )


def append_concat(layers, *, input_indices: List[int], concat_dim: int = 0) -> None:
    layers.append(
        {
            "kind": LayerKind.CONCAT.value,
            "params": {"concat_dim": concat_dim},
            "preds": input_indices,
        }
    )


def append_flatten(layers) -> None:
    layers.append({"kind": LayerKind.FLATTEN.value, "params": {"start_dim": 1}})


def append_rnn_family(
    layers: List[Dict[str, Any]],
    *,
    cell: str,
    input_size: int,
    hidden_size: int,
    seq_len: int,
    batch: int = 1,
    num_layers: int = 1,
    bidirectional: bool = False,
    batch_first: bool = True,
    use_bias: bool = True,
    nonlinearity: str = "tanh",
) -> Tuple[Tuple[int, ...], int]:
    """Append an RNN / LSTM / GRU layer to the spec list.

    Returns ``(output_shape, hidden_out)`` where ``hidden_out = hidden_size *
    (2 if bidirectional else 1)``. Output layout follows ``batch_first``:
    ``(B, T, hidden_out)`` if True else ``(T, B, hidden_out)``. Weights are
    populated downstream by NetFactory.create_network.
    """
    cell = cell.upper()
    if cell not in (LayerKind.RNN.value, LayerKind.LSTM.value, LayerKind.GRU.value):
        raise ValueError(f"append_rnn_family: unknown cell {cell!r}")
    directions = 2 if bidirectional else 1
    hidden_out = int(hidden_size) * directions
    if batch_first:
        input_shape = [int(batch), int(seq_len), int(input_size)]
        output_shape = [int(batch), int(seq_len), int(hidden_out)]
    else:
        input_shape = [int(seq_len), int(batch), int(input_size)]
        output_shape = [int(seq_len), int(batch), int(hidden_out)]
    params: Dict[str, Any] = {
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "bidirectional": bool(bidirectional),
        "batch_first": bool(batch_first),
        "input_shape": input_shape,
        "output_shape": output_shape,
        # Carries through to weight generation; not consumed by interval TF.
        "use_bias": bool(use_bias),
    }
    if cell == LayerKind.RNN.value:
        params["nonlinearity"] = str(nonlinearity)
    layers.append({"kind": cell, "params": params})
    return tuple(output_shape), hidden_out


# ============================================================================
# TF-Driven Operator Injection
# ============================================================================


def _inject_extra_ops(
    layers: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    feat_size: int,
    *,
    allow_dag: bool = True,
) -> None:
    norm_op = cfg.get("inject_norm_op")
    if norm_op:
        layers.append({"kind": norm_op, "params": {}})

    binary_op = cfg.get("inject_binary_op")
    if binary_op and allow_dag:
        branch1_idx = len(layers) - 1
        layers.append(
            {
                "kind": LayerKind.RELU.value,
                "params": {},
                "preds": [branch1_idx],
            }
        )
        branch2_idx = len(layers) - 1
        append_binary_op(
            layers, op_kind=binary_op, x_idx=branch1_idx, y_idx=branch2_idx
        )

    shape_op = cfg.get("inject_shape_op")
    if shape_op in (LayerKind.UNSQUEEZE.value, LayerKind.SQUEEZE.value):
        layers.append({"kind": LayerKind.UNSQUEEZE.value, "params": {"dims": [2]}})
        layers.append({"kind": LayerKind.SQUEEZE.value, "params": {"dims": [2]}})
    elif shape_op == LayerKind.RESHAPE.value:
        layers.append(
            {
                "kind": LayerKind.RESHAPE.value,
                "params": {"target_shape": [1, feat_size]},
            }
        )
    elif shape_op == LayerKind.TRANSPOSE.value:
        layers.append({"kind": LayerKind.UNSQUEEZE.value, "params": {"dims": [2]}})
        layers.append(
            {
                "kind": LayerKind.TRANSPOSE.value,
                "params": {"perm": [0, 2, 1]},
            }
        )
        layers.append(
            {
                "kind": LayerKind.TRANSPOSE.value,
                "params": {"perm": [0, 2, 1]},
            }
        )
        layers.append({"kind": LayerKind.SQUEEZE.value, "params": {"dims": [2]}})


# ============================================================================
# Network Builders
# ============================================================================


def build_mlp_layers(layers: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> None:
    shape = tuple(cfg["input_shape"])
    in_feat = int(shape[1]) if len(shape) == 2 else math.prod(shape[1:])

    if len(shape) > 2:
        append_flatten(layers)

    act_kind = _activation_kind(cfg["activation"])
    use_bias = bool(cfg["use_bias"])
    variant = cfg["variant"]

    if variant == "plain":
        for h in cfg["hidden_sizes"]:
            append_dense(
                layers, in_features=in_feat, out_features=int(h), use_bias=use_bias
            )
            append_act(layers, act_kind, act_params=cfg)
            in_feat = int(h)

    elif variant == "block":
        width = int(cfg["block_width"])
        append_dense(layers, in_features=in_feat, out_features=width, use_bias=use_bias)
        append_act(layers, act_kind, act_params=cfg)
        in_feat = width
        for _ in range(int(cfg["num_blocks"])):
            append_dense(
                layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias
            )
            append_act(layers, act_kind, act_params=cfg)
            append_dense(
                layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias
            )
            if cfg.get("post_block_activation", True):
                append_act(layers, act_kind, act_params=cfg)

    elif variant == "residual":
        width = int(cfg["residual_width"])
        if in_feat != width:
            append_dense(
                layers, in_features=in_feat, out_features=width, use_bias=use_bias
            )
            append_act(layers, act_kind, act_params=cfg)
            in_feat = width
        for _ in range(int(cfg["num_residual_blocks"])):
            skip_idx = len(layers) - 1
            append_dense(
                layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias
            )
            append_act(layers, act_kind, act_params=cfg)
            append_dense(
                layers, in_features=in_feat, out_features=in_feat, use_bias=use_bias
            )
            main_idx = len(layers) - 1
            append_add(layers, skip_idx=skip_idx, main_idx=main_idx)
            append_act(layers, act_kind, act_params=cfg)
    else:
        raise ValueError(f"Unsupported MLP variant '{variant}'")

    if cfg.get("use_bias_layer", False):
        append_bias(layers)
    if cfg.get("use_scale_layer", False):
        append_scale(layers)
    if cfg.get("use_unsqueeze_squeeze", False):
        layers.append({"kind": LayerKind.UNSQUEEZE.value, "params": {"dims": [2]}})
        layers.append({"kind": LayerKind.SQUEEZE.value, "params": {"dims": [2]}})

    _inject_extra_ops(layers, cfg, in_feat, allow_dag=(variant == "plain"))

    append_dense(
        layers, in_features=in_feat, out_features=int(cfg["num_classes"]), use_bias=True
    )


def build_rnn_layers(layers: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> None:
    """Generate a recurrent network: RNN/LSTM/GRU cell + Flatten + Dense head.

    Layout (with batch_first=True, single direction):
      INPUT (1, T, F)
        -> RNN/LSTM/GRU         (1, T, H)
        -> FLATTEN              (1, T*H)
        -> DENSE(num_classes)   (1, C)

    For bidirectional, hidden width doubles to 2H so the Dense input becomes
    T*2H. The cell type is chosen via ``cfg["cell"]`` (RNN/LSTM/GRU).
    """
    shape = tuple(cfg["input_shape"])
    if len(shape) != 3:
        raise ValueError(f"RNN family expects input_shape (1, T, F); got {shape}")
    batch, seq_len, in_feat = (int(s) for s in shape)
    if batch != 1:
        raise ValueError(f"RNN family currently restricted to batch=1; got batch={batch}")

    cell = str(cfg.get("cell", LayerKind.LSTM.value)).upper()
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg.get("num_layers", 1))
    bidirectional = bool(cfg.get("bidirectional", False))
    batch_first = bool(cfg.get("batch_first", True))
    nonlinearity = str(cfg.get("nonlinearity", "tanh"))
    use_bias = bool(cfg.get("use_bias", True))

    _, hidden_out = append_rnn_family(
        layers,
        cell=cell,
        input_size=in_feat,
        hidden_size=hidden_size,
        seq_len=seq_len,
        batch=batch,
        num_layers=num_layers,
        bidirectional=bidirectional,
        batch_first=batch_first,
        use_bias=use_bias,
        nonlinearity=nonlinearity,
    )

    flat_size = seq_len * hidden_out
    append_flatten(layers)
    append_dense(
        layers,
        in_features=flat_size,
        out_features=int(cfg["num_classes"]),
        use_bias=True,
    )


_POOL_KIND_TO_LAYER = {
    "maxpool": LayerKind.MAXPOOL2D.value,
    "avgpool": LayerKind.AVGPOOL2D.value,
}


def _conv2d(layers, *, in_ch, out_ch, H, W, kernel=3, stride=1, padding=1):
    return append_conv_nd(
        layers,
        ndim=2,
        in_ch=in_ch,
        out_ch=out_ch,
        spatial_dims=(H, W),
        kernel=kernel,
        stride=stride,
        padding=padding,
    )


def _pool2d(layers, *, kind, in_ch, H, W, kernel=2, stride=2, padding=0):
    return append_pool_nd(
        layers,
        ndim=2,
        kind=kind,
        in_ch=in_ch,
        spatial_dims=(H, W),
        kernel=kernel,
        stride=stride,
        padding=padding,
    )


def build_cnn_layers(
    layers: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    rng: random.Random,
) -> None:
    shape = tuple(cfg["input_shape"])
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
            H, W = _conv2d(
                layers,
                in_ch=in_ch,
                out_ch=out_ch,
                H=H,
                W=W,
                kernel=k,
                stride=s,
                padding=p,
            )
            if use_bn and i == 0:
                append_bn(layers)
            append_act(layers, act_kind, act_params=cfg)
            in_ch = out_ch
            if use_transpose and i == 0 and H == W:
                layers.append(
                    {
                        "kind": LayerKind.TRANSPOSE.value,
                        "params": {"perm": [0, 1, 3, 2]},
                    }
                )
                layers.append(
                    {
                        "kind": LayerKind.TRANSPOSE.value,
                        "params": {"perm": [0, 1, 3, 2]},
                    }
                )
            if cfg.get("use_pooling", cfg.get("use_maxpool", False)):
                pool_type = _POOL_KIND_TO_LAYER.get(cfg.get("pool_kind", "maxpool"))
                if pool_type:
                    pk, ps = (
                        int(cfg.get("pool_kernel", 2)),
                        int(cfg.get("pool_stride", 2)),
                    )
                    H, W = _pool2d(
                        layers,
                        kind=pool_type,
                        in_ch=in_ch,
                        H=H,
                        W=W,
                        kernel=pk,
                        stride=ps,
                    )
        append_flatten(layers)
        feat = in_ch * H * W
        append_dense(
            layers, in_features=feat, out_features=int(cfg["fc_hidden"]), use_bias=True
        )
        append_act(layers, act_kind, act_params=cfg)
        if cfg.get("use_scale_layer", False):
            append_scale(layers)
        _inject_extra_ops(layers, cfg, int(cfg["fc_hidden"]), allow_dag=False)
        append_dense(
            layers,
            in_features=int(cfg["fc_hidden"]),
            out_features=int(cfg["num_classes"]),
            use_bias=True,
        )

    elif variant == "residual":
        ch = int(cfg["residual_channels"])
        H, W = _conv2d(layers, in_ch=in_ch, out_ch=ch, H=H, W=W)
        append_act(layers, act_kind, act_params=cfg)
        for _ in range(int(cfg["num_residual_blocks"])):
            skip_idx = len(layers) - 1
            H, W = _conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W)
            append_act(layers, act_kind, act_params=cfg)
            H, W = _conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W)
            append_add(layers, skip_idx=skip_idx, main_idx=len(layers) - 1)
            append_act(layers, act_kind, act_params=cfg)
        while H > 1 or W > 1:
            H, W = _pool2d(layers, kind=LayerKind.AVGPOOL2D.value, in_ch=ch, H=H, W=W)
            if H <= 0 or W <= 0:
                raise ValueError("Invalid spatial dims after head pooling")
        append_flatten(layers)
        append_dense(
            layers,
            in_features=ch * H * W,
            out_features=int(cfg["num_classes"]),
            use_bias=True,
        )

    elif variant == "stage":
        ch = int(cfg["base_channels"])
        H, W = _conv2d(layers, in_ch=in_ch, out_ch=ch, H=H, W=W)
        append_act(layers, act_kind, act_params=cfg)
        for stage in range(int(cfg["stages"])):
            if stage > 0:
                next_ch = min(64, ch * int(cfg["channel_mult"]))
                ds = cfg.get("downsample", "maxpool")
                if ds == "stride2_conv":
                    H, W = _conv2d(
                        layers,
                        in_ch=ch,
                        out_ch=next_ch,
                        H=H,
                        W=W,
                        kernel=3,
                        stride=2,
                        padding=1,
                    )
                    append_act(layers, act_kind, act_params=cfg)
                    ch = next_ch
                else:
                    pool_type = _POOL_KIND_TO_LAYER.get(ds, LayerKind.MAXPOOL2D.value)
                    H, W = _pool2d(layers, kind=pool_type, in_ch=ch, H=H, W=W)
                    if next_ch != ch:
                        H, W = _conv2d(
                            layers,
                            in_ch=ch,
                            out_ch=next_ch,
                            H=H,
                            W=W,
                            kernel=1,
                            stride=1,
                            padding=0,
                        )
                        append_act(layers, act_kind, act_params=cfg)
                        ch = next_ch
            for _ in range(int(cfg["blocks_per_stage"])):
                if rng.random() < float(cfg.get("double_conv_p", 0.5)):
                    H, W = _conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W)
                    append_act(layers, act_kind, act_params=cfg)
                    H, W = _conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W)
                    append_act(layers, act_kind, act_params=cfg)
                else:
                    H, W = _conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W)
                    append_act(layers, act_kind, act_params=cfg)
        if cfg.get("head_pool_to_1x1", True):
            while H > 1 or W > 1:
                H, W = _pool2d(
                    layers, kind=LayerKind.AVGPOOL2D.value, in_ch=ch, H=H, W=W
                )
                if H <= 0 or W <= 0:
                    raise ValueError("Invalid spatial dims after head pooling")
        append_flatten(layers)
        append_dense(
            layers,
            in_features=ch * H * W,
            out_features=int(cfg["num_classes"]),
            use_bias=True,
        )
    else:
        raise ValueError(f"Unsupported CNN variant '{variant}'")


# ============================================================================
# TF capabilities and allowed layers
# ============================================================================


_DEFAULT_COVERAGE_LAYERS = sorted(
    lk.value
    for lk in LayerKind
    if lk.value
    not in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value)
)


@functools.lru_cache(maxsize=1)
def _get_tf_capabilities() -> Dict[str, FrozenSet[str]]:
    result = {}
    _tf_specs = [
        ("interval", "act.back_end.interval_tf", "IntervalTF", "_LAYER_REGISTRY"),
        ("hybridz", "act.back_end.hybridz_tf", "HybridzTF", "_LAYER_REGISTRY"),
        ("dual", "act.back_end.dual_tf", "DualTF", "_BACKWARD_REGISTRY"),
    ]
    for tf_name, module_path, class_name, registry_attr in _tf_specs:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            registry = getattr(cls, registry_attr, {})
            layers = set(k.upper() for k in registry.keys())
            result[tf_name] = frozenset(layers)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Cannot load {class_name}.{registry_attr}: {e}") from e
    return result


def _get_allowed_layers(tf_targets=None, mode="intersection"):
    if tf_targets is None:
        tf_targets = ["interval", "hybridz", "dual"]
    tf_targets = [t.lower().strip() for t in tf_targets]
    capabilities = _get_tf_capabilities()
    unknown = set(tf_targets) - set(capabilities.keys())
    if unknown:
        raise ValueError(
            f"Unknown TF targets: {unknown}. Available: {list(capabilities.keys())}"
        )
    target_sets = [capabilities[tf] for tf in tf_targets]
    if len(target_sets) == 1:
        result = target_sets[0]
    elif mode == "intersection":
        result = target_sets[0]
        for s in target_sets[1:]:
            result = result & s
    elif mode == "union":
        result = frozenset().union(*target_sets)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected 'intersection' or 'union'.")
    if not result:
        raise ValueError(
            f"Empty layer set for tf_targets={tf_targets}, mode={mode}. Check TF registries or try mode='union'."
        )
    return result


def _derive_seed(base_seed: int, idx: int, instance_id: str) -> int:
    payload = f"{base_seed}|{idx}|{instance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


# ============================================================================
# Variable generation functions and kind-category sets
# ============================================================================


def _generate_layer_variables(kind, i, vc, params, layers):
    # INPUT: create initial variables from shape
    if kind == LayerKind.INPUT.value:
        n = torch.Size(params["shape"]).numel()
        return [], list(range(vc, vc + n)), vc + n

    # INPUT_SPEC / ASSERT: passthrough, no new variables
    if kind in (LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
        pv = layers[i - 1].out_vars
        return list(pv), list(pv), vc

    # Binary ops (x_vars + y_vars already populated by create_network)
    x_vars = params.get("x_vars", [])
    y_vars = params.get("y_vars", [])
    if x_vars and y_vars:
        in_vars = list(x_vars) + list(y_vars)
        n_out = len(x_vars)
        return in_vars, list(range(vc, vc + n_out)), vc + n_out

    # Concat (multiple predecessors)
    preds = params.get("preds_indices", [])
    if len(preds) > 1:
        in_vars = []
        for pidx in preds:
            if pidx < len(layers):
                in_vars.extend(layers[pidx].out_vars)
        n_out = len(in_vars)
        return in_vars, list(range(vc, vc + n_out)), vc + n_out

    # Single predecessor — determine in_vars
    in_vars = list(layers[i - 1].out_vars)

    # Determine output count from params
    if "out_features" in params:
        n_out = int(params["out_features"])
    elif "output_shape" in params:
        n_out = torch.Size(params["output_shape"]).numel()
    else:
        n_out = len(in_vars)

    return in_vars, list(range(vc, vc + n_out)), vc + n_out


# ============================================================================
# ConfigSampler
# ============================================================================


class ConfigSampler:
    _FAMILY_REQUIRED_LAYERS = {
        "mlp": {LayerKind.DENSE.value, LayerKind.RELU.value},
        "cnn2d": {LayerKind.CONV2D.value, LayerKind.DENSE.value, LayerKind.RELU.value},
        "rnn": {LayerKind.LSTM.value, LayerKind.FLATTEN.value, LayerKind.DENSE.value},
    }

    _ALL_ACTIVATIONS = frozenset(lk.value for lk in _ACTIVATIONS)

    def __init__(
        self, config: Dict[str, Any], allowed_layers: Optional[FrozenSet[str]] = None
    ):
        self.config = config
        self.allowed_layers = allowed_layers or frozenset(_DEFAULT_COVERAGE_LAYERS)
        self.available_activations = self._ALL_ACTIVATIONS & self.allowed_layers
        if not self.available_activations:
            self.available_activations = frozenset({LayerKind.RELU.value})
        self.available_activations_list = sorted(self.available_activations)
        self.available_pool_kinds = [
            k for k, v in _POOL_KIND_TO_LAYER.items() if v in self.allowed_layers
        ]
        self.available_downsamples = ["stride2_conv"] + [
            k for k, v in _POOL_KIND_TO_LAYER.items() if v in self.allowed_layers
        ]
        self.can_head_pool = LayerKind.AVGPOOL2D.value in self.allowed_layers

        self.available_families = self._compute_available_families()

    def _compute_available_families(self) -> List[str]:
        available = []
        for family, required in self._FAMILY_REQUIRED_LAYERS.items():
            if family not in self.config.get("families", {}):
                continue
            if required <= self.allowed_layers:
                available.append(family)
        return available

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

    _RULE_KEYS = ("choice", "range", "weighted", "repeat", "probability", "const")

    def _sample_dict(self, rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in spec.items():
            if isinstance(value, dict):
                is_rule = any(k in value for k in self._RULE_KEYS)
                result[key] = (
                    self._sample_value(rng, value)
                    if is_rule
                    else self._sample_dict(rng, value)
                )
            else:
                result[key] = value
        return result

    def sample_family(self, rng: random.Random) -> Tuple[str, Dict[str, Any]]:
        if not self.available_families:
            raise ValueError("No families available for current allowed_layers")
        selection = self.config["family_selection"]
        if "weighted" in selection:
            filtered = {
                k: v
                for k, v in selection["weighted"].items()
                if k in self.available_families
            }
            if not filtered:
                raise ValueError(f"No families match: {self.available_families}")
            names = list(filtered.keys())
            weights = list(filtered.values())
            total = sum(weights)
            family = rng.choices(names, weights=[w / total for w in weights])[0]
        else:
            raise ValueError("family_selection must have 'weighted' strategy")
        params = self._sample_dict(rng, self.config["families"][family])

        for k in ("input_shape", "hidden_sizes", "conv_channels"):
            if k in params:
                params[k] = tuple(int(x) for x in params[k])

        # TF-capability filtering for activation / pool_kind / downsample.
        # These are deterministic overrides applied only when the YAML-sampled
        # value is not supported by the current TF set. Using ``rng.choice``
        # here would consume random state and break seed reproducibility with
        # consumers (incl. upstream paper baselines); we fall back to the
        # first allowed value instead, matching the pre-refactor convention.
        sampled_act = str(params.get("activation", "")).lower()
        if not self.available_activations:
            pass  # nothing to filter against; leave as-is
        elif sampled_act and sampled_act.upper() in self.available_activations:
            params["activation"] = sampled_act
        else:
            params["activation"] = next(iter(self.available_activations_list)).lower()

        if "use_pooling" in params:
            if self.available_pool_kinds:
                sampled_pool = params.get("pool_kind")
                if sampled_pool not in self.available_pool_kinds:
                    params["pool_kind"] = self.available_pool_kinds[0]
            else:
                params["use_pooling"] = False

        if "downsample" in params:
            sampled_ds = params.get("downsample")
            if self.available_downsamples and sampled_ds not in self.available_downsamples:
                params["downsample"] = self.available_downsamples[0]

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
            return {
                "kind": "BOX",
                "value_range": (lo, hi),
                "lb_val": lb_val,
                "ub_val": ub_val,
            }
        if kind == "LINF_BALL":
            center = lo + (hi - lo) * rng.random()
            eps = self._sample_value(rng, sc["eps"])
            eps = min(float(eps), 0.5 * (hi - lo)) if hi > lo else 0.0
            return {
                "kind": "LINF_BALL",
                "value_range": (lo, hi),
                "center_val": center,
                "eps": eps,
            }
        raise ValueError(f"Unsupported input_spec kind '{kind}'")

    def sample_output_spec(
        self, rng: random.Random, *, num_classes: int
    ) -> Dict[str, Any]:
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
            lb = [
                min(lo + (hi - lo) * rng.random(), lo + (hi - lo) * rng.random())
                for _ in range(num_classes)
            ]
            ub = [
                max(lo + (hi - lo) * rng.random(), lo + (hi - lo) * rng.random())
                for _ in range(num_classes)
            ]
            return {"kind": "RANGE", "lb": lb, "ub": ub}
        raise ValueError(f"Unsupported output_spec kind '{kind}'")


# ============================================================================
# NetFactory
# ============================================================================


class NetFactory:
    def __init__(
        self,
        gen_config_path: Optional[str] = None,
        *,
        output_dir: Optional[str] = None,
        base_seed: Optional[int] = None,
        num_instances: Optional[int] = None,
        name_prefix: Optional[str] = None,
        write_manifest: Optional[bool] = None,
        tf_targets: Optional[List[str]] = None,
        registry_mode: str = "intersection",
    ):
        if gen_config_path is None:
            gen_config_path = str(
                Path(__file__).parent / "examples" / "config_gen_act_net.yaml"
            )
        self.config_path = str(gen_config_path)
        self.config = self._load_config(self.config_path)
        common = self.config["common"]

        self.tf_targets = tf_targets
        self.registry_mode = registry_mode
        self.allowed_layers = self._compute_allowed_layers(tf_targets, registry_mode)
        self.sampler = ConfigSampler(self.config, allowed_layers=self.allowed_layers)

        self.base_seed = (
            int(base_seed)
            if base_seed is not None
            else (
                int(common["base_seed"])
                if common.get("base_seed")
                else int(secrets.randbits(32))
            )
        )
        self.num_instances = (
            int(num_instances)
            if num_instances is not None
            else int(common["num_instances"])
        )
        self.name_prefix = (
            str(name_prefix) if name_prefix is not None else str(common["name_prefix"])
        )

        od = output_dir or common["output_dir"]
        self.output_dir = Path(od)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_manifest = (
            bool(write_manifest)
            if write_manifest is not None
            else bool(common.get("write_manifest", True))
        )
        mp = common.get("manifest_path")
        self.manifest_path = (
            Path(mp) if mp else (self.output_dir / "_meta" / "manifest.json")
        )

        self.coverage_mode = common.get("coverage_mode", "basic")
        self.coverage_max_attempts = int(common.get("coverage_max_attempts", 1000))
        self.coverage_report = bool(common.get("coverage_report", True))
        self._init_coverage()
        self.total_generated = 0

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
            return _get_allowed_layers(tf_targets, mode)
        except Exception as e:
            logger.warning("TF capabilities unavailable: %s. Using defaults.", e)
            return frozenset(_DEFAULT_COVERAGE_LAYERS)

    def _init_coverage(self):
        skip = {
            LayerKind.INPUT.value,
            LayerKind.INPUT_SPEC.value,
            LayerKind.ASSERT.value,
        }
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

    def _gen_weight(self, kind: str, params: Dict[str, Any]) -> Optional[torch.Tensor]:
        if kind == LayerKind.DENSE.value:
            return (
                torch.randn(
                    int(params.get("out_features", 1)),
                    int(params.get("in_features", 1)),
                )
                * 0.1
            )
        if kind in (
            LayerKind.CONV1D.value,
            LayerKind.CONV2D.value,
            LayerKind.CONV3D.value,
            LayerKind.CONVTRANSPOSE2D.value,
        ):
            ic = int(params.get("in_channels", 1))
            oc = int(params.get("out_channels", 1))
            ks = params.get("kernel_size", 3)
            ks = int(ks) if isinstance(ks, int) else int(ks[0])
            ndim = {
                LayerKind.CONV1D.value: 1,
                LayerKind.CONV2D.value: 2,
                LayerKind.CONV3D.value: 3,
                LayerKind.CONVTRANSPOSE2D.value: 2,
            }[kind]
            if kind == LayerKind.CONVTRANSPOSE2D.value:
                shape = (ic, oc) + (ks,) * ndim
            else:
                shape = (oc, ic) + (ks,) * ndim
            return torch.randn(*shape) * 0.1
        return None

    def _populate_rnn_weights(
        self, kind: str, params: Dict[str, Any], dtype: torch.dtype
    ) -> None:
        """Generate weight_ih_l0 / weight_hh_l0 / bias_*_l0 (and reverse if
        bidirectional) for an RNN/LSTM/GRU layer spec, then drop the
        ``use_bias`` factory hint so it does not leak into Layer.params (the
        REGISTRY does not list ``use_bias`` as a valid key)."""
        gates = {LayerKind.RNN.value: 1, LayerKind.GRU.value: 3, LayerKind.LSTM.value: 4}[kind]
        input_size = int(params["input_size"])
        hidden_size = int(params["hidden_size"])
        bidirectional = bool(params.get("bidirectional", False))
        use_bias = bool(params.pop("use_bias", True))
        rows = gates * hidden_size

        # Small init scale matches DENSE/CONV (0.1) so generated bounds stay
        # in a regime the interval domain can handle without exploding.
        scale = 0.1
        if "weight_ih_l0" not in params:
            params["weight_ih_l0"] = torch.randn(rows, input_size, dtype=dtype) * scale
        if "weight_hh_l0" not in params:
            params["weight_hh_l0"] = torch.randn(rows, hidden_size, dtype=dtype) * scale
        if use_bias:
            if "bias_ih_l0" not in params:
                params["bias_ih_l0"] = torch.zeros(rows, dtype=dtype)
            if "bias_hh_l0" not in params:
                params["bias_hh_l0"] = torch.zeros(rows, dtype=dtype)
        if bidirectional:
            if "weight_ih_l0_reverse" not in params:
                params["weight_ih_l0_reverse"] = torch.randn(rows, input_size, dtype=dtype) * scale
            if "weight_hh_l0_reverse" not in params:
                params["weight_hh_l0_reverse"] = torch.randn(rows, hidden_size, dtype=dtype) * scale
            if use_bias:
                if "bias_ih_l0_reverse" not in params:
                    params["bias_ih_l0_reverse"] = torch.zeros(rows, dtype=dtype)
                if "bias_hh_l0_reverse" not in params:
                    params["bias_hh_l0_reverse"] = torch.zeros(rows, dtype=dtype)

    def _input_spec_params(self, params, input_shape, dtype):
        if params["kind"] == InKind.BOX:
            return {
                "lb": torch.full(
                    input_shape, float(params.get("lb_val", 0.0)), dtype=dtype
                ),
                "ub": torch.full(
                    input_shape, float(params.get("ub_val", 1.0)), dtype=dtype
                ),
            }
        if params["kind"] == InKind.LINF_BALL:
            center = torch.full(
                input_shape, float(params.get("center_val", 0.5)), dtype=dtype
            )
            eps = float(params.get("eps", 0.0))
            return {"center": center, "lb": center - eps, "ub": center + eps}
        raise ValueError(f"Unsupported INPUT_SPEC kind '{params.get('kind')}'")

    def _assert_params(self, params, dtype):
        kind = params.get("kind")
        if kind == OutKind.LINEAR_LE and isinstance(params.get("c"), list):
            params["c"] = torch.as_tensor(params["c"], dtype=dtype)
        elif kind == OutKind.RANGE:
            for k in ("lb", "ub"):
                if isinstance(params.get(k), list):
                    params[k] = torch.as_tensor(params[k], dtype=dtype)
        return params

    def _sample_instance(self, idx: int) -> Dict[str, Any]:
        temp_id = f"{self.name_prefix}{self.base_seed}_idx{idx:05d}"
        seed = _derive_seed(self.base_seed, idx, temp_id)
        rng = random.Random(seed)
        family, model_cfg = self.sampler.sample_family(rng)
        nc = int(model_cfg["num_classes"])
        instance_id = self._semantic_name(family, model_cfg, seed)
        return {
            "instance_id": instance_id,
            "seed": seed,
            "family": family,
            "model_cfg": model_cfg,
            "input_spec": self.sampler.sample_input_spec(rng),
            "output_spec": self.sampler.sample_output_spec(rng, num_classes=nc),
        }

    def _semantic_name(self, family: str, cfg: Dict[str, Any], seed: int) -> str:
        variant = cfg.get("variant", "plain")
        family_tag = (
            f"{family}_{variant}"
            if family != "cnn2d" or variant != "stage"
            else "resnet"
        )
        dims = (
            cfg["input_shape"][1:] if cfg["input_shape"][0] == 1 else cfg["input_shape"]
        )
        input_str = "x".join(str(d) for d in dims)
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
        elif family == "rnn":
            cell = str(cfg.get("cell", "LSTM")).lower()
            bidi = "bi" if cfg.get("bidirectional") else ""
            struct = f"{bidi}{cell}_h{cfg.get('hidden_size', 16)}"
            family_tag = f"rnn_{cell}"
        else:
            struct = "default"
        return f"{family_tag}_{input_str}_{struct}_{seed}"

    def _build_spec(self, instance: Dict[str, Any], dtype: str) -> Dict[str, Any]:
        cfg = instance["model_cfg"]
        input_shape = list(cfg["input_shape"])
        nc = int(cfg["num_classes"])
        layers: List[Dict[str, Any]] = []

        layers.append(
            {
                "kind": LayerKind.INPUT.value,
                "params": {
                    "shape": input_shape,
                    "dtype": dtype,
                    "num_classes": nc,
                    "value_range": list(instance["input_spec"]["value_range"]),
                },
            }
        )

        ik = str(instance["input_spec"]["kind"])
        sm: Dict[str, Any] = {"kind": ik}
        if ik == "BOX":
            sm["lb_val"] = float(instance["input_spec"]["lb_val"])
            sm["ub_val"] = float(instance["input_spec"]["ub_val"])
        elif ik == "LINF_BALL":
            sm["center_val"] = float(instance["input_spec"]["center_val"])
            sm["eps"] = float(instance["input_spec"]["eps"])
        layers.append({"kind": LayerKind.INPUT_SPEC.value, "params": sm})

        if instance["family"] == "mlp":
            build_mlp_layers(layers, cfg=cfg)
        elif instance["family"] == "cnn2d":
            build_cnn_layers(layers, cfg=cfg, rng=random.Random(int(instance["seed"])))
        elif instance["family"] == "rnn":
            build_rnn_layers(layers, cfg=cfg)
        else:
            raise ValueError(f"Unsupported family: {instance['family']}")

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
        layers.append({"kind": LayerKind.ASSERT.value, "params": {**om, **op}})

        return {"layers": layers}

    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        dtype = get_default_dtype()
        dtype_str = str(dtype)
        layers: List[Layer] = []
        vc = 0

        for i, ls in enumerate(spec["layers"]):
            params = dict(ls.get("params", {}))
            kind = ls["kind"]

            inputs = ls.get("inputs") or {}
            if "x" in inputs and "y" in inputs:
                params["x_vars"] = list(layers[inputs["x"]].out_vars)
                params["y_vars"] = list(layers[inputs["y"]].out_vars)

            if kind == LayerKind.CONCAT.value:
                pred_indices = ls.get("preds", [])
                if not pred_indices and len(layers) >= 2:
                    pred_indices = [len(layers) - 2, len(layers) - 1]
                params["preds_indices"] = pred_indices

            if "preds" in ls and "preds_indices" not in params:
                params["preds_indices"] = ls["preds"]

            if kind in (LayerKind.MAX.value, LayerKind.MIN.value):
                pred_indices = ls.get("preds", [])
                if pred_indices:
                    params["y_vars_list"] = [
                        list(layers[p].out_vars)
                        for p in pred_indices
                        if p < len(layers)
                    ]

            in_vars, out_vars, vc = _generate_layer_variables(
                kind, i, vc, params, layers
            )

            if kind == LayerKind.INPUT.value:
                params["dtype"] = dtype_str
            elif kind == LayerKind.INPUT_SPEC.value:
                params.update(
                    self._input_spec_params(params, layers[0].params["shape"], dtype)
                )
            elif kind == LayerKind.ASSERT.value:
                params = self._assert_params(params, dtype)
            elif kind == LayerKind.DENSE.value and "weight" not in params:
                inf = int(params.get("in_features", 1))
                outf = int(params.get("out_features", 1))
                params["weight"] = torch.randn(outf, inf, dtype=dtype) * 0.1
                params["in_features"] = inf
                params["out_features"] = outf
                if params.pop("use_bias", True):
                    params["bias"] = torch.zeros(outf, dtype=dtype)
            elif (
                kind
                in (
                    LayerKind.CONV1D.value,
                    LayerKind.CONV2D.value,
                    LayerKind.CONV3D.value,
                    LayerKind.CONVTRANSPOSE2D.value,
                )
                and "weight" not in params
            ):
                w = self._gen_weight(kind, params)
                if w is not None:
                    params["weight"] = w
            elif kind == LayerKind.BIAS.value and "c" not in params:
                params["c"] = torch.zeros(len(in_vars), dtype=dtype)
            elif kind == LayerKind.SCALE.value and "a" not in params:
                params["a"] = torch.ones(len(in_vars), dtype=dtype)
            elif kind in (
                LayerKind.RNN.value, LayerKind.LSTM.value, LayerKind.GRU.value,
            ):
                self._populate_rnn_weights(kind, params, dtype)

            params.pop("preds_indices", None)

            if kind == LayerKind.LRELU.value and "negative_slope" in params:
                params["alpha"] = params["negative_slope"]

            layer = Layer(
                id=i, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars
            )
            layers.append(layer)

        preds: Dict[int, List[int]] = {}
        for i, ls in enumerate(spec["layers"]):
            sp = ls.get("preds")
            preds[i] = list(sp) if sp is not None else ([i - 1] if i > 0 else [])

        succs: Dict[int, List[int]] = {i: [] for i in range(len(layers))}
        for i, pl in preds.items():
            for p in pl:
                succs[p].append(i)

        net = Net(layers=layers, preds=preds, succs=succs)
        setattr(net, "meta", {"name": name})
        validate_factory_schema_alignment(net)
        return net

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
        print(
            f"Coverage: {covered}/{total} ({rate:.1f}%)  Networks: {self.total_generated}"
        )
        uncov = self._uncovered()
        if uncov:
            print(f"\nUncovered ({len(uncov)}):")
            for l in sorted(uncov):
                print(f"  - {l}")
        else:
            print("\nAll target layers covered!")
        print(f"{'=' * 60}\n")

    def _generate_one(self, idx: int, dtype: str, names: List[str]) -> None:
        inst = self._sample_instance(idx)
        spec = self._build_spec(inst, dtype=dtype)
        net = self.create_network(inst["instance_id"], spec)
        self.save_network(net, inst["instance_id"])
        names.append(inst["instance_id"])
        self.total_generated += 1
        self._record(net)

    def generate(self) -> List[str]:
        for old in self.output_dir.glob("*.json"):
            old.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()

        dtype = str(self.config["common"]["dtype"])
        names: List[str] = []

        if self.coverage_mode == "full":
            print(
                f"Generating networks in FULL coverage mode (max {self.coverage_max_attempts} attempts)..."
            )
            for idx in range(self.coverage_max_attempts):
                self._generate_one(idx, dtype, names)
                if (idx + 1) % 50 == 0:
                    print(
                        f"  {idx + 1} generated, coverage: {self._coverage_rate():.1f}%, uncovered: {len(self._uncovered())}"
                    )
                if not self._uncovered():
                    print(f"\n  All layers covered after {idx + 1} networks!")
                    break

        else:
            print(f"Generating {self.num_instances} networks in BASIC mode...")
            for idx in range(self.num_instances):
                self._generate_one(idx, dtype, names)

        if self.write_manifest:
            self._write_manifest(names)

        print(f"\nAll networks saved to {self.output_dir}")
        self._print_coverage_report()
        return names


__all__ = [
    "NetFactory",
    "build_mlp_layers",
    "build_cnn_layers",
]