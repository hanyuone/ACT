#===- act/back_end/dual_tf/dual_tf.py - Dual Transfer Function Class ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualTF: forward-only TransferFunction for dual mode. Holds alpha parameters.
# All backward kernels live as module-level dispatch functions; the actual
# certified-bound solver is act.back_end.solver.solver_dual.DualSolver.
#===---------------------------------------------------------------------===#


import torch
from typing import Dict, List, Optional, Tuple
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.layer_schema import LayerKind
from act.back_end.transfer_functions import TransferFunction
from .tf_mlp import (
    backward_dense, backward_relu, backward_bias, backward_scale,
    backward_bn, backward_identity,
    forward_dense, forward_relu, forward_bias, forward_scale,
    forward_bn, forward_lrelu, forward_identity, forward_reshape,
)
from .tf_cnn import (
    backward_conv2d, backward_maxpool2d, backward_avgpool2d,
    forward_conv2d, forward_maxpool2d, forward_avgpool2d,
)
from .tf_smooth import (
    backward_sigmoid, backward_tanh,
    forward_sigmoid, forward_tanh,
)
from .tf_rnn import forward_lstm, backward_lstm, forward_gru, backward_gru
from .tf_transformer import (
    forward_attention, backward_attention,
    forward_layernorm, backward_layernorm,
    forward_gelu, backward_gelu,
)
from .tf_forward import (
    compute_forward_bounds, LinearBound, Frame,
    _sum_linear_bounds, _sum_interval_bounds, _concretize,
    _reset_forward_box, _align, _int_param,
)


# ---- ADD ----
def forward_add(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """ADD multi-pred forward handler.

    Source: tf_forward.py lines 287-322 (ADD branch of compute_forward_bounds).
    Semantics preserved verbatim — when all predecessor frames share the same
    object identity and A_lb shapes match, sum the dual-track linear bounds and
    concretize over the common frame; otherwise fall back to summing interval
    boxes and reset the dual-track state. Bias (if present) is added on both
    paths via _align. Returns (stored, out, lin, frame) where stored == out.
    """
    assert len(parent_boxes) >= 2, "forward_add: requires >=2 predecessors"
    can_dual = all(
        parent_frames[i] is parent_frames[0] for i in range(1, len(parent_frames))
    ) and all(
        parent_lins[i].A_lb.shape == parent_lins[0].A_lb.shape
        for i in range(1, len(parent_lins))
    )
    if can_dual:
        lin = _sum_linear_bounds(parent_lins)
        bias_param = L.params.get("bias")
        if bias_param is not None:
            bias_vec = _align(bias_param.flatten(), lin.b_lb.shape[1])
            lin = LinearBound(
                A_lb=lin.A_lb,
                b_lb=lin.b_lb + bias_vec,
                A_ub=lin.A_ub,
                b_ub=lin.b_ub + bias_vec,
            )
        frame = parent_frames[0]
        lb, ub = _concretize(lin, *frame)
    else:
        summed = _sum_interval_bounds(parent_boxes)
        lb, ub = summed.lb, summed.ub
        bias_param = L.params.get("bias")
        if bias_param is not None:
            bias_vec = _align(bias_param.flatten(), lb.shape[1])
            lb = lb + bias_vec
            ub = ub + bias_vec
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_add(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """ADD backward: identity skip — same ν routed to every predecessor.

    Bias contrib uses negative sign to match dual_bias_backward / dual_bn_backward
    / dual_dense_backward conventions (y = x + bias ⇒ contrib = -(ν · bias)).
    """
    B = nu.shape[0]
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    if "bias" in L.params and L.params["bias"] is not None:
        b = L.params["bias"].flatten()
        v = nu.flatten(start_dim=1)
        n = min(v.shape[-1], b.numel())
        contrib = -(v[..., :n] * b[:n]).sum(dim=-1)
    return [nu for _ in preds], contrib


# ---- CONCAT ----
def forward_concat(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """CONCAT multi-pred forward handler.

    Source: tf_forward.py lines 324-346 (CONCAT branch of compute_forward_bounds).
    Semantics preserved verbatim — when all predecessor frames share the same
    object identity and A_lb batch/input axes match, concatenate dual-track
    linear bounds along dim=1 and concretize; otherwise fall back to torch.cat
    on interval boxes along concat_dim (default 1) and reset dual-track state.
    Returns (stored, out, lin, frame) where stored == out.
    """
    assert len(parent_boxes) >= 2, "forward_concat: requires >=2 predecessors"
    concat_dim = _int_param(L.params.get("concat_dim", 1), 1)
    can_dual = all(
        parent_frames[i] is parent_frames[0] for i in range(1, len(parent_frames))
    ) and all(
        parent_lins[i].A_lb.shape[0] == parent_lins[0].A_lb.shape[0]
        and parent_lins[i].A_lb.shape[2] == parent_lins[0].A_lb.shape[2]
        for i in range(1, len(parent_lins))
    )
    if can_dual:
        lin = LinearBound(
            A_lb=torch.cat([lin.A_lb for lin in parent_lins], dim=1),
            b_lb=torch.cat([lin.b_lb for lin in parent_lins], dim=1),
            A_ub=torch.cat([lin.A_ub for lin in parent_lins], dim=1),
            b_ub=torch.cat([lin.b_ub for lin in parent_lins], dim=1),
        )
        frame = parent_frames[0]
        lb, ub = _concretize(lin, *frame)
    else:
        lb = torch.cat([box.lb for box in parent_boxes], dim=concat_dim)
        ub = torch.cat([box.ub for box in parent_boxes], dim=concat_dim)
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_concat(L, nu, bounds_dict, preds):
    """CONCAT backward. (Pending)
    Will require: concat_dim parameter to split nu into per-predecessor slices.
    """
    raise NotImplementedError("backward for CONCAT not implemented in dual_tf")


class DualTF(TransferFunction):
    """Forward-only TF for dual mode. Backward kernels are module-level dispatch
    functions; the actual solver lives at act.back_end.solver.DualSolver.

    `_BACKWARD_REGISTRY` maps layer-kind strings to callable dispatch functions
    with DAG-aware per-predecessor ν routing:

        (L: Layer, nu: Tensor[B, *shape], bounds_dict: Dict[int, Bounds],
         preds: List[int]) -> (pred_nus: List[Tensor], contrib: Tensor[B])

    Each ``pred_nus[i]`` is the ν routed to predecessor ``preds[i]``. Unary
    layers return ``[nu_out]``; ADD returns ``[nu] * len(preds)`` (identity
    skip, same ν to every predecessor). net_factory.py reads only ``.keys()``
    so callable values are fine.
    """

    _FORWARD_REGISTRY = {
        LayerKind.INPUT.value:      forward_identity,
        LayerKind.INPUT_SPEC.value: forward_identity,
        LayerKind.ASSERT.value:     forward_identity,
        LayerKind.DENSE.value:      forward_dense,
        LayerKind.BIAS.value:       forward_bias,
        LayerKind.SCALE.value:      forward_scale,
        LayerKind.BN.value:         forward_bn,
        LayerKind.RELU.value:       forward_relu,
        LayerKind.LRELU.value:      forward_lrelu,
        "LEAKY_RELU":               forward_lrelu,   # alias (not a LayerKind member)
        LayerKind.SIGMOID.value:    forward_sigmoid,
        LayerKind.TANH.value:       forward_tanh,
        LayerKind.CONV2D.value:     forward_conv2d,
        LayerKind.MAXPOOL2D.value:  forward_maxpool2d,
        LayerKind.AVGPOOL2D.value:  forward_avgpool2d,
        LayerKind.FLATTEN.value:    forward_reshape,
        LayerKind.RESHAPE.value:    forward_reshape,
        LayerKind.TRANSPOSE.value:  forward_identity,
        LayerKind.SQUEEZE.value:    forward_identity,
        LayerKind.UNSQUEEZE.value:  forward_identity,
        LayerKind.ADD.value:        forward_add,
        LayerKind.CONCAT.value:     forward_concat,
        LayerKind.LSTM.value:       forward_lstm,
        LayerKind.GRU.value:        forward_gru,
        LayerKind.ATT_SCORES.value: forward_attention,
        LayerKind.ATT_MIX.value:    forward_attention,
        LayerKind.MHA_SPLIT.value:  forward_attention,
        LayerKind.MHA_JOIN.value:   forward_attention,
        LayerKind.MASK_ADD.value:   forward_attention,
        LayerKind.LAYERNORM.value:  forward_layernorm,
        LayerKind.GELU.value:       forward_gelu,
    }

    _BACKWARD_REGISTRY = {
        LayerKind.INPUT.value:      backward_identity,
        LayerKind.INPUT_SPEC.value: backward_identity,
        LayerKind.ASSERT.value:     backward_identity,
        LayerKind.DENSE.value:      backward_dense,
        LayerKind.BIAS.value:       backward_bias,
        LayerKind.SCALE.value:      backward_scale,
        LayerKind.BN.value:         backward_bn,
        LayerKind.RELU.value:       backward_relu,
        LayerKind.LRELU.value:      backward_relu,
        "LEAKY_RELU":               backward_relu,   # alias (not a LayerKind member)
        LayerKind.SIGMOID.value:    backward_sigmoid,
        LayerKind.TANH.value:       backward_tanh,
        LayerKind.CONV2D.value:     backward_conv2d,
        LayerKind.MAXPOOL2D.value:  backward_maxpool2d,
        LayerKind.AVGPOOL2D.value:  backward_avgpool2d,
        LayerKind.FLATTEN.value:    backward_identity,
        LayerKind.RESHAPE.value:    backward_identity,
        LayerKind.TRANSPOSE.value:  backward_identity,
        LayerKind.SQUEEZE.value:    backward_identity,
        LayerKind.UNSQUEEZE.value:  backward_identity,
        LayerKind.ADD.value:        backward_add,
        LayerKind.CONCAT.value:     backward_concat,
        LayerKind.LSTM.value:       backward_lstm,
        LayerKind.GRU.value:        backward_gru,
        LayerKind.ATT_SCORES.value: backward_attention,
        LayerKind.ATT_MIX.value:    backward_attention,
        LayerKind.MHA_SPLIT.value:  backward_attention,
        LayerKind.MHA_JOIN.value:   backward_attention,
        LayerKind.MASK_ADD.value:   backward_attention,
        LayerKind.LAYERNORM.value:  backward_layernorm,
        LayerKind.GELU.value:       backward_gelu,
    }

    _UNIMPLEMENTED_KINDS = frozenset({
        LayerKind.LSTM.value,
        LayerKind.GRU.value,
        LayerKind.GELU.value,
        LayerKind.LAYERNORM.value,
        LayerKind.ATT_SCORES.value,
        LayerKind.ATT_MIX.value,
        LayerKind.MHA_SPLIT.value,
        LayerKind.MHA_JOIN.value,
        LayerKind.MASK_ADD.value,
    })

    def __init__(self):
        self._forward_bounds_cache: Dict[int, Bounds] = {}
        self._cache_net_id: Optional[int] = None

    @property
    def name(self) -> str: return "DualTF"

    def supports_layer(self, layer_kind: str) -> bool:
        # Registry entries for LSTM/GRU/GELU/LAYERNORM/attention-family are
        # placeholder stubs (raise NotImplementedError at runtime). Exclude
        # them here so validate_verifier and BaB skip these nets cleanly
        # rather than surfacing as runtime ERROR.
        k = layer_kind.upper()
        return k in self._BACKWARD_REGISTRY and k not in self._UNIMPLEMENTED_KINDS

    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Return unbatched Bounds Fact for analyze()/BaB integration."""
        net_id = id(net)
        if self._cache_net_id != net_id or not self._forward_bounds_cache:
            input_lb, input_ub = None, None
            for layer in net.layers:
                if layer.kind.upper() in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
                    if layer.id in before:
                        input_lb = before[layer.id].bounds.lb
                        input_ub = before[layer.id].bounds.ub
                        break
                    elif "lb" in layer.params and "ub" in layer.params:
                        input_lb = layer.params["lb"]
                        input_ub = layer.params["ub"]
                        break
            if input_lb is None or input_ub is None:
                input_lb, input_ub = input_bounds.lb, input_bounds.ub
            self._forward_bounds_cache = compute_forward_bounds(
                net, input_lb, input_ub, post_activation=True)
            self._cache_net_id = net_id

        if L.id in self._forward_bounds_cache:
            return Fact(bounds=self._forward_bounds_cache[L.id], cons=ConSet())
        return Fact(bounds=input_bounds, cons=ConSet())

    def clear_cache(self):
        self._forward_bounds_cache.clear()
        self._cache_net_id = None


# Explicit stub registry: any handler whose semantics are "raise NotImplementedError"
# goes here. Membership is the ground truth for stub detection; net_factory filters
# by identity against these sets.
# To implement a stub: fill its body AND remove it from this set in the same commit.
_FORWARD_STUBS = frozenset({
    forward_lstm, forward_gru, forward_attention,
    forward_layernorm, forward_gelu,
})
_BACKWARD_STUBS = frozenset({
    backward_maxpool2d, backward_avgpool2d, backward_concat,
    backward_lstm, backward_gru, backward_attention,
    backward_layernorm, backward_gelu,
})

# --- registry invariant (fires once at module import) ---
assert set(DualTF._FORWARD_REGISTRY.keys()) == set(DualTF._BACKWARD_REGISTRY.keys()), (
    f"DualTF registry keyset mismatch: "
    f"forward-only={set(DualTF._FORWARD_REGISTRY) - set(DualTF._BACKWARD_REGISTRY)}, "
    f"backward-only={set(DualTF._BACKWARD_REGISTRY) - set(DualTF._FORWARD_REGISTRY)}"
)
