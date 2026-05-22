#===- act/back_end/dual_tf/dual_tf.py - Dual Backward Registry Holder ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualTF: backward-kernel registry holder. Not a TransferFunction — dual is a
# --solver choice, not a --tf-mode. Instantiated internally by
# act.back_end.solver.solver_dual.DualSolver.
#===---------------------------------------------------------------------===#


import torch
from typing import Dict, List, Optional, Tuple
from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
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
                 preds: List[int], M: int = 1
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
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


def backward_concat(L, nu, bounds_dict, preds, M: int = 1):
    """CONCAT backward. (Pending)
    Will require: concat_dim parameter to split nu into per-predecessor slices.
    """
    raise NotImplementedError("backward for CONCAT not implemented in dual_tf")


class DualTF:
    """Backward-kernel registry holder for the dual solver.

    Holder of three registries (forward, backward, unimplemented). Dual
    semantics live in DualSolver's backward pass, not in propagated LP
    constraints, so this class is intentionally NOT a TransferFunction.

      * ``_FORWARD_REGISTRY`` — per-kind forward dispatch consumed by
        ``compute_forward_bounds`` (still a real forward computation, but
        invoked internally by ``DualSolver.evaluate_spec`` rather than via
        the analyze()/TF pipeline).
      * ``_BACKWARD_REGISTRY`` — per-kind backward dispatch consumed by
        ``DualSolver.compute_certified_bound``. Each entry has signature
        ``(L, nu, bounds_dict, preds) -> (pred_nus, contrib)``.
      * ``_UNIMPLEMENTED_KINDS`` — kinds whose backward is a stub
        (raises ``NotImplementedError``); ``supports_layer`` filters them
        so dual-incompatible nets get cleanly SKIPPED.

    DualSolver instantiates this internally; external code uses ``--solver
    dual`` rather than touching DualTF directly.
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
        # Backward kernels for these are stubs that raise NotImplementedError
        # at runtime. Listing them here makes supports_layer return False so
        # upstream callers (validate_verifier) cleanly SKIP affected nets
        # instead of surfacing runtime ERROR.
        LayerKind.CONCAT.value,
    })

    def supports_layer(self, layer_kind: str) -> bool:
        k = layer_kind.upper()
        return k in self._BACKWARD_REGISTRY and k not in self._UNIMPLEMENTED_KINDS


# Explicit stub registry: any handler whose semantics are "raise NotImplementedError"
# goes here. Membership is the ground truth for stub detection; net_factory filters
# by identity against these sets.
# To implement a stub: fill its body AND remove it from this set in the same commit.
_FORWARD_STUBS = frozenset({
    forward_lstm, forward_gru, forward_attention,
    forward_layernorm, forward_gelu,
})
_BACKWARD_STUBS = frozenset({
    backward_concat,
    backward_lstm, backward_gru, backward_attention,
    backward_layernorm, backward_gelu,
})

# --- registry invariants (fire once at module import) ---
assert set(DualTF._FORWARD_REGISTRY.keys()) == set(DualTF._BACKWARD_REGISTRY.keys()), (
    f"DualTF registry keyset mismatch: "
    f"forward-only={set(DualTF._FORWARD_REGISTRY) - set(DualTF._BACKWARD_REGISTRY)}, "
    f"backward-only={set(DualTF._BACKWARD_REGISTRY) - set(DualTF._FORWARD_REGISTRY)}"
)

# _UNIMPLEMENTED_KINDS must exactly equal the set of layer kinds whose backward
# handler is a stub (raises NotImplementedError). Drift between these two sets
# is a real risk: implementing a stub without updating _UNIMPLEMENTED_KINDS
# would silently keep skipping a now-working kind; the reverse causes runtime
# NotImplementedError on a kind that supports_layer claims to support.
_stub_kinds_from_registry = frozenset(
    k for k, fn in DualTF._BACKWARD_REGISTRY.items() if fn in _BACKWARD_STUBS
)
assert DualTF._UNIMPLEMENTED_KINDS == _stub_kinds_from_registry, (
    f"DualTF _UNIMPLEMENTED_KINDS drift: "
    f"declared={sorted(DualTF._UNIMPLEMENTED_KINDS)}, "
    f"stub-derived={sorted(_stub_kinds_from_registry)}"
)
