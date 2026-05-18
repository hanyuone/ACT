#===- act/back_end/dual_tf/tf_mlp.py - MLP Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware MLP backward kernels for dual (Wong-Kolter) bound computation.
#
# Kernel convention (STRICT, batch-first):
#   nu      : Tensor[B, *layer_shape]   # dual variable, batch-first
#   v_out   : Tensor[B, *next_shape]
#   contrib : Tensor[B]                 # per-instance scalar
#
# ReLU uses FIXED upper-bound slope for crossing neurons.
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import torch
from typing import Tuple, Optional, Dict, Any, List
from act.back_end.core import Bounds

from .tf_forward import (
    LinearBound, Frame,
    _fwd_dense, _fwd_relu, _fwd_bias, _fwd_scale, _fwd_bn, _fwd_lrelu,
    _concretize, _box_dense, _box_bias, _box_scale, _box_bn, _box_relu,
    _box_lrelu, _intersect_boxes, _reset_forward_box,
)


# ==========================================================================
# Forward dispatch handlers (uniform signature per plan §4.2):
#   (L, parent_boxes, parent_lins, parent_frames, preds,
#    post_activation, device, dtype) -> (stored, out, lin, frame)
#
# `parent_*` are parallel lists indexed by `preds`; unary handlers read [0].
# Function bodies are ported verbatim from the monolithic if/elif chain in
# tf_forward.compute_forward_bounds (pre-refactor; see source line ranges in
# each docstring). Driver still uses if/elif in Wave 2 — these are declared
# but not yet registered (registration lands in Wave 4 / Step D).
# ==========================================================================
# Dispatch functions : (L, nu, bounds_dict, preds) -> (pred_nus, contrib)
# Each pred_nus[i] is the ν routed to predecessor preds[i]. Unary layers
# (DENSE, RELU, BIAS, SCALE, BN) return [nu_out]. backward_identity handles
# both 0-pred (pure INPUT) and 1-pred (FLATTEN/RESHAPE/…) cases.
# ==========================================================================


# ---- HELPERS ----
def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align 1-D parameter tensor to size n by truncating or tiling."""
    if a.numel() == n: return a.flatten()
    elif a.numel() > n: return a.flatten()[:n]
    else: return a.flatten().repeat((n + a.numel() - 1) // a.numel())[:n]


def _batch_flatten_bounds(bounds: Bounds, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return lb, ub as [B, n]. If bounds are unbatched, broadcast to B."""
    if bounds.lb.dim() >= 2:
        return bounds.lb.flatten(start_dim=1), bounds.ub.flatten(start_dim=1)
    return (bounds.lb.flatten().unsqueeze(0).expand(B, -1),
            bounds.ub.flatten().unsqueeze(0).expand(B, -1))


# ---- IDENTITY ----
def forward_identity(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Pass-through handler for INPUT / INPUT_SPEC / ASSERT / TRANSPOSE / SQUEEZE / UNSQUEEZE.

    Source: tf_forward.py lines 357-358 (INPUT family) and 460-461
    (ASSERT / TRANSPOSE / SQUEEZE / UNSQUEEZE family). Both branches are
    `pass`, so stored/out/lin/frame are whatever the predecessor produced.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    return parent_box, parent_box, parent_lin, parent_frame


def backward_identity(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_identity_backward(nu)
    # 0 preds (pure INPUT) -> []; 1 pred (FLATTEN/RESHAPE/…) -> [nu_out].
    return [nu_out] * len(preds), contrib


def dual_identity_backward(nu: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten/Reshape/Transpose backward: v_out = nu, contrib = zeros[B]."""
    B = nu.shape[0]
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return nu, contrib


# ---- RESHAPE ----
def forward_reshape(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for FLATTEN / RESHAPE.

    Source: tf_forward.py lines 420-422. Reshapes predecessor box lb/ub to
    ``[B, -1]`` and keeps lin/frame unchanged; downstream dense layers
    rematch the output-feature axis via ``_match_lin_input_dim``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    B = parent_box.lb.shape[0]
    out = Bounds(parent_box.lb.reshape(B, -1), parent_box.ub.reshape(B, -1))
    stored = out
    return stored, out, parent_lin, parent_frame


# ---- DENSE ----
def forward_dense(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for DENSE (Wong-Kolter dual-track + interval intersection).

    Source: tf_forward.py lines 371-378. Composes lin via ``_fwd_dense``,
    concretizes against the predecessor frame, intersects with the interval
    box update, and returns ``stored == out`` (no pre/post distinction for
    an affine layer). Frame passes through unchanged.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_dense(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_dense(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_dense(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                   preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_dense_backward(nu, L.params["weight"], L.params.get("bias"))
    assert len(preds) == 1, f"DENSE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_dense_backward(nu: torch.Tensor, W: torch.Tensor,
                        b: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched dense backward: v_out = nu @ W.

    nu : [B, out], W : [out, in] -> v_out: [B, in], contrib: [B].
    """
    assert W.dim() == 2, f"W must be 2D, got {W.shape}"
    assert nu.dim() >= 2, f"nu must be batched (>=2D), got {nu.shape}"
    B = nu.shape[0]
    nu_flat = nu.flatten(start_dim=1)
    assert nu_flat.shape[-1] == W.shape[0], \
        f"nu last dim {nu_flat.shape[-1]} != W.shape[0] {W.shape[0]}"
    v_out = nu_flat @ W
    if b is not None:
        contrib = -(nu_flat @ b.flatten())
    else:
        contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return v_out, contrib


# ---- RELU / LRELU ----
def get_relu_masks(l: torch.Tensor, u: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Element-wise masks (on, off, amb); shape-preserving."""
    on, off = l >= 0, u <= 0
    return on, off, ~(on | off)


def forward_relu(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for RELU (linear relaxation + interval intersection).

    Source: tf_forward.py lines 360-369. When ``post_activation`` is True,
    ``stored == out`` (post-ReLU box) and lin/frame are reset to identity
    over the new concrete box via ``_reset_forward_box``. Otherwise
    ``stored`` is the pre-activation box and lin/frame pass through.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    lin = _fwd_relu(parent_lin, pre_lb, pre_ub)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_relu(pre_lb, pre_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    frame = parent_frame
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def forward_lrelu(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for LRELU / LEAKY_RELU (triangle linear relaxation).

    Source: tf_forward.py lines 436-446. Reads ``alpha`` from
    ``L.params.get("alpha", 0.01)``. ``post_activation`` handling mirrors
    :func:`forward_relu`: when True ``stored == out`` and lin/frame are
    reset to identity on the new box; otherwise ``stored`` is the
    pre-activation box and lin/frame pass through.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    alpha = L.params.get("alpha", 0.01)
    lin = _fwd_lrelu(parent_lin, pre_lb, pre_ub, alpha)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_lrelu(pre_lb, pre_ub, alpha)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    frame = parent_frame
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_relu(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_relu: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_relu_backward(nu, bounds)
    assert len(preds) == 1, f"RELU expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_relu_backward(nu: torch.Tensor, bounds: Bounds
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched ReLU backward with fixed upper slope.

    nu : [B, *shape] -> v_out: [B, *shape_or_flat], contrib: [B].
    """
    B = nu.shape[0]
    l, u = _batch_flatten_bounds(bounds, B)
    v = nu.flatten(start_dim=1)
    n = min(v.shape[-1], l.shape[-1])
    if v.shape[-1] != l.shape[-1]:
        l, u, v = l[..., :n], u[..., :n], v[..., :n]
    assert (l <= u).all(), "Invalid bounds: l > u"

    on, off, amb = get_relu_masks(l, u)
    d = torch.zeros_like(l)
    d = torch.where(on, torch.ones_like(d), d)
    if amb.any():
        denom = (u - l).clamp(min=1e-12)
        d = torch.where(amb, u / denom, d)

    v_out = d * v                                                # [B, n]
    if amb.any():
        crossing = torch.where(amb, v_out.clamp(min=0) * l, torch.zeros_like(l))
        contrib = crossing.sum(dim=-1)                           # [B]
    else:
        contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return v_out, contrib


# ---- BIAS ----
def forward_bias(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for BIAS (``y = x + c``).

    Source: tf_forward.py lines 393-400. Composes via ``_fwd_bias``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_bias(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_bias(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_bias(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_bias_backward(nu, L.params["c"])
    assert len(preds) == 1, f"BIAS expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_bias_backward(nu: torch.Tensor, c: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = x + c ; v_out = nu, contrib = -(v * c_flat).sum(dim=-1)."""
    v = nu.flatten(start_dim=1)
    c_flat = _align(c, v.shape[-1])
    contrib = -(v * c_flat).sum(dim=-1)                          # [B]
    return nu, contrib


# ---- SCALE ----
def forward_scale(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for SCALE (``y = a * x``, element-wise).

    Source: tf_forward.py lines 402-409. Composes via ``_fwd_scale``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_scale(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_scale(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_scale(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                   preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_scale_backward(nu, L.params["a"])
    assert len(preds) == 1, f"SCALE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_scale_backward(nu: torch.Tensor, a: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = a * x ; v_out = a * nu, contrib = 0."""
    B = nu.shape[0]
    flat = nu.flatten(start_dim=1)
    a_aligned = _align(a, flat.shape[-1])
    out = (a_aligned * flat).view(nu.shape)
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return out, contrib


# ---- BN ----
def forward_bn(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for BN (``y = A * x + c``, element-wise).

    Source: tf_forward.py lines 411-418. Composes via ``_fwd_bn``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_bn(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_bn(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_bn(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_bn_backward(nu, L.params["A"], L.params["c"])
    assert len(preds) == 1, f"BN expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_bn_backward(nu: torch.Tensor, A: torch.Tensor, c: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = A*x + c ; v_out = A*nu, contrib = -(v * c_flat).sum(dim=-1)."""
    B = nu.shape[0]
    v = nu.flatten(start_dim=1)
    A_aligned = _align(A, v.shape[-1])
    c_aligned = _align(c, v.shape[-1])
    out = (A_aligned * v).view(nu.shape)
    contrib = -(v * c_aligned).sum(dim=-1)                       # [B]
    return out, contrib
