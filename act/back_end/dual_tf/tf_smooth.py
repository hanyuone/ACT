#===- act/back_end/dual_tf/tf_smooth.py - Smooth Activation Dual TF -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware smooth (S-shaped) activation backward.
# nu: [B, *shape] -> v_out: [B, *shape], contrib: [B].
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import torch
from typing import Tuple, Callable, Dict, Any, List
from act.back_end.core import Bounds
from .tf_forward import LinearBound, Frame, _reset_forward_box


# ---- Shared primitives ----

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def dsigmoid(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x); return s * (1 - s)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def dtanh(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(x) ** 2


def compute_smooth_relaxation(
    l: torch.Tensor, u: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear relaxation (k_lo, b_lo, k_hi, b_hi) for S-shaped f on [l, u].

    Works element-wise on any broadcastable shape (including batched [B, n]).
    """
    assert (l <= u).all(), "Invalid bounds: l > u"
    f_l, f_u = func(l), func(u)
    denom = (u - l).clamp(min=1e-12)
    k_chord = (f_u - f_l) / denom
    k_lower, k_upper = k_chord.clone(), k_chord.clone()
    b_lower = f_l - k_lower * l
    b_upper = f_l - k_upper * l

    mask_pos = l >= 0
    if mask_pos.any():
        k_tan = dfunc(l)
        k_lower = torch.where(mask_pos, k_tan, k_lower)
        b_lower = torch.where(mask_pos, f_l - k_tan * l, b_lower)

    mask_neg = u <= 0
    if mask_neg.any():
        k_tan = dfunc(u)
        k_upper = torch.where(mask_neg, k_tan, k_upper)
        b_upper = torch.where(mask_neg, f_u - k_tan * u, b_upper)

    return k_lower, b_lower, k_upper, b_upper


def dual_smooth_backward(
    nu: torch.Tensor, bounds: Bounds,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
    M: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched smooth activation backward (sigmoid/tanh) with lazy M-broadcast.

    Same broadcast pattern as :func:`dual_relu_backward`: relaxation
    coefficients ``(k_lower, b_lower, k_upper, b_upper)`` depend on the
    bounds only (spec-agnostic) and are computed once at ``[B, 1, n]``,
    then broadcast against ``nu`` viewed at ``[B, M, n]``.

    Args:
        nu: dual variable, shape ``[B*M, *shape]``.
        bounds: layer bounds, shape ``[B, *shape]``. NOT M-expanded.
        func/dfunc: activation function and its derivative.
        M: spec-row multiplicity (default 1).
    """
    BM = nu.shape[0]
    assert BM % M == 0, f"dual_smooth_backward: nu batch {BM} not divisible by M={M}"
    B = BM // M

    v_flat = nu.flatten(start_dim=1)                              # [BM, n]
    l_B = bounds.lb.flatten(start_dim=1) if bounds.lb.dim() >= 2 \
          else bounds.lb.flatten().unsqueeze(0).expand(B, -1)
    u_B = bounds.ub.flatten(start_dim=1) if bounds.ub.dim() >= 2 \
          else bounds.ub.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]

    l = l_B.unsqueeze(1)                                          # [B, 1, n]
    u = u_B.unsqueeze(1)                                          # [B, 1, n]
    k_lower, b_lower, k_upper, b_upper = compute_smooth_relaxation(l, u, func, dfunc)

    v = v_flat.view(B, M, n)                                      # [B, M, n] view
    v_pos = v >= 0                                                # [B, M, n]
    k = torch.where(v_pos, k_lower, k_upper)                      # broadcast → [B, M, n]
    b = torch.where(v_pos, b_lower, b_upper)

    v_out = v * k                                                 # [B, M, n]
    contrib = (v * b).sum(dim=-1).view(BM)                        # [BM]
    return v_out.view(BM, n), contrib


# ---- SIGMOID ----

@torch.no_grad()
def forward_sigmoid(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for SIGMOID activation.

    Body copied from tf_forward.py lines 426-430 (SIGMOID branch).
    Returns (stored, out, lin, frame).
    """
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = Bounds(torch.sigmoid(pre_lb), torch.sigmoid(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sigmoid(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                     preds: List[int], M: int = 1
                     ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sigmoid: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sigmoid_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIGMOID expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sigmoid_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, sigmoid, dsigmoid, M)


# ---- TANH ----

@torch.no_grad()
def forward_tanh(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for TANH activation.

    Body copied from tf_forward.py lines 432-436 (TANH branch).
    Returns (stored, out, lin, frame).
    """
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = Bounds(torch.tanh(pre_lb), torch.tanh(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_tanh(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_tanh: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_tanh_backward(nu, bounds, M)
    assert len(preds) == 1, f"TANH expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_tanh_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, tanh, dtanh, M)
