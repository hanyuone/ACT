#===- act/back_end/dual_tf/tf_forward.py - Forward Bounds ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Forward bound propagation for DualTF using linear coefficient tracking.
#   Tracks linear coefficients: output = A @ input + bias
#   Bounds: lb = A @ x0 + bias - |A| @ eps, ub = A @ x0 + bias + |A| @ eps
#   
#   Much tighter than interval propagation for deeper networks.
#   For activation layers, returns PRE-activation bounds (needed by dual backward).
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from act.back_end.core import Bounds, Net, Layer
from act.back_end.layer_schema import LayerKind

# ============================================================================
# Main Entry Point
# ============================================================================

@torch.no_grad()
def compute_forward_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                           post_activation: bool = False) -> Dict[int, Bounds]:
    """
    Compute forward bounds using linear coefficient tracking.
    
    Args:
        net: ACT network
        input_lb, input_ub: Input bounds
        post_activation: If True, return POST-activation bounds (for validation).
                        If False, return PRE-activation bounds (for dual backward).
    """
    bounds_dict: Dict[int, Bounds] = {}
    lb_in, ub_in = input_lb.flatten(), input_ub.flatten()
    input_dim = lb_in.numel()
    device, dtype = lb_in.device, lb_in.dtype
    
    # State: output = A @ input + bias
    x0 = (lb_in + ub_in) / 2  # Center
    eps = (ub_in - lb_in) / 2  # Half-width
    A = torch.eye(input_dim, device=device, dtype=dtype)
    bias = torch.zeros(input_dim, device=device, dtype=dtype)
    lb, ub = lb_in.clone(), ub_in.clone()
    
    for layer in net.layers:
        lid, kind = layer.id, layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        
        # Input layers
        if kind in [LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, "INPUT", "INPUT_SPEC"]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            continue
        
        # Dispatch
        if kind in [LayerKind.RELU.value, "RELU"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            A, bias, lb, ub = _fwd_relu(A, bias, x0, eps, lb, ub)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
                # Reset state after ReLU in post_activation mode for sound interval propagation
                A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in [LayerKind.DENSE.value, "DENSE"]:
            A, bias, lb, ub = _fwd_dense(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind in [LayerKind.CONV2D.value, "CONV2D"]:
            lb, ub = _fwd_conv2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind == "BIAS":
            A, bias, lb, ub = _fwd_bias(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind == "SCALE":
            A, bias, lb, ub = _fwd_scale(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind == "BN":
            A, bias, lb, ub = _fwd_bn(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind in ["FLATTEN", "RESHAPE"]:
            lb, ub = lb.flatten(), ub.flatten()
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind in [LayerKind.SIGMOID.value, "SIGMOID"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            lb, ub = torch.sigmoid(lb), torch.sigmoid(ub)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in [LayerKind.TANH.value, "TANH"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            lb, ub = torch.tanh(lb), torch.tanh(ub)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in ["LRELU", "LEAKY_RELU"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            alpha = float(layer.params.get("alpha", 0.01))
            A, bias, lb, ub = _fwd_lrelu(A, bias, x0, eps, lb, ub, alpha)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
                # Keep forward validation sound: do not keep propagating affine
                # coefficients from an upper relaxation through subsequent layers.
                A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in ["MAXPOOL2D"]:
            lb, ub = _fwd_maxpool2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in ["AVGPOOL2D"]:
            lb, ub = _fwd_avgpool2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in [LayerKind.ASSERT.value, "ASSERT", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE"]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())

        elif kind in [LayerKind.CONSTANT.value, "CONSTANT"]:
            val = layer.params["value"].flatten().to(device=device, dtype=dtype)
            lb, ub = val.clone(), val.clone()
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.SIGN.value, "SIGN"]:
            lb = torch.sign(lb)
            ub = torch.sign(ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.COMPARE.value, "COMPARE"]:
            pred_ids = list(net.preds.get(lid, []) or [])
            if len(pred_ids) >= 2 and all(p in bounds_dict for p in pred_ids[:2]):
                op = layer.params["op"]
                n_out = len(layer.out_vars)
                def _bc(t, n):
                    if t.numel() == n: return t
                    if t.numel() == 1: return t.expand(n)
                    if n % t.numel() == 0: return t.repeat(n // t.numel())
                    return t[:n]
                lb_x = _bc(bounds_dict[pred_ids[0]].lb.flatten(), n_out)
                ub_x = _bc(bounds_dict[pred_ids[0]].ub.flatten(), n_out)
                lb_y = _bc(bounds_dict[pred_ids[1]].lb.flatten(), n_out)
                ub_y = _bc(bounds_dict[pred_ids[1]].ub.flatten(), n_out)
                if op == "lt":
                    dt, df = ub_x < lb_y, lb_x >= ub_y
                elif op == "le":
                    dt, df = ub_x <= lb_y, lb_x > ub_y
                elif op == "gt":
                    dt, df = lb_x > ub_y, ub_x <= lb_y
                elif op == "ge":
                    dt, df = lb_x >= ub_y, ub_x < lb_y
                elif op == "eq":
                    pt = (lb_x == ub_x) & (lb_y == ub_y)
                    dt, df = pt & (lb_x == lb_y), (ub_x < lb_y) | (lb_x > ub_y)
                elif op == "ne":
                    pt = (lb_x == ub_x) & (lb_y == ub_y)
                    dt, df = (ub_x < lb_y) | (lb_x > ub_y), pt & (lb_x == lb_y)
                else:
                    raise ValueError(f"COMPARE forward: unknown op '{op}'")
                z, o = torch.zeros_like(lb_x), torch.ones_like(lb_x)
                lb = torch.where(dt, o, z)
                ub = torch.where(df, z, o)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.WHERE.value, "WHERE"]:
            pred_ids = list(net.preds.get(lid, []) or [])
            if len(pred_ids) >= 3 and all(p in bounds_dict for p in pred_ids[:3]):
                n_out = len(layer.out_vars)
                def _bc(t, n):
                    if t.numel() == n: return t
                    if t.numel() == 1: return t.expand(n)
                    if n % t.numel() == 0: return t.repeat(n // t.numel())
                    return t[:n]
                cl = _bc(bounds_dict[pred_ids[0]].lb.flatten(), n_out)
                cu = _bc(bounds_dict[pred_ids[0]].ub.flatten(), n_out)
                xl = _bc(bounds_dict[pred_ids[1]].lb.flatten(), n_out)
                xu = _bc(bounds_dict[pred_ids[1]].ub.flatten(), n_out)
                yl = _bc(bounds_dict[pred_ids[2]].lb.flatten(), n_out)
                yu = _bc(bounds_dict[pred_ids[2]].ub.flatten(), n_out)
                ct = cl >= 0.5
                cf = cu < 0.5
                lb = torch.where(ct, xl, torch.where(cf, yl, torch.minimum(xl, yl)))
                ub = torch.where(ct, xu, torch.where(cf, yu, torch.maximum(xu, yu)))
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.MATMUL.value, "MATMUL"]:
            pred_ids = list(net.preds.get(lid, []) or [])
            if len(pred_ids) >= 2 and all(p in bounds_dict for p in pred_ids[:2]):
                x_shape = tuple(layer.params["x_shape"])
                y_shape = tuple(layer.params["y_shape"])
                A_lb = bounds_dict[pred_ids[0]].lb.view(*x_shape).unsqueeze(-1)
                A_ub = bounds_dict[pred_ids[0]].ub.view(*x_shape).unsqueeze(-1)
                B_lb = bounds_dict[pred_ids[1]].lb.view(*y_shape).unsqueeze(-3)
                B_ub = bounds_dict[pred_ids[1]].ub.view(*y_shape).unsqueeze(-3)
                c1, c2 = A_lb * B_lb, A_lb * B_ub
                c3, c4 = A_ub * B_lb, A_ub * B_ub
                lo = torch.minimum(torch.minimum(c1, c2), torch.minimum(c3, c4))
                hi = torch.maximum(torch.maximum(c1, c2), torch.maximum(c3, c4))
                lb = lo.sum(dim=-2).reshape(-1)
                ub = hi.sum(dim=-2).reshape(-1)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.ARG_EXTREMUM.value, "ARG_EXTREMUM"]:
            in_shape = layer.params.get("input_shape")
            axis = int(layer.params.get("axis", 0))
            if in_shape is not None and axis < 0:
                axis += len(in_shape)
            axis_dim = int(in_shape[axis]) if in_shape else 1
            n_out = len(layer.out_vars)
            lb = torch.zeros(n_out, dtype=dtype, device=device)
            ub = torch.full((n_out,), float(max(0, axis_dim - 1)), dtype=dtype, device=device)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.UPSAMPLE.value, "UPSAMPLE"]:
            in_shape = layer.params.get("input_shape")
            out_shape = layer.params.get("output_shape")
            mode = str(layer.params.get("mode", "nearest")).lower()
            align_corners = layer.params.get("align_corners")
            if in_shape is not None and out_shape is not None:
                in_shape = tuple(int(d) for d in in_shape)
                out_shape = tuple(int(d) for d in out_shape)
                spatial = len(in_shape) - 2
                if spatial < 1:
                    in_shape = (1, 1) + in_shape
                    out_shape = (1, 1) + out_shape
                if mode == "nearest":
                    torch_mode = "nearest"; ac_kwarg = {}
                else:
                    torch_mode = mode if mode in ("bilinear", "trilinear", "bicubic") else (
                        "bilinear" if len(in_shape) == 4 else "trilinear")
                    ac_kwarg = {"align_corners": bool(align_corners) if align_corners is not None else False}
                tgt = out_shape[-(len(in_shape) - 2):]
                lb = F.interpolate(lb.view(*in_shape), size=tgt, mode=torch_mode, **ac_kwarg).reshape(-1)
                ub = F.interpolate(ub.view(*in_shape), size=tgt, mode=torch_mode, **ac_kwarg).reshape(-1)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.EXPAND.value, "EXPAND"]:
            in_shape = layer.params.get("input_shape")
            out_shape = layer.params.get("output_shape") or layer.params.get("shape")
            if in_shape is not None and out_shape is not None:
                in_shape = tuple(int(d) for d in in_shape)
                out_shape = tuple(int(d) for d in out_shape)
                lb = lb.view(*in_shape).broadcast_to(out_shape).reshape(-1).clone()
                ub = ub.view(*in_shape).broadcast_to(out_shape).reshape(-1).clone()
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.SCATTER_ND.value, "SCATTER_ND"]:
            pred_ids = list(net.preds.get(lid, []) or [])
            if len(pred_ids) >= 3 and all(p in bounds_dict for p in pred_ids[:3]):
                d_lb = bounds_dict[pred_ids[0]].lb.flatten()
                d_ub = bounds_dict[pred_ids[0]].ub.flatten()
                u_lb = bounds_dict[pred_ids[2]].lb.flatten()
                u_ub = bounds_dict[pred_ids[2]].ub.flatten()
                n_out = len(layer.out_vars)
                if d_lb.numel() != n_out:
                    d_lb = d_lb[:n_out] if d_lb.numel() > n_out else d_lb.repeat((n_out + d_lb.numel() - 1) // d_lb.numel())[:n_out]
                    d_ub = d_ub[:n_out] if d_ub.numel() > n_out else d_ub.repeat((n_out + d_ub.numel() - 1) // d_ub.numel())[:n_out]
                if u_lb.numel() > 0:
                    lb = torch.minimum(d_lb, u_lb.min().expand_as(d_lb))
                    ub = torch.maximum(d_ub, u_ub.max().expand_as(d_ub))
                else:
                    lb, ub = d_lb, d_ub
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.REDUCE_SUM.value, "REDUCE_SUM"]:
            axes = layer.params.get("axes")
            keepdims = bool(layer.params.get("keepdims", 0))
            in_shape = layer.params.get("input_shape")
            lb_in, ub_in = lb, ub
            if in_shape is not None and len(in_shape) > 0:
                lb_in = lb_in.view(*in_shape)
                ub_in = ub_in.view(*in_shape)
            dim = tuple(int(a) for a in axes) if axes else tuple(range(lb_in.dim()))
            lb = lb_in.sum(dim=dim, keepdim=keepdims).reshape(-1)
            ub = ub_in.sum(dim=dim, keepdim=keepdims).reshape(-1)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind == "ADD":
            # ADD layer: z = x + y (+ bias if present)
            # Predecessor layer IDs come from the ACT Net graph (``net.preds``).
            # The prior implementation read ``x_src`` / ``y_src`` from
            # ``layer.params``, but ``NetFactory.create_network`` writes the
            # operands into ``params["x_vars"]`` / ``params["y_vars"]``
            # (variable IDs) and the predecessor *layer* IDs into
            # ``net.preds[layer.id]``. The missing keys sent execution down
            # the "keep current lb, ub" fallback, which yielded
            # ``bounds_dict[ADD] == bounds_dict[main_pred]`` (ignoring the
            # skip path) and produced *unsound* bounds on residual nets.
            pred_ids = list(net.preds.get(lid, []) or [])
            if len(pred_ids) >= 2 and pred_ids[0] in bounds_dict and pred_ids[1] in bounds_dict:
                x_src, y_src = pred_ids[0], pred_ids[1]
                lb_x, ub_x = bounds_dict[x_src].lb.flatten(), bounds_dict[x_src].ub.flatten()
                lb_y, ub_y = bounds_dict[y_src].lb.flatten(), bounds_dict[y_src].ub.flatten()
                
                # Handle shape mismatch (broadcasting)
                if lb_x.numel() != lb_y.numel():
                    min_size = min(lb_x.numel(), lb_y.numel())
                    lb_x, ub_x = lb_x[:min_size], ub_x[:min_size]
                    lb_y, ub_y = lb_y[:min_size], ub_y[:min_size]
                
                lb = lb_x + lb_y
                ub = ub_x + ub_y
                
                # Add bias if present
                if "bias" in layer.params and layer.params["bias"] is not None:
                    b = layer.params["bias"].flatten()
                    if b.numel() != lb.numel():
                        b = b[:lb.numel()] if b.numel() > lb.numel() else b.repeat((lb.numel() + b.numel() - 1) // b.numel())[:lb.numel()]
                    lb = lb + b
                    ub = ub + b
            # else: keep current lb, ub as fallback
            
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        else:
            raise NotImplementedError(
                f"DualTF.compute_forward_bounds: layer kind '{kind}' (id={lid}) has no "
                f"forward handler. Implement an `elif` branch in tf_forward.py or remove "
                f"the layer from the network."
            )
    
    return bounds_dict

# ============================================================================
# Layer Handlers
# ============================================================================

def _fwd_dense(layer: Layer, A: torch.Tensor, bias: torch.Tensor, 
               x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense: new = W @ (A @ x + bias) + b = (W @ A) @ x + (W @ bias + b)"""
    W = layer.params["weight"]
    b = layer.params.get("bias")
    
    n_in = W.shape[1]
    if A.shape[0] != n_in:
        if A.shape[0] < n_in:
            pad = n_in - A.shape[0]
            A = torch.cat([A, torch.zeros(pad, A.shape[1])], dim=0)
            bias = torch.cat([bias, torch.zeros(pad)])
        else:
            A, bias = A[:n_in, :], bias[:n_in]
    
    A_new = W @ A
    bias_new = W @ bias + b if b is not None else W @ bias
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    return A_new, bias_new, center - radius, center + radius

def _fwd_relu(A: torch.Tensor, bias: torch.Tensor, x0: torch.Tensor, eps: torch.Tensor,
              lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ReLU forward bounds using CROWN-style upper envelope, interval-style lower bound.
    
    Upper bound: uses linear relaxation y <= (u/(u-l)) * (x - l) for crossing neurons
    Lower bound: uses interval propagation lb_out = max(0, lb_in) for soundness
    
    This ensures sound bounds (lb_out <= true <= ub_out) at the cost of some tightness
    in the lower bound. This is the standard approach for forward propagation.
    """
    device, dtype = lb.device, lb.dtype
    on, off, amb = lb >= 0, ub <= 0, ~((lb >= 0) | (ub <= 0))
    
    # Upper bound linear relaxation slope
    d_ub = torch.where(on, torch.ones_like(lb), torch.zeros_like(lb))
    offset_ub = torch.zeros_like(lb)
    
    if amb.any():
        denom = (ub - lb).clamp(min=1e-12)
        slope = ub / denom
        d_ub = torch.where(amb, slope, d_ub)
        offset_ub = torch.where(amb, -slope * lb, offset_ub)
    
    # Compute upper bound using linear relaxation
    A_ub = d_ub.unsqueeze(1) * A
    bias_ub = d_ub * bias + offset_ub
    center_ub = A_ub @ x0 + bias_ub
    radius_ub = A_ub.abs() @ eps
    ub_out = center_ub + radius_ub
    
    # Lower bound: use interval propagation (sound but looser)
    # lb_out = max(0, lb_in), ub_out from linear relaxation
    lb_out = lb.clamp(min=0)
    
    # For tracking: use upper bound coefficients (will be used for next layer)
    # But the lb_out is computed via interval for soundness
    return A_ub, bias_ub, lb_out, ub_out

def _fwd_bias(layer: Layer, A: torch.Tensor, bias: torch.Tensor,
              x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bias: y = x + c"""
    c = _align(layer.params["c"], bias.numel())
    bias_new = bias + c
    center = A @ x0 + bias_new
    radius = A.abs() @ eps
    return A, bias_new, center - radius, center + radius

def _fwd_scale(layer: Layer, A: torch.Tensor, bias: torch.Tensor,
               x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale: y = a * x"""
    a = _align(layer.params["a"], A.shape[0])
    A_new = a.unsqueeze(1) * A
    bias_new = a * bias
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    return A_new, bias_new, center - radius, center + radius

def _fwd_bn(layer: Layer, A: torch.Tensor, bias: torch.Tensor,
            x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """BatchNorm: y = A_bn * x + c"""
    A_bn = _align(layer.params["A"], A.shape[0])
    c = _align(layer.params["c"], bias.numel())
    A_new = A_bn.unsqueeze(1) * A
    bias_new = A_bn * bias + c
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    return A_new, bias_new, center - radius, center + radius

def _fwd_lrelu(A: torch.Tensor, bias: torch.Tensor, x0: torch.Tensor, eps: torch.Tensor,
               lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leaky ReLU: y = x if x >= 0, y = α*x if x < 0
    
    Upper bound relaxation (line through (l, αl) and (u, u)):
    - Active (lb >= 0): d = 1, offset = 0
    - Inactive (ub <= 0): d = α, offset = 0
    - Crossing: d = (u - αl) / (u - l), offset = αl - d*l
    """
    on, off, amb = lb >= 0, ub <= 0, ~((lb >= 0) | (ub <= 0))
    
    # Active: slope = 1, Inactive: slope = α
    d = torch.where(on, torch.ones_like(lb), torch.full_like(lb, alpha))
    offset = torch.zeros_like(lb)
    
    if amb.any():
        # Line through (l, αl) and (u, u): slope = (u - αl) / (u - l)
        denom = (ub - lb).clamp(min=1e-12)
        slope = (ub - alpha * lb) / denom
        # offset = αl - slope * l = l * (α - slope)
        off_val = alpha * lb - slope * lb
        d = torch.where(amb, slope, d)
        offset = torch.where(amb, off_val, offset)
    
    A_new = d.unsqueeze(1) * A
    bias_new = d * bias + offset
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    ub_out = center + radius

    # Sound lower bound for monotone LeakyReLU:
    # f(x)=x for x>=0, alpha*x for x<0 (alpha>0), so minimum over [lb,ub] is f(lb).
    lb_out = torch.where(on, lb, alpha * lb)
    return A_new, bias_new, lb_out, ub_out

def _fwd_conv2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conv2D interval based"""
    weight, bias = layer.params["weight"], layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    
    if isinstance(stride, (list, tuple)): stride = stride[0]
    if isinstance(padding, (list, tuple)): padding = padding[0]
    if isinstance(dilation, (list, tuple)): dilation = dilation[0]
    
    out_c, in_c_per_g, kH, kW = weight.shape
    in_c = in_c_per_g * groups
    spatial = lb.numel() // in_c
    in_h = in_w = int(spatial ** 0.5)
    
    try:
        lb_4d, ub_4d = lb.view(1, in_c, in_h, in_w), ub.view(1, in_c, in_h, in_w)
    except RuntimeError:
        return lb, ub
    
    W_pos, W_neg = weight.clamp(min=0), weight.clamp(max=0)
    lb_out = F.conv2d(lb_4d, W_pos, None, stride, padding, dilation, groups) + \
             F.conv2d(ub_4d, W_neg, None, stride, padding, dilation, groups)
    ub_out = F.conv2d(ub_4d, W_pos, None, stride, padding, dilation, groups) + \
             F.conv2d(lb_4d, W_neg, None, stride, padding, dilation, groups)
    
    if bias is not None:
        lb_out, ub_out = lb_out + bias.view(1, -1, 1, 1), ub_out + bias.view(1, -1, 1, 1)
    return lb_out.flatten(), ub_out.flatten()

def _fwd_maxpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2D (interval based)"""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    input_shape = layer.params.get("input_shape")
    if input_shape is None: return lb, ub
    
    b, c, h, w = input_shape
    lb_out = F.max_pool2d(lb.view(b, c, h, w), kernel_size, stride, padding, dilation)
    ub_out = F.max_pool2d(ub.view(b, c, h, w), kernel_size, stride, padding, dilation)
    return lb_out.flatten(), ub_out.flatten()

def _fwd_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D (interval based)"""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    input_shape = layer.params.get("input_shape")
    if input_shape is None: return lb, ub
    
    b, c, h, w = input_shape
    lb_out = F.avg_pool2d(lb.view(b, c, h, w), kernel_size, stride, padding)
    ub_out = F.avg_pool2d(ub.view(b, c, h, w), kernel_size, stride, padding)
    return lb_out.flatten(), ub_out.flatten()

# ============================================================================
# Helpers
# ============================================================================

def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align tensor to size n."""
    a = a.flatten()
    if a.numel() == n: return a
    elif a.numel() > n: return a[:n]
    else: return a.repeat((n + a.numel() - 1) // a.numel())[:n]

def _reset_state(lb: torch.Tensor, ub: torch.Tensor, device, dtype):
    """Reset linear tracking state after non-linear layers."""
    curr_dim = lb.numel()
    A = torch.eye(curr_dim, device=device, dtype=dtype)
    bias = torch.zeros(curr_dim, device=device, dtype=dtype)
    x0 = (lb + ub) / 2
    eps = (ub - lb) / 2
    return A, bias, x0, eps