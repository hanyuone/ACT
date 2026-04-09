#===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
#===---------------------------------------------------------------------===#


import torch
import torch.nn.functional as F
from typing import List, Tuple
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D convolution with enhanced precision."""
    # Extract convolution parameters
    weight = L.params["weight"]  # (out_channels, in_channels, kernel_h, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)

    # Infer spatial dims from bounds size + weight shape (same approach as interval_tf)
    out_ch, in_ch_per_group, kh, kw = weight.shape
    in_ch = in_ch_per_group * groups
    spatial = Bin.lb.numel() // in_ch
    in_h = in_w = int(spatial ** 0.5)
    if in_h * in_w != spatial:
        for h in range(int(spatial ** 0.5) + 10, 0, -1):
            if spatial % h == 0:
                in_h, in_w = h, spatial // h
                break

    Bin_reshaped_lb = Bin.lb.view(1, in_ch, in_h, in_w)
    Bin_reshaped_ub = Bin.ub.view(1, in_ch, in_h, in_w)
    
    # Apply convolution to bounds
    # For HybridZ: more precise bound computation considering kernel structure
    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)
    
    # Lower bound: positive weights * lower bounds + negative weights * upper bounds
    lb_conv = F.conv2d(Bin_reshaped_lb, weight_pos, bias=None, stride=stride, 
                       padding=padding, dilation=dilation, groups=groups)
    lb_conv += F.conv2d(Bin_reshaped_ub, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    # Upper bound: positive weights * upper bounds + negative weights * lower bounds  
    ub_conv = F.conv2d(Bin_reshaped_ub, weight_pos, bias=None, stride=stride,
                       padding=padding, dilation=dilation, groups=groups)
    ub_conv += F.conv2d(Bin_reshaped_lb, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1)
    
    # Flatten output if needed
    lb = lb_conv.reshape(-1)
    ub = ub_conv.reshape(-1)
    assert lb.numel() == len(L.out_vars)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    # Generate convolution constraints
    cons = ConSet()
    cons.add_op( f"conv2d:{L.id}", list(L.out_vars + L.in_vars), weight=weight, 
                bias=bias if bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype),
                stride=stride, padding=padding, dilation=dilation, groups=groups, input_shape=L.params.get("input_shape"), output_shape=L.params.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D max pooling."""
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    
    # Reshape input if flattened
    in_shape = L.params.get("input_shape")  # (channels, height, width)
    if in_shape is not None:
        s = list(in_shape)
        C, H, W = (s[1], s[2], s[3]) if len(s) == 4 else (s[0], s[1], s[2])
    else:
        raise ValueError(f"MAXPOOL2D layer {L.id} missing input_shape")
    Bin_lb = Bin.lb.view(1, C, H, W)
    Bin_ub = Bin.ub.view(1, C, H, W)

    # Max pooling: upper bounds of pooling regions
    # For HybridZ: track which neurons contribute to maximum
    lb_pool = F.max_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.max_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    
    # Flatten output to 1-D (matching len(L.out_vars))
    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)

    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    # Max pooling generates max constraints
    cons.add_op( f"maxpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size, 
        stride=stride, padding=padding, input_shape=in_shape, output_shape=L.params.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D average pooling."""
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    
    # Infer spatial dims from input_shape metadata
    in_shape = L.params.get("input_shape")
    if in_shape is not None:
        s = list(in_shape)
        C, H, W = (s[1], s[2], s[3]) if len(s) == 4 else (s[0], s[1], s[2])
    else:
        raise ValueError(f"AVGPOOL2D layer {L.id} missing input_shape")
    Bin_lb = Bin.lb.view(1, C, H, W)
    Bin_ub = Bin.ub.view(1, C, H, W)

    # Average pooling is linear - exact bounds
    lb_pool = F.avg_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    
    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(
        f"avgpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size, stride=stride,
        padding=padding, input_shape=in_shape, output_shape=L.params.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for tensor flattening."""
    # Flattening is just reshaping - bounds remain the same
    start_dim = L.params.get("start_dim", 1)
    end_dim = L.params.get("end_dim", -1)
    
    # Simple reshape - no change in bounds
    lb = Bin.lb.flatten()
    ub = Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"flatten:{L.id}", list(L.out_vars + L.in_vars), start_dim=start_dim, end_dim=end_dim, input_shape=L.params.get("input_shape"), output_shape=L.params.get("output_shape"))
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for general tensor reshaping."""
    target_shape = L.params.get("target_shape")
    
    # Reshape bounds preserving values
    lb = Bin.lb.reshape(target_shape) if target_shape else Bin.lb
    ub = Bin.ub.reshape(target_shape) if target_shape else Bin.ub
    
    # Flatten for output variables
    lb = lb.flatten()
    ub = ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"reshape:{L.id}", list(L.out_vars + L.in_vars), target_shape=target_shape, input_shape=L.params.get("input_shape"), output_shape=L.params.get("output_shape"))
    
    return Fact(bounds=Bout, cons=cons)