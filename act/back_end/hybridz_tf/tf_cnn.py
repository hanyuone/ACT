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
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import HZono, hz_compute_bounds
from act.back_end.hybridz_tf.tf_mlp import _hz_fact
import act.back_end.interval_tf.tf_cnn as interval


# --- HZ transfer functions (CNN) ---

def tf_conv2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        input_shape = L.params.get("input_shape")
        if input_shape is not None:
            tf._hz_cache[L.id] = hz_conv2d(
                hz_in, L.params["weight"], L.params.get("bias"),
                L.params.get("stride", 1), L.params.get("padding", 0),
                L.params.get("dilation", 1), L.params.get("groups", 1), input_shape,
            )
        else:
            hz_in = None
    fact = interval.tf_conv2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_maxpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        bds = hz_compute_bounds(hz_in)
        shape = L.params.get("input_shape")
        done = False
        if shape is not None:
            if len(shape) == 4:
                _, C, H, W = shape
            elif len(shape) == 3:
                C, H, W = shape
            else:
                hz_in = None
            if hz_in is not None:
                spatial = C * H * W
                B = bds.lb.numel() // spatial
                _, idx = F.max_pool2d(
                    bds.lb.view(B, C, H, W),
                    kernel_size=L.params.get("kernel_size", 2),
                    stride=L.params.get("stride", L.params.get("kernel_size", 2)),
                    padding=L.params.get("padding", 0),
                    return_indices=True,
                )
                # idx values are per-batch flat indices in [0, C*H*W);
                # add per-batch offsets to recover global flat indices in
                # [0, B*C*H*W) so hz_in.c[w] / .Gc[w] / .Gb[w] index the
                # correct batch element's row.
                offsets = (
                    torch.arange(B, device=idx.device, dtype=idx.dtype)
                    .view(B, 1, 1, 1) * spatial
                )
                w = (idx + offsets).reshape(-1)
                tf._hz_cache[L.id] = HZono(
                    c=hz_in.c[w], Gc=hz_in.Gc[w], Gb=hz_in.Gb[w],
                    Ac=hz_in.Ac.clone(), Ab=hz_in.Ab.clone(), b=hz_in.b.clone(),
                )
                done = True
        if not done:
            hz_in = None
    fact = interval.tf_maxpool2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


# --- HZ conv2d (zonotope domain) ---

def _conv2d_generators(
    G, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
):
    """Apply conv2d to a generator matrix ``(B*C*H*W, ng)`` and return
    a generator matrix ``(B*n_out_per_sample, ng)``. Each generator
    column is convolved independently per batch element by stacking
    ``ng * B`` images into conv2d's leading "batch" axis.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    # (B*C*H*W, ng) → (ng, B*C*H*W) → (ng, B, C, H, W) → (ng*B, C, H, W)
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out.shape
    # (ng*B, Cp, Hp, Wp) → (ng, B, Cp, Hp, Wp) → (B, Cp, Hp, Wp, ng)
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    """Apply conv2d to a hybrid zonotope: convolve the center as one
    ``(B, C, H, W)`` image and each generator column as ``B`` per-batch
    images. ``B`` is recovered from ``hz.c.numel() // (C*H*W)`` so this
    works uniformly for B=1 and B>1 without materialising a
    block-diagonal weight.
    """
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}, expected 3D or 4D")
    weight = weight.to(hz.c)

    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    c_img = hz.c.view(B, C, H, W)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(hz.c) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out_c.shape
    new_c = out_c.reshape(-1, 1)
    n_out_per_sample = Cp * Hp * Wp

    new_Gc = _conv2d_generators(
        hz.Gc, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )
