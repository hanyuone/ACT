#===- act/back_end/dual_tf/tf_transformer.py - Transformer Dual TFs -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transformer dual forward/backward handlers (attention-family, LayerNorm,
#   GELU). Registry-contract stubs — bodies pending real implementations.
#   See plan §6.6 / §5.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple, Optional, Dict
from act.back_end.core import Bounds


def forward_attention(L, parent_boxes, parent_lins, parent_frames, preds,
                      post_activation, device, dtype):
    """Multi-head attention forward bounds. (Pending)
    Will require: W_Q, W_K, W_V, W_O, num_heads, per-head bounds.
    Shared by ATT_SCORES / ATT_MIX / MHA_SPLIT / MHA_JOIN / MASK_ADD kinds
    via registry aliasing.
    """
    raise NotImplementedError("forward for attention-family not implemented in dual_tf")


def backward_attention(L, nu, bounds_dict, preds):
    """Attention backward via linearization around reference weights. (Pending)
    Shared by all attention-family kinds via registry aliasing.
    """
    raise NotImplementedError("backward for attention-family not implemented in dual_tf")


def forward_layernorm(L, parent_boxes, parent_lins, parent_frames, preds,
                      post_activation, device, dtype):
    """LayerNorm forward bounds. (Pending)
    Will require: gamma, beta, eps, normalized_shape, reference_mean/var.
    """
    raise NotImplementedError("forward for LAYERNORM not implemented in dual_tf")


def backward_layernorm(L, nu, bounds_dict, preds):
    """LayerNorm backward via linearization around reference mean/var. (Pending)"""
    raise NotImplementedError("backward for LAYERNORM not implemented in dual_tf")


def forward_gelu(L, parent_boxes, parent_lins, parent_frames, preds,
                 post_activation, device, dtype):
    """GELU forward bounds via linear relaxation similar to smooth activations. (Pending)"""
    raise NotImplementedError("forward for GELU not implemented in dual_tf")


def backward_gelu(L, nu, bounds_dict, preds):
    """GELU backward: linear relaxation similar to smooth activations. (Pending)"""
    raise NotImplementedError("backward for GELU not implemented in dual_tf")
