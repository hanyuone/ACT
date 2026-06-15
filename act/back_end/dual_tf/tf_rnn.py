#===- act/back_end/dual_tf/tf_rnn.py - RNN Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   RNN dual forward/backward handlers (LSTM, GRU). Registry-contract stubs
#   — bodies pending real implementations.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple, Optional, Dict
from act.back_end.core import Bounds


def forward_lstm(L, parent_boxes, parent_lins, parent_frames, preds,
                 post_activation, device, dtype):
    """LSTM forward bounds via linear relaxation through time. (Pending)
    Will require: gate bounds per timestep, weight_ih/weight_hh, seq_len,
    hidden_size.
    """
    raise NotImplementedError("forward for LSTM not implemented in dual_tf")


def backward_lstm(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """LSTM backward through time. (Pending)
    Will require: gate bounds from forward pass, per-timestep weight matrices
    (weight_ih, weight_hh), seq_len, hidden_size.
    """
    raise NotImplementedError("backward for LSTM not implemented in dual_tf")


def forward_gru(L, parent_boxes, parent_lins, parent_frames, preds,
                post_activation, device, dtype):
    """GRU forward bounds via linear relaxation through time. (Pending)
    Will require: gate bounds per timestep, weight_ih/weight_hh, seq_len,
    hidden_size.
    """
    raise NotImplementedError("forward for GRU not implemented in dual_tf")


def backward_gru(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """GRU backward through time. (Pending)
    Will require: gate bounds from forward pass, per-timestep weight matrices.
    """
    raise NotImplementedError("backward for GRU not implemented in dual_tf")
