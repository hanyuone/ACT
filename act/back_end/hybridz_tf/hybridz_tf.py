# ===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation.
#
#   Each hz_tf_* is a complete TF for one layer kind, combining
#   HZ zonotope propagation with interval_tf constraint generation.
#   hz_tf_* live in tf_mlp.py / tf_cnn.py alongside their layer types.
#   HZ domain ops co-locate with the hz_tf_* that use them.
#
# ===---------------------------------------------------------------------===#

""" """

import torch
from typing import Dict, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import HZono, hz_from_bounds, hz_compute_bounds

from act.back_end.hybridz_tf.tf_mlp import (
    hz_tf_dense,
    hz_tf_bias,
    hz_tf_scale,
    hz_tf_relu,
    hz_tf_lrelu,
    hz_tf_tanh,
    hz_tf_sigmoid,
    hz_tf_abs,
    hz_tf_bn,
    hz_tf_add,
    hz_tf_mul,
    hz_tf_concat,
)
from act.back_end.hybridz_tf.tf_cnn import hz_tf_conv2d, hz_tf_maxpool2d

from act.back_end.interval_tf.tf_mlp import (
    tf_clip,
    tf_softplus,
    tf_silu,
    tf_relu6,
    tf_hardtanh,
    tf_hardsigmoid,
    tf_hardswish,
    tf_mish,
    tf_softsign,
    tf_square,
    tf_power,
    tf_max,
    tf_min,
    tf_reshape,
    tf_transpose,
    tf_squeeze,
    tf_unsqueeze,
    tf_tile,
    tf_expand,
)
from act.back_end.interval_tf.tf_cnn import (
    tf_avgpool2d,
    tf_conv1d,
    tf_conv3d,
    tf_convtranspose2d,
    tf_flatten,
)
from act.back_end.hybridz_tf.tf_rnn import (
    hz_tf_lstm,
    hz_tf_gru,
    hz_tf_rnn,
    hz_tf_embedding,
)
from act.back_end.hybridz_tf.tf_transformer import (
    hz_tf_posenc,
    hz_tf_layernorm,
    hz_tf_gelu,
    hz_tf_att_scores,
    hz_tf_softmax,
    hz_tf_att_mix,
    hz_tf_mha_split,
    hz_tf_mha_join,
    hz_tf_mask_add,
)


class HybridzTF(TransferFunction):
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 2
        self._sigmoid_K: int = 2

    _LAYER_REGISTRY = {
        LayerKind.INPUT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        # MLP: HZ + interval
        LayerKind.DENSE.value: lambda L, b, tf: hz_tf_dense(L, b, tf),
        "BIAS": lambda L, b, tf: hz_tf_bias(L, b, tf),
        "SCALE": lambda L, b, tf: hz_tf_scale(L, b, tf),
        LayerKind.RELU.value: lambda L, b, tf: hz_tf_relu(L, b, tf),
        "LRELU": lambda L, b, tf: hz_tf_lrelu(L, b, tf),
        "TANH": lambda L, b, tf: hz_tf_tanh(L, b, tf),
        "SIGMOID": lambda L, b, tf: hz_tf_sigmoid(L, b, tf),
        "ABS": lambda L, b, tf: hz_tf_abs(L, b, tf),
        "BN": lambda L, b, tf: hz_tf_bn(L, b, tf),
        # Multi-input: HZ + interval
        "ADD": lambda L, b, tf: hz_tf_add(L, b, tf),
        "MUL": lambda L, b, tf: hz_tf_mul(L, b, tf),
        "CONCAT": lambda L, b, tf: hz_tf_concat(L, b, tf),
        # CNN: HZ + interval
        "CONV2D": lambda L, b, tf: hz_tf_conv2d(L, b, tf),
        "MAXPOOL2D": lambda L, b, tf: hz_tf_maxpool2d(L, b, tf),
        # Interval-only activations
        "CLIP": lambda L, b, tf: tf_clip(L, b),
        "SOFTPLUS": lambda L, b, tf: tf_softplus(L, b),
        "SILU": lambda L, b, tf: tf_silu(L, b),
        "RELU6": lambda L, b, tf: tf_relu6(L, b),
        "HARDTANH": lambda L, b, tf: tf_hardtanh(L, b),
        "HARDSIGMOID": lambda L, b, tf: tf_hardsigmoid(L, b),
        "HARDSWISH": lambda L, b, tf: tf_hardswish(L, b),
        "MISH": lambda L, b, tf: tf_mish(L, b),
        "SOFTSIGN": lambda L, b, tf: tf_softsign(L, b),
        "SQUARE": lambda L, b, tf: tf_square(L, b),
        "POWER": lambda L, b, tf: tf_power(L, b),
        "MAX": lambda L, b, tf: tf_max(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        "MIN": lambda L, b, tf: tf_min(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        # CNN interval-only
        "AVGPOOL2D": lambda L, b, tf: tf_avgpool2d(L, b),
        "CONV1D": lambda L, b, tf: tf_conv1d(L, b),
        "CONV3D": lambda L, b, tf: tf_conv3d(L, b),
        "CONVTRANSPOSE2D": lambda L, b, tf: tf_convtranspose2d(L, b),
        "FLATTEN": lambda L, b, tf: tf_flatten(L, b),
        # Shape ops
        "RESHAPE": lambda L, b, tf: tf_reshape(L, b),
        "TRANSPOSE": lambda L, b, tf: tf_transpose(L, b),
        "SQUEEZE": lambda L, b, tf: tf_squeeze(L, b),
        "UNSQUEEZE": lambda L, b, tf: tf_unsqueeze(L, b),
        "TILE": lambda L, b, tf: tf_tile(L, b),
        "EXPAND": lambda L, b, tf: tf_expand(L, b),
        # RNN
        "LSTM": lambda L, b, tf: hz_tf_lstm(L, b, tf),
        "GRU": lambda L, b, tf: hz_tf_gru(L, b, tf),
        "RNN": lambda L, b, tf: hz_tf_rnn(L, b, tf),
        "EMBEDDING": lambda L, b, tf: hz_tf_embedding(L, b, tf),
        # Transformer
        "EMBEDDING_TF": lambda L, b, tf: hz_tf_embedding(L, b, tf),
        "POSENC": lambda L, b, tf: hz_tf_posenc(L, b, tf),
        "LAYERNORM": lambda L, b, tf: hz_tf_layernorm(L, b, tf),
        "GELU": lambda L, b, tf: hz_tf_gelu(L, b, tf),
        "ATT_SCORES": lambda L, b, tf: hz_tf_att_scores(L, b, tf),
        "SOFTMAX": lambda L, b, tf: hz_tf_softmax(L, b, tf),
        "ATT_MIX": lambda L, b, tf: hz_tf_att_mix(L, b, tf),
        "MHA_SPLIT": lambda L, b, tf: hz_tf_mha_split(L, b, tf),
        "MHA_JOIN": lambda L, b, tf: hz_tf_mha_join(L, b, tf),
        "MASK_ADD": lambda L, b, tf: tf_mask_add(L, b),
    }

    @property
    def name(self) -> str:
        return "HybridzTF"

    def supports_layer(self, layer_kind: str) -> bool:
        return layer_kind.upper() in self._LAYER_REGISTRY

    _HZ_MAX_INPUT_DIM = 1024

    def _hz_from_bounds(self, bounds: Bounds) -> Optional[HZono]:
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        n = lb.shape[0]
        if n > self._HZ_MAX_INPUT_DIM:
            return None
        dtype, device = lb.dtype, lb.device
        c = ((lb + ub) / 2.0).view(-1, 1)
        rad = (ub - lb) / 2.0
        return HZono(
            c=c,
            Gc=torch.diag(rad),
            Gb=torch.zeros((n, 0), dtype=dtype, device=device),
            Ac=torch.zeros((0, n), dtype=dtype, device=device),
            Ab=torch.zeros((0, 0), dtype=dtype, device=device),
            b=torch.zeros((0, 1), dtype=dtype, device=device),
        )

    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")

        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._cache_net_id = net_id

        self._net = net
        self._before = before
        self._after = after

        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(input_bounds)
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        elif k != "ASSERT":
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]

        hz_before = self._hz_cache.get(L.id)
        result = self._LAYER_REGISTRY[k](L, input_bounds, self)

        if hz_before is not None and self._hz_cache.get(L.id) is hz_before:
            self._hz_cache[L.id] = hz_from_bounds(
                result.bounds, result.bounds.lb.dtype, result.bounds.lb.device
            )

        return result
