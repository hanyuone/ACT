#===- act/back_end/hybridz_tf/tf_transformer.py - HybridZ Transformer TF -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transformer Transfer Functions. Delegates to interval_tf with
#   unified hz_tf_* signature for future HZ expansion.
#
#===---------------------------------------------------------------------===#

from act.back_end.core import Bounds, Fact
import act.back_end.interval_tf.tf_transformer as interval


def tf_posenc(L, bounds, tf):
    return interval.tf_posenc(L, bounds)


def tf_layernorm(L, bounds, tf):
    return interval.tf_layernorm(L, bounds)


def tf_gelu(L, bounds, tf):
    return interval.tf_gelu(L, bounds)


def tf_att_scores(L, bounds, tf):
    return interval.tf_att_scores(L,
        tf._before[L.params["q_src"]].bounds,
        tf._before[L.params["k_src"]].bounds)


def tf_softmax(L, bounds, tf):
    return interval.tf_softmax(L, bounds)


def tf_att_mix(L, bounds, tf):
    return interval.tf_att_mix(L,
        tf._before[L.params["w_src"]].bounds,
        tf._before[L.params["v_src"]].bounds)


def tf_mha_split(L, bounds, tf):
    return interval.tf_mha_split(L, bounds)


def tf_mha_join(L, bounds, tf):
    return interval.tf_mha_join(L,
        tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before))


def tf_mask_add(L, bounds, tf):
    return interval.tf_mask_add(L, bounds)
