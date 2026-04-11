#===- act/back_end/hybridz_tf/tf_rnn.py - HybridZ RNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ RNN Transfer Functions. Delegates to interval_tf with unified
#   hz_tf_* signature for future HZ expansion.
#
#===---------------------------------------------------------------------===#

from act.back_end.core import Bounds, Fact
import act.back_end.interval_tf.tf_rnn as interval


def tf_lstm(L, bounds, tf):
    return interval.tf_lstm(L, bounds)


def tf_gru(L, bounds, tf):
    return interval.tf_gru(L, bounds)


def tf_rnn(L, bounds, tf):
    return interval.tf_rnn(L, bounds)


def tf_embedding(L, bounds, tf):
    return interval.tf_embedding(L, bounds)
