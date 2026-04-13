#===- act/back_end/hybridz_tf/__init__.py - HybridZ Transfer Functions --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Functions. Constraint generation reuses interval_tf
#   (single source of truth). HZ precision via solver_hz zonotope operations.
#
#===---------------------------------------------------------------------===#

from .hybridz_tf import HybridzTF

__all__ = [
    "HybridzTF",
]
