# ===- act/back_end/bab/node.py - Subproblem Representation ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Tensor-native representation of BaB subproblems.
#
#   SubproblemBatch is the primary data structure — every field is a tensor
#   with leading batch dimension (N, …) so that branching, bounding, and
#   (future) batched solving operate in pure tensor arithmetic.
#
#   BabNode is a thin single-subproblem wrapper that carries one priority
#   score and one optional candidate counterexample. ``BabNode.to_batch``
#   converts a single node into a ``SubproblemBatch`` of size 1 so that
#   call sites holding a single node can dispatch through the batched
#   code path.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from act.back_end.core import Bounds


# ---------------------------------------------------------------------------
# Tensor-native batch representation (primary)
# ---------------------------------------------------------------------------


@dataclass
class SubproblemBatch:
    """Batched BaB subproblems for tensor-driven processing.

    Shape convention
    ~~~~~~~~~~~~~~~~
    * ``lb``, ``ub``:  ``(N, D)``  — input-space bounds per subproblem.
    * ``depths``:       ``(N,)``    — tree depth of each subproblem.

    All operations on this class are designed for batch-parallel execution.
    Future batch-solving will pass an entire ``SubproblemBatch`` to the
    solver backend in one call.
    """

    lb: torch.Tensor  # (N, D)  lower bounds
    ub: torch.Tensor  # (N, D)  upper bounds
    depths: torch.Tensor  # (N,)    tree depth

    # -- warm-start fields (D.5) --------------------------------------------
    warm_alpha: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n]
    warm_eta: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n]
    split_signs: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n] in {-1, 0, +1}
    parent_margins: Optional[torch.Tensor] = None  # [N]

    # -- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Number of subproblems in this batch."""
        return self.lb.shape[0]

    @property
    def input_dim(self) -> int:
        """Dimensionality of the input space."""
        return self.lb.shape[-1]

    def __len__(self) -> int:
        return self.batch_size

    # -- constructors -------------------------------------------------------

    @staticmethod
    def from_bounds(bounds: Bounds, depth: int = 0) -> SubproblemBatch:
        """Wrap ``Bounds[B, *shape]`` into ``SubproblemBatch[B, D]``.

        Multi-sample inputs (``B > 1`` — e.g., wrapped models carrying several
        specs at once) become ``B`` independent subproblems so each gets its
        own BaB exploration tree. Single-sample (``B == 1``) is the legacy
        path. Unbatched ``Bounds`` (``lb.dim() < 2``) is treated as ``B = 1``.
        """
        lb_raw = bounds.lb.detach()
        ub_raw = bounds.ub.detach()
        b = lb_raw.shape[0] if lb_raw.dim() >= 2 else 1
        lb = lb_raw.reshape(b, -1)
        ub = ub_raw.reshape(b, -1)
        depths = torch.full((b,), depth, dtype=torch.long, device=lb.device)
        return SubproblemBatch(lb=lb, ub=ub, depths=depths)

    # -- conversions --------------------------------------------------------

    def to_bounds_list(self) -> list[Bounds]:
        """Convert back to a list of ``Bounds`` for solver compatibility."""
        return [Bounds(self.lb[i], self.ub[i]) for i in range(self.batch_size)]

    # -- geometry -----------------------------------------------------------

    def widths(self) -> torch.Tensor:
        """Per-dimension widths: ``(N, D)``."""
        return self.ub - self.lb

    def total_width(self) -> torch.Tensor:
        """Sum of widths per subproblem: ``(N,)``."""
        return self.widths().sum(dim=-1)


# ---------------------------------------------------------------------------
# Batch splitting (tensor-native)
# ---------------------------------------------------------------------------


def split_subproblems(
    batch: SubproblemBatch,
    split_dims: torch.Tensor,
) -> tuple[SubproblemBatch, SubproblemBatch]:
    """Bisect each subproblem along the chosen input dimension.

    This is a pure tensor operation — no Python loops over the batch. Warm-state
    fields are deep-copied into both children so they do not alias the parent or
    each other.

    Args:
        batch:      ``(N, D)`` subproblems.
        split_dims: ``(N,)`` long tensor — dimension to bisect per subproblem.

    Returns:
        ``(left, right)`` — two ``SubproblemBatch`` of the same shape,
        where ``left.ub[i, d] == right.lb[i, d] == midpoint``.
    """
    mid = (batch.lb + batch.ub) / 2  # (N, D)
    split_vals = mid.gather(1, split_dims.unsqueeze(1))  # (N, 1)

    # Left child: upper bound clamped at midpoint
    left_ub = batch.ub.clone()
    left_ub.scatter_(1, split_dims.unsqueeze(1), split_vals)

    # Right child: lower bound raised to midpoint
    right_lb = batch.lb.clone()
    right_lb.scatter_(1, split_dims.unsqueeze(1), split_vals)

    new_depths = batch.depths + 1

    def _clone_dict_tensors(
        tensors: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        return (
            {key: tensor.clone() for key, tensor in tensors.items()}
            if tensors is not None
            else None
        )

    left_warm_alpha = _clone_dict_tensors(batch.warm_alpha)
    left_warm_eta = _clone_dict_tensors(batch.warm_eta)
    left_split_signs = _clone_dict_tensors(batch.split_signs)
    left_parent_margins = (
        batch.parent_margins.clone() if batch.parent_margins is not None else None
    )

    right_warm_alpha = _clone_dict_tensors(batch.warm_alpha)
    right_warm_eta = _clone_dict_tensors(batch.warm_eta)
    right_split_signs = _clone_dict_tensors(batch.split_signs)
    right_parent_margins = (
        batch.parent_margins.clone() if batch.parent_margins is not None else None
    )

    left = SubproblemBatch(
        lb=batch.lb.clone(),
        ub=left_ub,
        depths=new_depths.clone(),
        warm_alpha=left_warm_alpha,
        warm_eta=left_warm_eta,
        split_signs=left_split_signs,
        parent_margins=left_parent_margins,
    )
    right = SubproblemBatch(
        lb=right_lb,
        ub=batch.ub.clone(),
        depths=new_depths.clone(),
        warm_alpha=right_warm_alpha,
        warm_eta=right_warm_eta,
        split_signs=right_split_signs,
        parent_margins=right_parent_margins,
    )
    return left, right


# ---------------------------------------------------------------------------
# Single-subproblem wrapper
# ---------------------------------------------------------------------------


@dataclass
class BabNode:
    """Single-subproblem record: one bounds box, one priority score, one
    optional candidate counterexample.

    Prefer :class:`SubproblemBatch` for batch-parallel processing; use this
    record when a call site only ever holds one subproblem at a time and
    wants priority-queue ordering. ``to_batch`` lifts a single node into a
    ``SubproblemBatch`` of size 1 for dispatch through the batched path.
    """

    box: Bounds
    depth: int
    score: float
    candidate_ce: Optional[np.ndarray] = None

    def __lt__(self, other: BabNode) -> bool:  # max-heap by score
        return self.score > other.score

    def to_batch(self) -> SubproblemBatch:
        """Upgrade to tensor batch of size 1."""
        return SubproblemBatch.from_bounds(self.box, depth=self.depth)
