# ===- act/back_end/bab/branching/random_scheduling.py - Scheduling ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Subproblem schedulers for Branch-and-Bound.
#
#   Contains the abstract base class ``Scheduler`` and the
#   ``RandomScheduler`` baseline implementation.
#
#   A scheduler maintains a *pool* of pending subproblems and decides
#   which ones to process next.  All data flows through
#   ``SubproblemBatch`` (tensor-native) so that:
#
#     * ``push`` and ``pop`` operate on batches, not individual nodes.
#     * Internal storage can be a single tensor block (GPU-friendly).
#     * Future batch-parallel BaB pops N subproblems at once for
#       vectorised solving.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from act.back_end.bab.node import SubproblemBatch


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Scheduler(ABC):
    """Abstract subproblem scheduler for Branch-and-Bound.

    Lifecycle (called by the BaB engine)::

        scheduler.push(root_batch)
        while not scheduler.empty:
            batch = scheduler.pop(batch_size=N)
            …solve / branch…
            scheduler.push(children_batch)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``push`` accepts any-sized ``SubproblemBatch``.
    * ``pop(k)`` returns *at most* ``k`` subproblems; fewer if the
      pool is smaller.  Raises ``IndexError`` on empty pool.
    * ``__len__`` returns the current pool size.
    """

    @abstractmethod
    def push(self, batch: SubproblemBatch) -> None:
        """Enqueue a batch of subproblems.

        Args:
            batch: ``(N, D)`` subproblems to add to the pool.
        """
        ...

    @abstractmethod
    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        """Dequeue subproblems according to the scheduling policy.

        Args:
            batch_size: Maximum number of subproblems to return.

        Returns:
            ``(M, D)`` batch where ``M <= batch_size``.

        Raises:
            IndexError: If the pool is empty.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of pending subproblems."""
        ...

    @property
    def empty(self) -> bool:
        """True when no subproblems remain."""
        return len(self) == 0


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomScheduler(Scheduler):
    """Uniform-random subproblem scheduler.

    ``pop(k)`` selects ``k`` subproblems uniformly at random from the
    pool (without replacement).

    Internal storage is fully tensor-native: three tensors ``(M, D)``,
    ``(M, D)``, ``(M,)`` for lower bounds, upper bounds, and depths
    respectively.
    """

    def __init__(self) -> None:
        self._lb: Optional[torch.Tensor] = None  # (M, D)
        self._ub: Optional[torch.Tensor] = None  # (M, D)
        self._depths: Optional[torch.Tensor] = None  # (M,)

    # -- Scheduler interface ------------------------------------------------

    def push(self, batch: SubproblemBatch) -> None:
        if self._lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
        else:
            self._lb = torch.cat([self._lb, batch.lb], dim=0)
            self._ub = torch.cat([self._ub, batch.ub], dim=0)
            self._depths = torch.cat([self._depths, batch.depths], dim=0)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty scheduler")

        n = min(batch_size, len(self))
        perm = torch.randperm(len(self), device=self._lb.device)
        selected = perm[:n]
        remaining = perm[n:]

        result = SubproblemBatch(
            lb=self._lb[selected],
            ub=self._ub[selected],
            depths=self._depths[selected],
        )

        if len(remaining) > 0:
            self._lb = self._lb[remaining]
            self._ub = self._ub[remaining]
            self._depths = self._depths[remaining]
        else:
            self._lb = None
            self._ub = None
            self._depths = None

        return result

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]
