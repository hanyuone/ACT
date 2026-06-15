# ===- act/back_end/bab/branching/bounding.py - Subproblem Bounding ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Subproblem pool management for Branch-and-Bound.
#
#   Pool strategies (subclasses of ``BoundingStrategy``):
#     * ``RandomBounding`` — uniform-random subproblem selection. Baseline.
#     * ``TopKBounding`` — keep the top-k (bounded by the batch/tensor size) ranked
#       by a swappable order strategy; supports frontier-cap eviction (``evict_to``),
#       which drops worst-priority leaves and forces a sound ``UNKNOWN``.
#
#   Order strategies (the ``OrderFunction`` Protocol; pluggable via ``ORDER_REGISTRY``):
#     * ``DepthLowerBoundOrder`` — depth + lower-bound blend (default, 50/50).
#     * ``GreedyOrder`` — best-first by lower bound (Oliva-Greedy, ``|lb|``).
#     * ``SAOrder`` — simulated-annealing exploration (temperature-annealed; stochastically cools to greedy).
#   ``GreedyOrder`` / ``SAOrder`` implement the Oliva order-leading exploration of the
#   BaB tree — "Efficient Neural Network Verification via Order Leading Exploration of
#   Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang, H.M.N. Dilum
#   Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025 (arXiv:2507.17453).
#
#   A bounding strategy maintains a *pool* of pending subproblems and
#   decides which ones to process next.  All data flows through
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
from typing import Dict, Optional, Protocol

import torch

from act.back_end.bab.node import SubproblemBatch


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BoundingStrategy(ABC):
    """Abstract subproblem pool for Branch-and-Bound.

    Lifecycle (called by the BaB engine)::

        pool.push(root_batch)
        while not pool.empty:
            batch = pool.pop(batch_size=N)
            …solve / branch…
            pool.push(children_batch)

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
        """Dequeue subproblems for the next bounding iteration.

        Args:
            batch_size: Maximum number of subproblems to return.

        Returns:
            ``(M, D)`` batch where ``M <= batch_size``.

        Raises:
            IndexError: If the pool is empty.
        """
        ...

    @abstractmethod
    def evict_to(self, cap: int) -> int:
        """Drop pending subproblems until at most cap remain."""
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


class RandomBounding(BoundingStrategy):
    """Uniform-random subproblem selection.

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

    # -- BoundingStrategy interface -----------------------------------------

    def push(self, batch: SubproblemBatch) -> None:
        if self._lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
        else:
            assert self._ub is not None and self._depths is not None
            self._lb = torch.cat([self._lb, batch.lb], dim=0)
            self._ub = torch.cat([self._ub, batch.ub], dim=0)
            self._depths = torch.cat([self._depths, batch.depths], dim=0)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty pool")

        n = min(batch_size, len(self))
        assert self._lb is not None and self._ub is not None and self._depths is not None
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

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        assert self._lb is not None and self._ub is not None and self._depths is not None
        self._lb = self._lb[:cap]
        self._ub = self._ub[:cap]
        self._depths = self._depths[:cap]
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]


# ---------------------------------------------------------------------------
# Top-k priority selection (quantitative total order: depth + lower bound)
# ---------------------------------------------------------------------------


def _clone_optional_dict(
    d: Optional[Dict[int, torch.Tensor]],
) -> Optional[Dict[int, torch.Tensor]]:
    return {key: tensor.clone() for key, tensor in d.items()} if d is not None else None


def _index_optional_dict(
    d: Optional[Dict[int, torch.Tensor]], idx: torch.Tensor
) -> Optional[Dict[int, torch.Tensor]]:
    if d is None:
        return None
    return {key: tensor.index_select(0, idx.to(tensor.device)) for key, tensor in d.items()}


def _merge_optional_dict(
    existing: Optional[Dict[int, torch.Tensor]],
    n_existing: int,
    incoming: Optional[Dict[int, torch.Tensor]],
    n_incoming: int,
) -> Optional[Dict[int, torch.Tensor]]:
    # Per-key concat with key-union; a subproblem missing a key is padded with
    # zeros (e.g. split_signs keys differ per branch — a missing layer means "not
    # split", i.e. all-zero signs). Keeps the pool lossless across heterogeneous
    # incremental-state/split structures.
    if existing is None and incoming is None:
        return None
    existing = existing or {}
    incoming = incoming or {}
    merged: Dict[int, torch.Tensor] = {}
    for key in sorted(set(existing) | set(incoming)):
        te = existing.get(key)
        ti = incoming.get(key)
        ref = te if te is not None else ti
        assert ref is not None
        if te is not None and ti is not None:
            assert te.shape[1:] == ti.shape[1:], (
                f"optional dict key {key} trailing shape mismatch: "
                f"existing {tuple(te.shape[1:])} vs incoming {tuple(ti.shape[1:])}"
            )
        trailing = ref.shape[1:]
        if te is None:
            te = torch.zeros((n_existing, *trailing), dtype=ref.dtype, device=ref.device)
        if ti is None:
            ti = torch.zeros((n_incoming, *trailing), dtype=ref.dtype, device=ref.device)
        merged[key] = torch.cat([te, ti], dim=0)
    return merged


class OrderFunction(Protocol):
    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        ...


class DepthLowerBoundOrder:
    def __init__(self, depth_weight: float = 0.5, bound_weight: float = 0.5) -> None:
        self.depth_weight = depth_weight
        self.bound_weight = bound_weight

    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        dtype = lower_bound.dtype
        eps = torch.finfo(dtype).eps
        d = depths.to(dtype=dtype)
        d_norm = (d - d.min()) / (d.max() - d.min()).clamp(min=eps)
        urgency = (lower_bound.max() - lower_bound) / (
            lower_bound.max() - lower_bound.min()
        ).clamp(min=eps)
        return self.depth_weight * d_norm + self.bound_weight * urgency


class GreedyOrder(DepthLowerBoundOrder):
    """Oliva-Greedy: best-first by lower bound (``|lb|``).

    "Efficient Neural Network Verification via Order Leading Exploration of
    Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang,
    H.M.N. Dilum Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025.
    """

    def __init__(self) -> None:
        super().__init__(depth_weight=0.0, bound_weight=1.0)


class SAOrder:
    """Oliva-SA: temperature-annealed exploration order.

    "Efficient Neural Network Verification via Order Leading Exploration of
    Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang,
    H.M.N. Dilum Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025.

    ``temp = cooling_rate ** step`` cools each call, so selection explores early and
    converges to greedy (``|lb|`` best-first) as it cools.
    """

    def __init__(self, cooling_rate: float = 0.99) -> None:
        self.cooling_rate = cooling_rate
        self.step = 0

    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        dtype = lower_bound.dtype
        eps = torch.finfo(dtype).eps
        temp = max(self.cooling_rate ** self.step, 1e-6)
        self.step += 1
        base = (lower_bound.max() - lower_bound) / (
            lower_bound.max() - lower_bound.min()
        ).clamp(min=eps)
        u = torch.rand_like(base).clamp(min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        return base / temp + gumbel


ORDER_REGISTRY: Dict[str, type] = {
    "depth_lb": DepthLowerBoundOrder,
    "greedy": GreedyOrder,
    "sa": SAOrder,
}


class TopKBounding(BoundingStrategy):
    """Priority pool: keep the top-k subproblems chosen by an order callable.

    The BaB tensor (batch) size is capped by compute resources, so when the pool
    holds more subproblems than the requested batch size, the next wave keeps only
    the k highest-priority ones; the rest stay pooled. Priority comes from a
    swappable order strategy (default :class:`DepthLowerBoundOrder` — a
    50/50 blend of depth and lower bound).

    Storage is lossless: bounds, depth, lower bound, parent margins and every
    incremental-state dict (including split_signs, which neuron-split BaB requires) are
    preserved across push/pop.
    """

    def __init__(self, order: Optional[OrderFunction] = None) -> None:
        self.order: OrderFunction = order if order is not None else DepthLowerBoundOrder()
        self._lb: Optional[torch.Tensor] = None
        self._ub: Optional[torch.Tensor] = None
        self._depths: Optional[torch.Tensor] = None
        self._lower_bound: Optional[torch.Tensor] = None
        self._parent_margins: Optional[torch.Tensor] = None
        self._node_id: Optional[torch.Tensor] = None
        self._parent_id: Optional[torch.Tensor] = None
        self._incremental_alpha: Optional[Dict[int, torch.Tensor]] = None
        self._incremental_eta: Optional[Dict[int, torch.Tensor]] = None
        self._split_signs: Optional[Dict[int, torch.Tensor]] = None

    def push(self, batch: SubproblemBatch) -> None:
        n_new = batch.batch_size
        device, dtype = batch.lb.device, batch.lb.dtype
        lower = (
            batch.lower_bound
            if batch.lower_bound is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        parent = (
            batch.parent_margins
            if batch.parent_margins is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        prev_lb, prev_ub, prev_depths = self._lb, self._ub, self._depths
        prev_lower, prev_parent = self._lower_bound, self._parent_margins
        if prev_lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
            self._lower_bound = lower.clone()
            self._parent_margins = parent.clone()
            self._node_id = batch.node_id.clone() if batch.node_id is not None else None
            self._parent_id = batch.parent_id.clone() if batch.parent_id is not None else None
            self._incremental_alpha = _clone_optional_dict(batch.incremental_alpha)
            self._incremental_eta = _clone_optional_dict(batch.incremental_eta)
            self._split_signs = _clone_optional_dict(batch.split_signs)
            return

        assert prev_ub is not None and prev_depths is not None
        assert prev_lower is not None and prev_parent is not None
        assert (self._node_id is None) == (batch.node_id is None)
        assert (self._parent_id is None) == (batch.parent_id is None)
        n_old = prev_lb.shape[0]
        self._incremental_alpha = _merge_optional_dict(self._incremental_alpha, n_old, batch.incremental_alpha, n_new)
        self._incremental_eta = _merge_optional_dict(self._incremental_eta, n_old, batch.incremental_eta, n_new)
        self._split_signs = _merge_optional_dict(self._split_signs, n_old, batch.split_signs, n_new)
        self._lb = torch.cat([prev_lb, batch.lb], dim=0)
        self._ub = torch.cat([prev_ub, batch.ub], dim=0)
        self._depths = torch.cat([prev_depths, batch.depths], dim=0)
        self._lower_bound = torch.cat([prev_lower, lower.to(prev_lower)], dim=0)
        self._parent_margins = torch.cat([prev_parent, parent.to(prev_parent)], dim=0)
        if self._node_id is not None:
            assert batch.node_id is not None
            self._node_id = torch.cat([self._node_id, batch.node_id.to(self._node_id.device)], dim=0)
        if self._parent_id is not None:
            assert batch.parent_id is not None
            self._parent_id = torch.cat([self._parent_id, batch.parent_id.to(self._parent_id.device)], dim=0)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        lb = self._lb
        if lb is None:
            raise IndexError("pop from empty pool")
        total = lb.shape[0]
        n = min(batch_size, total)
        if n >= total:
            selected = torch.arange(total, device=lb.device)
            remaining: Optional[torch.Tensor] = None
        else:
            order = torch.argsort(self._priority_scores(), descending=True)
            selected = order[:n]
            remaining = order[n:]

        result = self._build(selected)
        if remaining is None or remaining.numel() == 0:
            self._clear()
        else:
            self._restrict(remaining)
        return result

    def _priority_scores(self) -> torch.Tensor:
        depths_t, lb = self._depths, self._lower_bound
        assert depths_t is not None and lb is not None
        return self.order(depths_t, lb)

    def _build(self, idx: torch.Tensor) -> SubproblemBatch:
        lb, ub, depths = self._lb, self._ub, self._depths
        lower, parent = self._lower_bound, self._parent_margins
        assert lb is not None and ub is not None and depths is not None
        assert lower is not None and parent is not None
        idx = idx.to(lb.device)
        return SubproblemBatch(
            lb=lb.index_select(0, idx),
            ub=ub.index_select(0, idx),
            depths=depths.index_select(0, idx),
            incremental_alpha=_index_optional_dict(self._incremental_alpha, idx),
            incremental_eta=_index_optional_dict(self._incremental_eta, idx),
            split_signs=_index_optional_dict(self._split_signs, idx),
            parent_margins=parent.index_select(0, idx),
            lower_bound=lower.index_select(0, idx),
            node_id=(
                self._node_id.index_select(0, idx.to(self._node_id.device))
                if self._node_id is not None
                else None
            ),
            parent_id=(
                self._parent_id.index_select(0, idx.to(self._parent_id.device))
                if self._parent_id is not None
                else None
            ),
        )

    def _restrict(self, idx: torch.Tensor) -> None:
        kept = self._build(idx)
        self._lb, self._ub, self._depths = kept.lb, kept.ub, kept.depths
        self._lower_bound, self._parent_margins = kept.lower_bound, kept.parent_margins
        self._node_id, self._parent_id = kept.node_id, kept.parent_id
        self._incremental_alpha, self._incremental_eta, self._split_signs = (
            kept.incremental_alpha,
            kept.incremental_eta,
            kept.split_signs,
        )

    def _clear(self) -> None:
        self._lb = self._ub = self._depths = None
        self._lower_bound = self._parent_margins = None
        self._node_id = self._parent_id = None
        self._incremental_alpha = self._incremental_eta = self._split_signs = None

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        order = torch.argsort(self._priority_scores(), descending=True)
        self._restrict(order[:cap])
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]
