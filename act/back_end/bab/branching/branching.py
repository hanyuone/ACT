# ===- act/back_end/bab/branching/branching.py - Branching ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Branching strategies for Branch-and-Bound.
#
#   A branching strategy decides *which dimension (or neuron) to split* for each
#   subproblem in a batch. Two paradigms are supported:
#     1. Input splitting  — bisect (or N-ary section) the input domain along a dim.
#     2. Neuron splitting  — fix an unstable ReLU's activation phase (on/off).
#
#   Strategies (subclasses of ``BranchingStrategy``):
#     * ``RandomBranching`` — uniform-random over eligible dims (width-weighted for
#       input splits; masked to unstable neurons for neuron splits). Baseline.
#     * ``BaBSRBranching`` — BaBSR score (|slope-gradient|·width) over candidate
#       neurons from the dual ν/bounds; width-based fallback when α-gradients are
#       unavailable.
#     * ``FSBBranching`` — Filtered Smart Branching; extends BaBSR by re-scoring the
#       top candidates with the dual solver and picking the best bound improvement.
#
#   Result types: ``BranchingScores`` (per-dim / per-neuron scores) and
#   ``SplitDecision`` (``input_axis`` / ``cut_dim`` + ``fanout`` for input splits;
#   ``layer_id`` / ``neuron_idx`` for neuron splits). Factory:
#   ``_build_branching_strategy``.
#
#   All tensor shapes follow the (N, D) batch convention for batch-parallel scoring.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds, Net


# ---------------------------------------------------------------------------
# Branching result types
# ---------------------------------------------------------------------------


@dataclass
class BranchingScores:
    flat: Optional[torch.Tensor] = None
    per_layer: Optional[Dict[int, torch.Tensor]] = None
    intercept_per_layer: Optional[Dict[int, torch.Tensor]] = None


@dataclass
class SplitDecision:
    kind: str
    input_axis: Optional[torch.Tensor | int] = None
    cut_dim: Optional[torch.Tensor] = None
    fanout: int = 2
    layer_id: Optional[torch.Tensor] = None
    neuron_idx: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BranchingStrategy(ABC):
    """Abstract branching strategy for Branch-and-Bound.

    Lifecycle (called by the BaB engine per iteration)::

        scores     = strategy.compute_scores(batch, net, unstable_mask)
        split_dims = strategy.select(scores)
        left, right = split_subproblems(batch, split_dims)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``compute_scores`` **must** return ``(N, D)`` float tensor.
    * Dimensions that must not be split should receive score ``-inf``
      (or ``0`` when ``select`` uses ``argmax``).
    * ``select`` defaults to row-wise ``argmax``; override for
      stochastic or top-k selection.
    """

    @abstractmethod
    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | BranchingScores:
        """Score every candidate split dimension.

        Args:
            batch:          Current subproblems ``(N, D)``.
            net:            ACT network (for layer structure / neuron info).
            unstable_mask:  ``(D,)`` or ``(N, D)`` bool tensor marking
                            neurons eligible for splitting.
                            ``None`` ⇒ all dimensions are candidates
                            (input-split mode).

        Returns:
            ``(N, D)`` float tensor — higher score = better split.
        """
        ...

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        """Pick one split dimension per subproblem.

        Default implementation: deterministic ``argmax`` per row.
        Override for stochastic or multi-split selection.

        Args:
            scores: ``(N, D)`` branching scores.

        Returns:
            ``(N,)`` long tensor of selected dimension indices.
        """
        if isinstance(scores, BranchingScores):
            if scores.flat is None:
                raise ValueError("Base BranchingStrategy requires flat scores")
            scores = scores.flat
        return scores.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBranching(BranchingStrategy):
    """Uniform-random branching over eligible dimensions.

    Supports both paradigms:

    * **Input split** (``unstable_mask is None``):
      Random scores weighted by domain width — wider dimensions are more
      likely to be chosen, and zero-width dimensions are excluded.

    * **Neuron split** (``unstable_mask`` provided):
      Uniform-random scores masked to unstable neurons.  Stable neurons
      receive score 0 and are never selected.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, D = batch.batch_size, batch.input_dim
        device = batch.lb.device

        scores = torch.rand(N, D)

        if unstable_mask is not None:
            # Neuron-split mode: zero-out stable neurons
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(N, -1)  # (D,) → (N, D)
            scores = scores * mask
        else:
            # Input-split mode: weight by width (zero-width → score 0)
            widths = batch.widths()  # (N, D)
            scores = scores * (widths > 0).float()

        return scores


# ---------------------------------------------------------------------------
# Score-based branching (width-weighted, optional slope-gradient scoring)
# ---------------------------------------------------------------------------


class BaBSRBranching(BranchingStrategy):
    """BaBSR neuron-split scoring (Bunel et al., 1909.06588).

    For each ambiguous ReLU (``l<0<u``) with backward coefficient ν and pre-activation
    bias ``b``, scores the estimated bound improvement of splitting it as
    ``|bias_term + intercept_term|`` where ``s = u/(u-l)``,
    ``bias_term = max(b·ν·(s-1), b·ν·s)`` and ``intercept_term = clamp(ν, max=0)·(-l)·u/(u-l)``.
    An intercept-only score (``(-l)·u/(u-l)·clamp(-ν, 0)``) is kept as the backup.
    When ν / intermediate bounds are unavailable it falls back to width-based scores
    compatible with ``RandomBranching``.
    """

    def __init__(
        self,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.intercept_fallback_max = intercept_fallback_max
        self.sparsest_layer = sparsest_layer
        self.icp_score_counter = 0

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        if bounds_dict is None or nu_per_layer is None:
            return BranchingScores(flat=self._baseline_scores(batch, unstable_mask))

        per_layer: Dict[int, torch.Tensor] = {}
        intercept_per_layer: Dict[int, torch.Tensor] = {}
        for lid, bounds in bounds_dict.items():
            if self.sparsest_layer is not None and lid != self.sparsest_layer:
                continue
            if lid not in nu_per_layer:
                continue
            lb = bounds.lb.flatten(start_dim=1)
            ub = bounds.ub.flatten(start_dim=1)
            N, n_neurons = lb.shape
            ambiguous = (lb < 0) & (ub > 0)
            nu = nu_per_layer[lid]
            if nu.dim() > 2:
                nu = nu.flatten(start_dim=1)
            if nu.shape[0] == N:
                nu_view = nu.reshape(N, 1, n_neurons)
            elif nu.shape[0] % N == 0:
                nu_view = nu.reshape(N, nu.shape[0] // N, n_neurons)
            else:
                continue

            split_mask = self._already_split_mask(batch, lid, n_neurons)
            effective = ambiguous & ~split_mask
            denom = torch.clamp(ub - lb, min=torch.finfo(lb.dtype).eps)
            slope = (ub / denom).unsqueeze(1)
            relax_intercept = ((-lb) * ub / denom).unsqueeze(1)
            preact_bias = _preact_bias_of(net, lid)
            if preact_bias.numel() != n_neurons:
                preact_bias = torch.zeros(n_neurons, dtype=lb.dtype, device=lb.device)
            preact_bias = preact_bias.reshape(1, 1, -1).to(device=nu_view.device, dtype=nu_view.dtype)
            bias_term = torch.maximum(
                preact_bias * nu_view * (slope - 1.0),
                preact_bias * nu_view * slope,
            )
            intercept_term = nu_view.clamp(max=0.0) * relax_intercept
            primary = (bias_term + intercept_term).abs().mean(dim=1)
            primary = primary.masked_fill(~effective, float("-inf"))
            intercept = ((-lb) * ub / denom) * nu_view.mean(dim=1).neg().clamp(min=0.0)
            intercept = intercept.masked_fill(~effective, float("-inf"))
            per_layer[lid] = primary
            intercept_per_layer[lid] = intercept

        if not per_layer:
            return BranchingScores(flat=self._baseline_scores(batch, unstable_mask))
        return BranchingScores(
            flat=None,
            per_layer=per_layer,
            intercept_per_layer=intercept_per_layer,
        )

    def _baseline_scores(
        self,
        batch: SubproblemBatch,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        widths = batch.ub - batch.lb

        if batch.incremental_alpha is None:
            scores = widths * torch.rand_like(widths)
        else:
            scores = widths

        if unstable_mask is not None:
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch.batch_size, -1)
            scores = scores * mask

        return scores

    def _already_split_mask(
        self,
        batch: SubproblemBatch,
        lid: int,
        n_neurons: int,
    ) -> torch.Tensor:
        if batch.split_signs is None or lid not in batch.split_signs:
            return torch.zeros(
                (batch.batch_size, n_neurons),
                device=batch.lb.device,
                dtype=torch.bool,
            )
        signs = batch.split_signs[lid]
        return (signs != 0).any(dim=1)

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        if isinstance(scores, torch.Tensor):
            return scores.argmax(dim=-1)
        if scores.flat is not None:
            return SplitDecision(kind="input_axis", input_axis=scores.flat.argmax(dim=-1))
        per_layer = scores.per_layer
        if not per_layer:
            return SplitDecision(kind="input_axis", input_axis=0)

        N = next(iter(per_layer.values())).shape[0]
        device = next(iter(per_layer.values())).device
        decisions_layer = torch.zeros(N, dtype=torch.long, device=device)
        decisions_neuron = torch.zeros(N, dtype=torch.long, device=device)

        for n in range(N):
            best_lid: Optional[int] = None
            best_idx = 0
            best_val = float("-inf")
            for lid, score in per_layer.items():
                val, idx = score[n].max(dim=0)
                if float(val.item()) > best_val:
                    best_val = float(val.item())
                    best_lid = lid
                    best_idx = int(idx.item())

            if best_lid is not None and best_val > self.decision_threshold:
                decisions_layer[n] = best_lid
                decisions_neuron[n] = best_idx
                self.icp_score_counter = 0
                continue

            intercept = scores.intercept_per_layer
            if intercept is not None and self.icp_score_counter < self.intercept_fallback_max:
                ic_lid: Optional[int] = None
                ic_idx = 0
                ic_val = float("-inf")
                for lid, score in intercept.items():
                    val, idx = score[n].max(dim=0)
                    if float(val.item()) > ic_val:
                        ic_val = float(val.item())
                        ic_lid = lid
                        ic_idx = int(idx.item())
                if ic_lid is not None and ic_val > float("-inf"):
                    decisions_layer[n] = ic_lid
                    decisions_neuron[n] = ic_idx
                    self.icp_score_counter += 1
                    continue

            self.icp_score_counter = 0
            return SplitDecision(kind="input_axis", input_axis=0)

        return SplitDecision(kind="neuron", layer_id=decisions_layer, neuron_idx=decisions_neuron)


def _preact_bias_of(net: Net, lid: int) -> torch.Tensor:
    layer = net.by_id[lid]
    n_neurons = len(layer.out_vars)
    preds = net.preds.get(lid, [])
    if preds:
        pred = net.by_id[preds[0]]
        bias = pred.params.get("bias")
        if isinstance(bias, torch.Tensor):
            return bias.detach().clone()
        for value in pred.params.values():
            if isinstance(value, torch.Tensor):
                return torch.zeros(n_neurons, dtype=value.dtype, device=value.device)
    dtype = torch.float32
    device = torch.device("cpu")
    for value in layer.params.values():
        if isinstance(value, torch.Tensor):
            dtype = value.dtype
            device = value.device
            break
    return torch.zeros(n_neurons, dtype=dtype, device=device)


class FSBBranching(BaBSRBranching):
    def __init__(
        self,
        dual_solver: Any,
        branching_candidates: int = 3,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
    ) -> None:
        super().__init__(
            decision_threshold=decision_threshold,
            intercept_fallback_max=intercept_fallback_max,
            sparsest_layer=sparsest_layer,
        )
        self.dual_solver = dual_solver
        self.branching_candidates = branching_candidates

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        bsr = super().compute_scores(
            batch,
            net,
            unstable_mask,
            bounds_dict=bounds_dict,
            nu_per_layer=nu_per_layer,
        )
        if bsr.per_layer is None or not bsr.per_layer:
            return bsr

        hypothesis_list: List[Dict[int, torch.Tensor]] = []
        candidate_metadata: List[Tuple[int, torch.Tensor]] = []
        topk = self.branching_candidates
        for lid, score in bsr.per_layer.items():
            k = min(topk, score.shape[-1])
            if k > 0:
                top_values, top_idx = score.topk(k, dim=-1)
                for idx in range(top_idx.shape[-1]):
                    if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                        continue
                    neuron_per_lane = top_idx[:, idx]
                    hypothesis_list.append(
                        self._clone_split_signs_with_hypothesis(batch, lid, neuron_per_lane, net)
                    )
                    candidate_metadata.append((lid, neuron_per_lane))
        if bsr.intercept_per_layer is not None:
            for lid, score in bsr.intercept_per_layer.items():
                k = min(topk, score.shape[-1])
                if k > 0:
                    top_values, top_idx = score.topk(k, dim=-1)
                    for idx in range(top_idx.shape[-1]):
                        if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                            continue
                        neuron_per_lane = top_idx[:, idx]
                        hypothesis_list.append(
                            self._clone_split_signs_with_hypothesis(batch, lid, neuron_per_lane, net)
                        )
                        candidate_metadata.append((lid, neuron_per_lane))

        if not hypothesis_list:
            return bsr

        N = batch.batch_size
        baseline = batch.parent_margins
        try:
            result = self._evaluate_hypotheses(net, bounds_dict, hypothesis_list)
            stacked_margins = result.margins.to(device=batch.lb.device, dtype=batch.lb.dtype)
            if stacked_margins.dim() == 1:
                stacked_margins = stacked_margins.reshape(1, -1)
            if stacked_margins.dim() > 2:
                stacked_margins = stacked_margins.reshape(stacked_margins.shape[0], N, -1).mean(dim=-1)
            if stacked_margins.shape[0] != len(hypothesis_list):
                raise ValueError("KFSB solver returned wrong candidate dimension")
            if stacked_margins.shape[1] != N:
                stacked_margins = stacked_margins.reshape(len(hypothesis_list), N, -1).mean(dim=-1)
            improvements = (
                stacked_margins - baseline.to(stacked_margins.device).unsqueeze(0)
                if baseline is not None
                else stacked_margins
            )
        except Exception:
            improvements = self._evaluate_hypotheses_serial(
                net,
                bounds_dict,
                hypothesis_list,
                baseline,
                N,
            )

        if bool(torch.isneginf(improvements).all().item()):
            return bsr

        best_cand_idx = improvements.argmax(dim=0)
        final_per_layer: Dict[int, torch.Tensor] = {}
        for lane in range(N):
            ci = int(best_cand_idx[lane].item())
            lid, neuron_tensor = candidate_metadata[ci]
            neuron_idx = int(neuron_tensor[lane].item())
            source = bsr.per_layer.get(lid) if bsr.per_layer is not None else None
            if source is None and bsr.intercept_per_layer is not None:
                source = bsr.intercept_per_layer.get(lid)
            if source is None:
                continue
            if lid not in final_per_layer:
                final_per_layer[lid] = torch.full_like(source, float("-inf"))
            final_per_layer[lid][lane, neuron_idx] = improvements[ci, lane]

        return BranchingScores(flat=None, per_layer=final_per_layer, intercept_per_layer=bsr.intercept_per_layer)

    def _evaluate_hypotheses(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> Any:
        if bounds_dict is None:
            raise ValueError("FSB hypothesis evaluation requires bounds_dict")
        c = self._zero_objective(net, bounds_dict)
        return self.dual_solver.compute_certified_bound(
            net,
            bounds_dict,
            c,
            M=1,
            split_signs=split_signs,
            optimize=False,
        )

    def _evaluate_hypotheses_serial(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        hypothesis_list: List[Dict[int, torch.Tensor]],
        baseline: Optional[torch.Tensor],
        N: int,
    ) -> torch.Tensor:
        improvements = torch.full(
            (len(hypothesis_list), N),
            float("-inf"),
            device=next(iter(hypothesis_list[0].values())).device,
            dtype=next(iter(hypothesis_list[0].values())).dtype,
        )
        for c_idx, hypo in enumerate(hypothesis_list):
            try:
                result = self._evaluate_hypothesis(net, bounds_dict, hypo)
                margins = result.margins.reshape(N, -1).mean(dim=-1).to(improvements)
                improvements[c_idx] = margins - baseline.to(margins.device) if baseline is not None else margins
            except Exception:
                # Recoverable: an individual hypothetical split may be incompatible with a legacy solver/mock.
                continue
        return improvements

    def _evaluate_hypothesis(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: Dict[int, torch.Tensor],
    ) -> Any:
        if bounds_dict is None:
            raise ValueError("FSB hypothesis evaluation requires bounds_dict")
        c = self._zero_objective(net, bounds_dict)
        return self.dual_solver.compute_certified_bound(
            net,
            bounds_dict,
            c,
            M=1,
            split_signs=split_signs,
            optimize=False,
        )

    def _zero_objective(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
    ) -> torch.Tensor:
        assert_layers = [layer for layer in net.layers if layer.kind == "ASSERT"]
        if assert_layers:
            assert_layer = assert_layers[-1]
            preds = net.preds.get(assert_layer.id, [])
            n_out = len(net.by_id[preds[0]].out_vars) if preds else 1
        else:
            n_out = 1
        sample = next(iter(bounds_dict.values()))
        B = sample.lb.shape[0]
        return torch.zeros((B, n_out), dtype=sample.lb.dtype, device=sample.lb.device)

    def _clone_split_signs_with_hypothesis(
        self,
        batch: SubproblemBatch,
        lid: int,
        neuron_idx_per_lane: torch.Tensor,
        net: Net,
    ) -> Dict[int, torch.Tensor]:
        hypo: Dict[int, torch.Tensor] = {}
        if batch.split_signs is not None:
            for key, value in batch.split_signs.items():
                hypo[key] = value.clone()

        n_neurons = len(net.by_id[lid].out_vars)
        n_specs = 1
        if batch.incremental_alpha is not None and lid in batch.incremental_alpha:
            n_specs = int(batch.incremental_alpha[lid].shape[1])
        elif batch.incremental_eta is not None and lid in batch.incremental_eta:
            n_specs = int(batch.incremental_eta[lid].shape[1])

        if lid not in hypo:
            hypo[lid] = torch.zeros(
                (batch.batch_size, n_specs, n_neurons),
                device=batch.lb.device,
                dtype=batch.lb.dtype,
            )
        for lane in range(batch.batch_size):
            idx = int(neuron_idx_per_lane[lane].item())
            hypo[lid][lane, :, idx] = +1.0
        return hypo


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def _build_branching_strategy(
    method: str,
    *,
    dual_solver: Any = None,
    branching_candidates: int = 3,
) -> BranchingStrategy:
    if method == "random":
        return RandomBranching()
    if method == "babsr":
        return BaBSRBranching()
    if method == "fsb":
        if dual_solver is None:
            raise ValueError("FSB branching requires a dual_solver instance (inject via factory).")
        return FSBBranching(dual_solver=dual_solver, branching_candidates=branching_candidates)
    raise ValueError(f"Unknown branching method: {method!r}")
