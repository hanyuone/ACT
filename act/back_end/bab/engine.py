# ===- act/back_end/bab/engine.py - BaB Verification Engine ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Main BaB loop that orchestrates branching, scheduling, solving,
#   and CE validation.  Solver-agnostic — any ``Solver`` backend works.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from act.back_end.bab.config import BaBConfig
from act.back_end.bab.node import SubproblemBatch, split_subproblems
from act.back_end.bab.branching.random_branching import BranchingStrategy
from act.back_end.bab.branching.random_scheduling import Scheduler
from act.back_end.bab.test_bab import check_violation_at_point

from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    seed_from_input_specs,
    setup_and_solve,
)
from act.util.stats import VerifyStatus, VerifyResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_branching_strategy(method: str) -> BranchingStrategy:
    if method == "random":
        from act.back_end.bab.branching.random_branching import RandomBranching

        return RandomBranching()
    raise ValueError(f"Unknown branching method: {method!r}")


def _build_scheduler(method: str) -> Scheduler:
    if method == "random":
        from act.back_end.bab.branching.random_scheduling import RandomScheduler

        return RandomScheduler()
    raise ValueError(f"Unknown scheduling method: {method!r}")


# ---------------------------------------------------------------------------
# BaB engine
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_bab(
    net: Net,
    solver: Solver,
    config: Optional[BaBConfig] = None,
    *,
    # Legacy kwargs (CLI backward compat)
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_subproblems: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    timelimit: Optional[float] = None,
    verbose: bool = False,
) -> VerifyResult:
    """Branch-and-Bound verification with pluggable strategies.

    Status mapping::

        Solver SAT + real CE     →  FALSIFIED  (stop)
        Solver SAT + spurious CE →  branch     (tighten relaxation)
        Solver UNKNOWN           →  branch     (inconclusive)
        Solver UNSAT             →  prune      (region certified)
        All pruned / exhausted   →  CERTIFIED
    """
    if config is None:
        config = BaBConfig(
            max_depth=max_depth if max_depth is not None else 20,
            max_nodes=(max_nodes or max_subproblems or 2000),
            time_budget_s=(time_budget_s or timelimit or 300.0),
            verbose=verbose,
        )

    brancher = _build_branching_strategy(config.branching_method)
    scheduler = _build_scheduler(config.scheduling_method)

    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    root_bounds = seed_from_input_specs(spec_layers)

    scheduler.push(SubproblemBatch.from_bounds(root_bounds))

    start = time.time()
    processed = 0

    while (
        not scheduler.empty
        and (time.time() - start) < config.time_budget_s
        and processed < config.max_nodes
    ):
        batch = scheduler.pop(batch_size=1)

        for bounds in batch.to_bounds_list():
            processed += 1
            status, ce_input, _ = setup_and_solve(net, bounds, solver, timelimit=None)

            if status == SolveStatus.UNSAT:
                continue

            if status == SolveStatus.SAT and ce_input is not None:
                if check_violation_at_point(net, ce_input, assert_layer):
                    return VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=torch.from_numpy(ce_input),
                        metadata={"nodes": processed},
                    )

            # UNKNOWN or spurious CE → branch if depth allows
            node_batch = SubproblemBatch(
                lb=bounds.lb.unsqueeze(0),
                ub=bounds.ub.unsqueeze(0),
                depths=batch.depths[:1],
            )

            if int(node_batch.depths[0].item()) >= config.max_depth:
                continue

            scores = brancher.compute_scores(node_batch, net)
            split_dims = brancher.select(scores)
            left, right = split_subproblems(node_batch, split_dims)
            scheduler.push(left)
            scheduler.push(right)

    return VerifyResult(VerifyStatus.CERTIFIED, metadata={"nodes": processed})
