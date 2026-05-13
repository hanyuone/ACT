# ===- act/back_end/bab/bab.py - BaB Verification Engine -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Main BaB loop that orchestrates branching, bounding, solving,
#   and CE validation.  Solver-agnostic — any ``Solver`` backend works.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from typing import Any, Optional

import numpy as np
import torch

from act.back_end.config import BaBConfig
from act.back_end.bab.node import BabNode, SubproblemBatch, split_subproblems
from act.back_end.bab.branching.branching import BranchingStrategy, RandomBranching
from act.back_end.bab.branching.bounding import BoundingStrategy, RandomBounding

from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    seed_from_input_specs,
    setup_and_solve,
)
from act.front_end.specs import OutKind
from act.util.model_inference import infer_single_model
from act.util.stats import VerifyStatus, VerifyResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CE validation (single-instance contract)
#
# BaB operates on single-instance subproblems by construction: each BaB
# node encodes one MILP for one (input box, ASSERT) pair, with existential
# encoding of y_true / margin. The `_bab_*` helpers below enforce B=1 at
# the API boundary so that batched callers cannot accidentally pipe
# [B, ...] tensors into BaB. ``verify_once`` itself stays batched; only
# the BaB MILP encoding is single-instance.
# ---------------------------------------------------------------------------


def _bab_scalar(val: Any, name: str) -> float:
    """Single-instance scalar extraction for BaB CE validation; mirrors
    ``verifier._b1_scalar``. Rejects multi-element tensors (B>1)."""
    if isinstance(val, torch.Tensor):
        if val.numel() != 1:
            raise ValueError(
                f"BaB CE validation requires B=1; '{name}' has "
                f"numel={val.numel()}, shape={tuple(val.shape)}"
            )
        return float(val.item())
    return float(val)


def _bab_int(val: Any, name: str) -> int:
    if isinstance(val, torch.Tensor):
        if val.numel() != 1:
            raise ValueError(
                f"BaB CE validation requires B=1; '{name}' has "
                f"numel={val.numel()}, shape={tuple(val.shape)}"
            )
        return int(val.item())
    return int(val)


def _bab_vec(val: Any, name: str, expected_len: int) -> torch.Tensor:
    t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
    if t.dim() == 2:
        if t.shape[0] != 1:
            raise ValueError(
                f"BaB CE validation requires B=1; '{name}' has "
                f"shape={tuple(t.shape)}"
            )
        t = t[0]
    if t.dim() != 1 or t.shape[0] != expected_len:
        raise ValueError(
            f"BaB CE validation: '{name}' must reduce to [{expected_len}]; "
            f"got shape={tuple(t.shape)}"
        )
    return t


def check_violation_at_point(net: Net, x: torch.Tensor, assert_layer) -> bool:
    x_batch = x.unsqueeze(0) if x.dim() == 1 else x
    success, output, _ = infer_single_model("ce_validate", net, x_batch)
    if not success:
        return False
    y = output.squeeze(0)
    n_out = y.shape[0]

    k = assert_layer.params.get("kind")

    if k == OutKind.TOP1_ROBUST:
        t = _bab_int(assert_layer.params["y_true"], "y_true")
        mask = torch.ones(n_out, dtype=torch.bool)
        mask[t] = False
        return (y[mask] - y[t]).max().item() >= 0.0

    if k == OutKind.MARGIN_ROBUST:
        t = _bab_int(assert_layer.params["y_true"], "y_true")
        margin = _bab_scalar(assert_layer.params["margin"], "margin")
        mask = torch.ones(n_out, dtype=torch.bool)
        mask[t] = False
        return (y[mask] - y[t]).max().item() >= margin

    if k == OutKind.LINEAR_LE:
        c = _bab_vec(assert_layer.params["c"], "c", n_out).to(y.dtype)
        d = _bab_scalar(assert_layer.params["d"], "d")
        return (c @ y).item() >= d + 1e-8

    if k == OutKind.RANGE:
        lb_raw = assert_layer.params.get("lb")
        ub_raw = assert_layer.params.get("ub")
        if lb_raw is not None:
            lb_t = _bab_vec(lb_raw, "lb", n_out).to(y.dtype)
            if (y < lb_t - 1e-8).any():
                return True
        if ub_raw is not None:
            ub_t = _bab_vec(ub_raw, "ub", n_out).to(y.dtype)
            if (y > ub_t + 1e-8).any():
                return True
        return False

    if k == OutKind.UNSAFE_LINEAR:
        c_raw = assert_layer.params["c"]
        C = c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)
        if C.dim() == 3:
            if C.shape[0] != 1:
                raise ValueError(
                    f"BaB CE validation requires B=1; UNSAFE_LINEAR c has "
                    f"shape={tuple(C.shape)}"
                )
            C = C[0]
        if C.dim() == 1:
            C = C.unsqueeze(0)
        d_raw = assert_layer.params["d"]
        d_t = d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)
        if d_t.dim() == 2:
            if d_t.shape[0] != 1:
                raise ValueError(
                    f"BaB CE validation requires B=1; UNSAFE_LINEAR d has "
                    f"shape={tuple(d_t.shape)}"
                )
            d_t = d_t[0]
        d_vec = d_t.reshape(-1).to(y.dtype)
        C = C.to(y.dtype)
        Cy = C @ y.reshape(-1)
        in_unsafe_region = (Cy <= d_vec + 1e-8).all()
        return bool(in_unsafe_region.item())

    raise NotImplementedError(f"ASSERT kind not supported: {k}")


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_branching_strategy(method: str) -> BranchingStrategy:
    if method == "random":
        return RandomBranching()
    raise ValueError(f"Unknown branching method: {method!r}")


def _build_bounding(method: str) -> BoundingStrategy:
    if method == "random":
        return RandomBounding()
    raise ValueError(f"Unknown bounding method: {method!r}")


# ---------------------------------------------------------------------------
# BaB engine
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_bab(
    net: Net,
    solver: Solver,
    config: Optional[BaBConfig] = None,
    *,
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
            verbose=verbose,
        )

    budget = time_budget_s or timelimit or 300.0

    brancher = _build_branching_strategy(config.branching_method)
    pool = _build_bounding(config.bounding_method)

    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    root_bounds = seed_from_input_specs(spec_layers)

    pool.push(SubproblemBatch.from_bounds(root_bounds))

    start = time.time()
    processed = 0

    while (
        not pool.empty
        and (time.time() - start) < budget
        and processed < config.max_nodes
    ):
        batch = pool.pop(batch_size=1)

        for bounds in batch.to_bounds_list():
            processed += 1
            # Fresh solver per subproblem: setup_and_solve mutates solver state
            # (adds constraints/vars), and that state must not leak across BaB
            # iterations. Reconstruct from the original type to preserve class
            # configuration while resetting accumulated state.
            iter_solver = type(solver)()
            status, ce_input, _ = setup_and_solve(net, bounds, iter_solver, timelimit=None)

            if status == SolveStatus.UNSAT:
                continue

            if status == SolveStatus.SAT and ce_input is not None:
                ce_tensor = torch.from_numpy(ce_input).to(device=root_bounds.lb.device)
                if check_violation_at_point(net, ce_tensor, assert_layer):
                    return VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=ce_tensor,
                        metadata={"nodes": processed},
                    )

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
            pool.push(left)
            pool.push(right)

    return VerifyResult(VerifyStatus.CERTIFIED, metadata={"nodes": processed})


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


class _StubNet:
    layers = []


def test_imports():
    for sym in (
        verify_bab,
        BaBConfig,
        BabNode,
        SubproblemBatch,
        split_subproblems,
        check_violation_at_point,
        BranchingStrategy,
        BoundingStrategy,
        RandomBranching,
        RandomBounding,
    ):
        assert sym is not None


def test_config_yaml_roundtrip():
    c1 = BaBConfig()
    assert c1.max_depth == 20

    c2 = BaBConfig.from_yaml()
    assert c2.branching_method == "random"

    c3 = BaBConfig.from_yaml(max_depth=50, branching_method="kfsb")
    assert c3.max_depth == 50 and c3.branching_method == "kfsb"

    # Round-trip through a standalone BaB YAML (uses top-level "bab" key)
    tmp = tempfile.mktemp(suffix=".yaml")
    try:
        c3.to_yaml(tmp)
        c4 = BaBConfig.from_yaml(tmp)
        assert c4.max_depth == 50
        assert c4.branching_method == "kfsb"
    finally:
        os.unlink(tmp)

    # BaBConfig must not expose a time_budget_s attribute.
    assert not hasattr(c1, "time_budget_s")


def test_subproblem_batch():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))

    assert batch.batch_size == 1
    assert batch.input_dim == 3
    assert batch.total_width().item() == 12.0

    bounds = Bounds(lb.squeeze(0), ub.squeeze(0))
    batch2 = SubproblemBatch.from_bounds(bounds)
    assert torch.equal(batch2.lb, lb)

    back = batch2.to_bounds_list()
    assert len(back) == 1
    assert torch.equal(back[0].lb, bounds.lb)


def test_split_subproblems():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    split_dim = torch.tensor([1])

    left, right = split_subproblems(batch, split_dim)

    mid = (lb[0, 1] + ub[0, 1]) / 2
    assert torch.isclose(left.ub[0, 1], mid)
    assert torch.isclose(right.lb[0, 1], mid)
    assert left.depths[0] == 1
    assert right.depths[0] == 1

    assert torch.equal(left.lb[0, 0], lb[0, 0])
    assert torch.equal(right.ub[0, 2], ub[0, 2])


def test_random_branching():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))

    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, _StubNet())
    assert scores.shape == (1, 3)
    assert (scores >= 0).all()

    dims = brancher.select(scores)
    assert dims.shape == (1,)
    assert 0 <= dims.item() <= 2


def test_random_branching_with_mask():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    mask = torch.tensor([False, True, False])

    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, _StubNet(), unstable_mask=mask)
    assert scores[0, 0].item() == 0.0
    assert scores[0, 2].item() == 0.0
    assert brancher.select(scores).item() == 1


def test_random_bounding():
    lb = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    ub = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0, 1]))

    pool = RandomBounding()
    assert pool.empty

    pool.push(batch)
    assert len(pool) == 2

    popped = pool.pop(1)
    assert popped.batch_size == 1
    assert len(pool) == 1

    pool.pop(1)
    assert pool.empty


def test_babnode_compat():
    bounds = Bounds(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
    node = BabNode(box=bounds, depth=3, score=0.5)
    batch = node.to_batch()
    assert batch.batch_size == 1
    assert batch.depths[0].item() == 3


_TESTS = [
    test_imports,
    test_config_yaml_roundtrip,
    test_subproblem_batch,
    test_split_subproblems,
    test_random_branching,
    test_random_branching_with_mask,
    test_random_bounding,
    test_babnode_compat,
]


def run_all_tests() -> int:
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("Running BaB module tests\n")
    sys.exit(run_all_tests())
