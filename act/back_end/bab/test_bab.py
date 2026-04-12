# ===- act/back_end/bab/test_bab.py - Validation & BaB Tests -------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Counterexample validation for BaB + module-level tests.
#
#   When the solver returns SAT with a candidate CE, the CE may be
#   *spurious* — an artefact of the linear relaxation gap.
#   ``check_violation_at_point`` evaluates the ACT Net at the candidate
#   point to decide whether the property is truly violated.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.front_end.specs import OutKind


# ---------------------------------------------------------------------------
# CE validation (was validation.py)
# ---------------------------------------------------------------------------


def check_violation_at_point(net, x_np: np.ndarray, assert_layer) -> bool:
    """Evaluate the ACT Net at a single point and check property violation.

    Uses ``analyze()`` with a tight point-box (lb == ub) so the abstract
    interpretation degenerates to concrete evaluation.

    Args:
        net:           ACT ``Net`` instance.
        x_np:          Candidate counterexample as numpy array.
        assert_layer:  The ``ASSERT`` layer of the net.

    Returns:
        ``True`` if the property is *violated* at ``x_np``
        (i.e. the CE is genuine), ``False`` otherwise (spurious).
    """
    from act.back_end.analyze import analyze
    from act.back_end.verifier import find_entry_layer_id

    x_tensor = torch.from_numpy(x_np)
    point_bounds = Bounds(x_tensor, x_tensor)

    entry_fact = Fact(bounds=point_bounds, cons=ConSet())
    entry_id = find_entry_layer_id(net)
    _, after, _ = analyze(net, entry_id, entry_fact)

    output_layer_id = net.layers[-2].id
    y_bounds = after[output_layer_id].bounds
    y_mid = ((y_bounds.lb + y_bounds.ub) / 2).cpu().numpy()

    k = assert_layer.params.get("kind")

    if k == OutKind.TOP1_ROBUST:
        t = int(assert_layer.params["y_true"])
        others = [i for i in range(len(y_mid)) if i != t]
        return float((y_mid[others] - y_mid[t]).max()) >= 0.0

    if k == OutKind.MARGIN_ROBUST:
        t = int(assert_layer.params["y_true"])
        margin = float(assert_layer.params["margin"])
        others = [i for i in range(len(y_mid)) if i != t]
        return float((y_mid[others] - y_mid[t]).max()) >= margin

    if k == OutKind.LINEAR_LE:
        c = np.asarray(assert_layer.params["c"], dtype=float)
        d = float(assert_layer.params["d"])
        return float(np.dot(c, y_mid)) >= d + 1e-8

    if k == OutKind.RANGE:
        lb = assert_layer.params.get("lb")
        ub = assert_layer.params.get("ub")
        if lb is not None and np.any(y_mid < np.asarray(lb, dtype=float) - 1e-8):
            return True
        if ub is not None and np.any(y_mid > np.asarray(ub, dtype=float) + 1e-8):
            return True
        return False

    raise NotImplementedError(f"ASSERT kind not supported: {k}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from act.back_end.bab.config import BaBConfig
from act.back_end.bab.node import BabNode, SubproblemBatch, split_subproblems
from act.back_end.bab.branching.random_branching import (
    BranchingStrategy,
    RandomBranching,
)
from act.back_end.bab.branching.random_scheduling import Scheduler, RandomScheduler


class _StubNet:
    layers = []


def test_imports():
    from act.back_end.bab.engine import verify_bab

    for sym in (
        verify_bab,
        BaBConfig,
        BabNode,
        SubproblemBatch,
        split_subproblems,
        check_violation_at_point,
        BranchingStrategy,
        Scheduler,
        RandomBranching,
        RandomScheduler,
    ):
        assert sym is not None


def test_config_yaml_roundtrip():
    c1 = BaBConfig()
    assert c1.max_depth == 20

    c2 = BaBConfig.from_yaml()
    assert c2.branching_method == "random"

    c3 = BaBConfig.from_yaml(max_depth=50, branching_method="kfsb")
    assert c3.max_depth == 50 and c3.branching_method == "kfsb"

    tmp = tempfile.mktemp(suffix=".yaml")
    try:
        c3.to_yaml(tmp)
        c4 = BaBConfig.from_yaml(tmp)
        assert c4.max_depth == 50
        assert c4.branching_method == "kfsb"
    finally:
        os.unlink(tmp)


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


def test_random_scheduler():
    lb = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    ub = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0, 1]))

    sched = RandomScheduler()
    assert sched.empty

    sched.push(batch)
    assert len(sched) == 2

    popped = sched.pop(1)
    assert popped.batch_size == 1
    assert len(sched) == 1

    sched.pop(1)
    assert sched.empty


def test_babnode_compat():
    bounds = Bounds(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
    node = BabNode(box=bounds, depth=3, score=0.5)
    batch = node.to_batch()
    assert batch.batch_size == 1
    assert batch.depths[0].item() == 3


# ---------------------------------------------------------------------------

_TESTS = [
    test_imports,
    test_config_yaml_roundtrip,
    test_subproblem_batch,
    test_split_subproblems,
    test_random_branching,
    test_random_branching_with_mask,
    test_random_scheduler,
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


def main():
    print("Running BaB module tests\n")
    sys.exit(run_all_tests())


if __name__ == "__main__":
    main()
