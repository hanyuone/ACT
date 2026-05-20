# ===- act/back_end/bab/bab.py - BaB Verification Engine -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   BaB loop on a single-spec instance.  Subproblems explored in K-batched
#   waves via solve_batch; CE validation per SAT lane.  Solver-agnostic.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from typing import Callable, List, Optional, cast

import torch

from act.back_end.config import BaBConfig
from act.back_end.bab.node import BabNode, SubproblemBatch, split_subproblems
from act.back_end.bab.branching.branching import BranchingStrategy, RandomBranching
from act.back_end.bab.branching.bounding import BoundingStrategy, RandomBounding

from act.back_end.core import Bounds, Layer, Net, ParamValue
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
    setup_and_solve_batch,
)
from act.front_end.specs import OutKind
from act.util.model_inference import infer_single_model
from act.util.stats import VerifyStatus, VerifyResult

log = logging.getLogger(__name__)



def _as_batched_vector(
    value: object,
    n_batch: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=dtype)
    if t.dim() == 0:
        return t.expand(n_batch, width).contiguous()
    if t.dim() == 1:
        if t.numel() == width:
            return t.unsqueeze(0).expand(n_batch, -1).contiguous()
        if width == 1 and t.numel() == n_batch:
            return t.reshape(n_batch, 1).contiguous()
    if t.dim() == 2:
        if t.shape == (1, width):
            return t.expand(n_batch, -1).contiguous()
        if t.shape == (n_batch, width):
            return t.contiguous()
    raise ValueError(
        f"{name}: expected scalar, ({width},), (1,{width}), or "
        f"({n_batch},{width}); got {tuple(t.shape)}"
    )


def _as_batched_index(
    value: object,
    n_batch: int,
    n_out: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=torch.long).reshape(-1)
    if t.numel() == 1:
        t = t.expand(n_batch)
    if t.numel() != n_batch:
        raise ValueError(
            f"{name}: expected 1 or {n_batch} indices, got {t.numel()}"
        )
    if bool(((t < 0) | (t >= n_out)).any().item()):
        raise ValueError(f"{name}: index out of range for n_out={n_out}: {t.tolist()}")
    return t.contiguous()


# Per-net cache of reconstructed PyTorch nn.Module used for CE validation.
# Without this, every check_violations_batched call rebuilds the module from
# scratch via ACTToTorch.run() — costly under K-batched BaB which can invoke
# CE validation dozens of times for a single net.  Cleared per top-level
# verify-all dispatch via clear_violation_check_module_cache().
_VIOLATION_CHECK_MODULE_CACHE: dict[int, torch.nn.Module] = {}


def clear_violation_check_module_cache() -> None:
    _VIOLATION_CHECK_MODULE_CACHE.clear()


def _forward_for_violation_check(net: object, x_batch: torch.Tensor) -> torch.Tensor:
    if isinstance(net, torch.nn.Module):
        module = net
    else:
        key = id(net)
        cached = _VIOLATION_CHECK_MODULE_CACHE.get(key)
        if cached is None:
            from act.pipeline.verification.act2torch import ACTToTorch

            cached = ACTToTorch(cast(Net, net)).run()
            _VIOLATION_CHECK_MODULE_CACHE[key] = cached
        module = cached
    _ = module.eval()
    try:
        target_dtype = next(module.parameters()).dtype
    except StopIteration:
        target_dtype = x_batch.dtype
    if x_batch.dtype != target_dtype:
        x_batch = x_batch.to(dtype=target_dtype)
    success, output, error = infer_single_model("ce_validate_batched", module, x_batch)
    if not success or output is None:
        raise RuntimeError(f"check_violations_batched: model forward failed: {error}")
    if output.dim() < 2:
        raise ValueError(
            f"check_violations_batched: model output must be batched, got "
            f"shape={tuple(output.shape)}"
        )
    return output.reshape(output.shape[0], -1)


@torch.no_grad()
def check_violations_batched(net: object, x_batch: torch.Tensor, assert_layer: Layer) -> torch.Tensor:
    """[BATCHED-API] Return a ``[N]`` bool tensor for concrete ASSERT violations.

    ``x_batch`` is always treated as a tensor-view batch ``[N, *input_shape]``;
    N=1 is represented by a length-one leading dimension. ASSERT parameters are
    read directly from ``assert_layer.params`` in their batch-native form.
    """
    if x_batch.dim() < 2:
        raise ValueError(
            f"check_violations_batched: x_batch must be [N, *input_shape], "
            f"got shape={tuple(x_batch.shape)}"
        )
    y_batch = _forward_for_violation_check(net, x_batch)
    n_batch = int(x_batch.shape[0])
    if int(y_batch.shape[0]) != n_batch:
        raise ValueError(
            f"check_violations_batched: output batch {int(y_batch.shape[0])} "
            f"!= input batch {n_batch}"
        )
    n_out = int(y_batch.shape[1])
    device = y_batch.device
    dtype = y_batch.dtype
    params = assert_layer.params
    kind = params.get("kind")
    eps = 1e-8

    if kind == OutKind.TOP1_ROBUST:
        y_true = _as_batched_index(params["y_true"], n_batch, n_out, device, "y_true")
        y_true_scores = y_batch.gather(1, y_true.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(y_batch, dtype=torch.bool)
        _ = mask.scatter_(1, y_true.unsqueeze(1), False)
        other_scores = y_batch.masked_fill(~mask, -float("inf"))
        return (other_scores.max(dim=1).values - y_true_scores) >= 0

    if kind == OutKind.MARGIN_ROBUST:
        y_true = _as_batched_index(params["y_true"], n_batch, n_out, device, "y_true")
        margin = _as_batched_vector(
            params["margin"], n_batch, 1, device, dtype, "margin"
        ).reshape(n_batch)
        y_true_scores = y_batch.gather(1, y_true.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(y_batch, dtype=torch.bool)
        _ = mask.scatter_(1, y_true.unsqueeze(1), False)
        other_scores = y_batch.masked_fill(~mask, -float("inf"))
        return (other_scores.max(dim=1).values - y_true_scores) >= margin

    if kind == OutKind.LINEAR_LE:
        coeff = _as_batched_vector(params["c"], n_batch, n_out, device, dtype, "c")
        bound = _as_batched_vector(params["d"], n_batch, 1, device, dtype, "d").reshape(n_batch)
        return (coeff * y_batch).sum(dim=1) >= bound + eps

    if kind == OutKind.RANGE:
        result = torch.zeros(n_batch, dtype=torch.bool, device=device)
        lb_raw = params.get("lb")
        ub_raw = params.get("ub")
        if lb_raw is not None:
            lb = _as_batched_vector(lb_raw, n_batch, n_out, device, dtype, "lb")
            result = result | (y_batch < lb - eps).any(dim=1)
        if ub_raw is not None:
            ub = _as_batched_vector(ub_raw, n_batch, n_out, device, dtype, "ub")
            result = result | (y_batch > ub + eps).any(dim=1)
        return result

    if kind == OutKind.UNSAFE_LINEAR:
        m_raw = params.get("M", 1)
        if isinstance(m_raw, torch.Tensor):
            m_rows = int(m_raw.item())
        elif isinstance(m_raw, int):
            m_rows = m_raw
        else:
            raise ValueError(f"UNSAFE_LINEAR: M must be int, got {m_raw!r}")
        c_raw = params.get("C", params.get("c"))
        if c_raw is None:
            raise ValueError("UNSAFE_LINEAR requires C or c params")
        c_tensor = c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)
        c_tensor = c_tensor.to(device=device, dtype=dtype)
        if c_tensor.dim() == 2:
            if c_tensor.shape == (m_rows, n_out):
                c_view = c_tensor.unsqueeze(0).expand(n_batch, -1, -1).contiguous()
            elif c_tensor.shape == (n_batch * m_rows, n_out):
                c_view = c_tensor.reshape(n_batch, m_rows, n_out).contiguous()
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: C shape {tuple(c_tensor.shape)} incompatible "
                    f"with N={n_batch}, M={m_rows}, n_out={n_out}"
                )
        elif c_tensor.dim() == 3:
            if c_tensor.shape == (1, m_rows, n_out):
                c_view = c_tensor.expand(n_batch, -1, -1).contiguous()
            elif c_tensor.shape == (n_batch, m_rows, n_out):
                c_view = c_tensor.contiguous()
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: c shape {tuple(c_tensor.shape)} incompatible "
                    f"with N={n_batch}, M={m_rows}, n_out={n_out}"
                )
        else:
            raise ValueError(f"UNSAFE_LINEAR: unsupported C dim {c_tensor.dim()}")
        d_raw = params.get("thresholds", params.get("d"))
        if d_raw is None:
            raise ValueError("UNSAFE_LINEAR requires thresholds or d params")
        d_tensor = d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)
        d_tensor = d_tensor.to(device=device, dtype=dtype)
        if d_tensor.dim() == 1 and d_tensor.numel() == m_rows:
            d_view = d_tensor.unsqueeze(0).expand(n_batch, -1).contiguous()
        elif d_tensor.shape == (1, m_rows):
            d_view = d_tensor.expand(n_batch, -1).contiguous()
        elif d_tensor.shape == (n_batch, m_rows):
            d_view = d_tensor.contiguous()
        else:
            raise ValueError(
                f"UNSAFE_LINEAR: d shape {tuple(d_tensor.shape)} incompatible "
                f"with N={n_batch}, M={m_rows}"
            )
        lhs = torch.einsum("bmo,bo->bm", c_view, y_batch)
        return (lhs <= d_view + eps).all(dim=1)

    raise NotImplementedError(f"ASSERT kind not supported: {kind}")


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
def verify_bab_batched(
    net: Net,
    solver_factory: Callable[[], Solver],
    config: Optional[BaBConfig] = None,
    *,
    max_batch_size: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    verbose: bool = False,
    _k_log: Optional[List[int]] = None,
) -> VerifyResult:
    """[BATCHED-API] K-batched Branch-and-Bound verification (single instance).

    Per iteration::

        K       = min(len(pool), max_batch_size, max_nodes - processed)
        batch   = pool.pop(K)                       # [K, D_flat]
        sol     = setup_and_solve_batch(net, [K,*input_shape] bounds, solver_factory())
        # decode per-lane:
        #   UNSAT       -> prune (region certified)
        #   SAT + violation (check_violations_batched) -> FALSIFIED (terminate)
        #   SAT spurious / UNKNOWN -> branch (or drop at max_depth)

    Soundness: returns CERTIFIED only when the pool drains via UNSAT pruning
    with every processed sub-box resolved (``all_resolved_unsat`` and
    ``pool.empty``). If the time/node budget exhausts with unproven sub-boxes
    remaining (branched-then-never-revisited, or dropped at ``max_depth``),
    returns UNKNOWN with
    ``metadata['reason'] == 'budget_exhausted_with_unproven_subboxes'``.

    Args:
        net: ACT network with a single-instance INPUT_SPEC (B=1 seed).
        solver_factory: callable returning a fresh ``Solver`` per iteration
            (no state leakage across iterations).
        config: ``BaBConfig``; ``bab_max_batch_size`` (if present, otherwise 8)
            caps K. ``max_depth`` and ``max_nodes`` cap the search tree.
        max_batch_size: explicit override for K cap; takes precedence over
            ``config.bab_max_batch_size``.
        time_budget_s: wall-clock budget (default 300 s).
        verbose: reserved.
        _k_log: diagnostic only — if supplied, the actual K used per iteration
            is appended. Tests use this to verify K fluctuates per D4.
    """
    if config is None:
        config = BaBConfig()
    if max_batch_size is None:
        max_batch_size = int(getattr(config, "bab_max_batch_size", 8))
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

    budget_s = time_budget_s if time_budget_s is not None else 300.0

    brancher = _build_branching_strategy(config.branching_method)
    pool = _build_bounding(config.bounding_method)

    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    root_bounds = seed_from_input_specs(spec_layers)
    input_shape: tuple[int, ...] = tuple(root_bounds.lb.shape[1:])

    pool.push(SubproblemBatch.from_bounds(root_bounds))

    start = time.time()
    processed = 0
    any_dropped_max_depth = False

    while not pool.empty:
        elapsed = time.time() - start
        if elapsed >= budget_s or processed >= config.max_nodes:
            break

        remaining_nodes = config.max_nodes - processed
        k_requested = min(len(pool), max_batch_size, remaining_nodes)
        if k_requested <= 0:
            break

        batch = pool.pop(batch_size=k_requested)
        k_actual = batch.batch_size
        if _k_log is not None:
            _k_log.append(k_actual)

        if input_shape:
            k_lb = batch.lb.reshape(k_actual, *input_shape)
            k_ub = batch.ub.reshape(k_actual, *input_shape)
        else:
            k_lb = batch.lb
            k_ub = batch.ub
        batched_bounds = Bounds(k_lb, k_ub)

        solver = solver_factory()
        solution = setup_and_solve_batch(
            net, batched_bounds, solver, timelimit=None,
        )

        sat_lane_idx = [
            i for i, s in enumerate(solution.statuses) if s == SolveStatus.SAT
        ]
        if sat_lane_idx:
            input_ids = get_input_ids(net)
            input_index = torch.tensor(
                input_ids, device=solution.x.device, dtype=torch.long,
            )
            sat_idx_t = torch.tensor(
                sat_lane_idx, device=solution.x.device, dtype=torch.long,
            )
            x_full = solution.x.index_select(0, sat_idx_t)
            x_input_flat = x_full.index_select(1, input_index)
            x_input_shaped = (
                x_input_flat.reshape(len(sat_lane_idx), *input_shape)
                if input_shape
                else x_input_flat
            )
            violations = check_violations_batched(net, x_input_shaped, assert_layer)
            for j, lane in enumerate(sat_lane_idx):
                if bool(violations[j].item()):
                    return VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_input_shaped[j].detach().cpu().clone(),
                        metadata={
                            "nodes": processed + k_actual,
                            "lane": lane,
                            "K": k_actual,
                        },
                    )

        unresolved_idx = torch.tensor(
            [i for i, status in enumerate(solution.statuses) if status != SolveStatus.UNSAT],
            device=batch.lb.device,
            dtype=torch.long,
        )
        if int(unresolved_idx.numel()) > 0:
            unresolved = SubproblemBatch(
                lb=batch.lb.index_select(0, unresolved_idx),
                ub=batch.ub.index_select(0, unresolved_idx),
                depths=batch.depths.index_select(0, unresolved_idx.cpu()),
            )
            branch_mask = unresolved.depths < int(config.max_depth)
            if bool((~branch_mask).any().item()):
                any_dropped_max_depth = True
            branch_idx = torch.where(branch_mask)[0]
            if int(branch_idx.numel()) > 0:
                branch_batch = SubproblemBatch(
                    lb=unresolved.lb.index_select(0, branch_idx.to(unresolved.lb.device)),
                    ub=unresolved.ub.index_select(0, branch_idx.to(unresolved.ub.device)),
                    depths=unresolved.depths.index_select(0, branch_idx),
                )
                scores = brancher.compute_scores(branch_batch, net)
                split_dims = brancher.select(scores)
                left, right = split_subproblems(branch_batch, split_dims)
                pool.push(left)
                pool.push(right)

        processed += k_actual

    pool_remaining = len(pool)
    elapsed_total = time.time() - start
    exhausted_time = elapsed_total >= budget_s
    exhausted_nodes = processed >= config.max_nodes

    if not any_dropped_max_depth and pool_remaining == 0:
        return VerifyResult(
            VerifyStatus.CERTIFIED,
            metadata={
                "nodes": processed,
                "pool_remaining": 0,
                "exhausted_budget_time": exhausted_time,
                "exhausted_budget_nodes": exhausted_nodes,
            },
        )

    return VerifyResult(
        VerifyStatus.UNKNOWN,
        metadata={
            "nodes": processed,
            "pool_remaining": pool_remaining,
            "exhausted_budget_time": exhausted_time,
            "exhausted_budget_nodes": exhausted_nodes,
            "reason": "budget_exhausted_with_unproven_subboxes",
        },
    )


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
    """Legacy single-solver Branch-and-Bound entry; backward-compat shim.

    Thin wrapper over [BATCHED-API] ``verify_bab_batched`` with K=1 (legacy
    one-subproblem-per-iteration semantics). Constructs a solver factory from
    the supplied solver instance's type so each BaB iteration gets a fresh
    instance. New callers should prefer ``verify_bab_batched`` directly.
    """
    if config is None:
        config = BaBConfig(
            max_depth=max_depth if max_depth is not None else 20,
            max_nodes=(max_nodes or max_subproblems or 2000),
            verbose=verbose,
        )
    budget = (
        time_budget_s if time_budget_s is not None
        else (timelimit if timelimit is not None else 300.0)
    )
    solver_type = type(solver)
    return verify_bab_batched(
        net=net,
        solver_factory=lambda: solver_type(),
        config=config,
        max_batch_size=1,
        time_budget_s=budget,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


class _StubNet:  # pragma: no cover
    layers = []


def test_imports():  # pragma: no cover
    for sym in (
        verify_bab,
        BaBConfig,
        BabNode,
        SubproblemBatch,
        split_subproblems,
        BranchingStrategy,
        BoundingStrategy,
        RandomBranching,
        RandomBounding,
    ):
        assert sym is not None


def test_config_yaml_roundtrip():  # pragma: no cover
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


def test_subproblem_batch():  # pragma: no cover
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


def test_split_subproblems():  # pragma: no cover
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


def test_random_branching():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))

    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, cast(Net, cast(object, _StubNet())))
    assert scores.shape == (1, 3)
    assert (scores >= 0).all()

    dims = brancher.select(scores)
    assert dims.shape == (1,)
    assert 0 <= dims.item() <= 2


def test_random_branching_with_mask():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    mask = torch.tensor([False, True, False])

    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, cast(Net, cast(object, _StubNet())), unstable_mask=mask)
    assert scores[0, 0].item() == 0.0
    assert scores[0, 2].item() == 0.0
    assert brancher.select(scores).item() == 1


def test_random_bounding():  # pragma: no cover
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


def test_babnode_compat():  # pragma: no cover
    bounds = Bounds(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
    node = BabNode(box=bounds, depth=3, score=0.5)
    batch = node.to_batch()
    assert batch.batch_size == 1
    assert batch.depths[0].item() == 3


class _IdentityOutput(torch.nn.Module):  # pragma: no cover
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


def _make_assert_layer(kind: str, params: dict[str, ParamValue], n_out: int) -> Layer:  # pragma: no cover
    from act.back_end.layer_schema import LayerKind

    merged: dict[str, ParamValue] = {"kind": kind}
    merged.update(params)
    if "C" not in merged or "thresholds" not in merged or "M" not in merged:
        batch_size = 1
        for key in ("y_true", "margin", "c", "d", "lb", "ub"):
            value = merged.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_size = max(batch_size, int(value.shape[0]))
        if kind == OutKind.UNSAFE_LINEAR:
            c_value = merged.get("c")
            d_value = merged.get("d")
            if not isinstance(c_value, torch.Tensor) or not isinstance(d_value, torch.Tensor):
                raise ValueError("UNSAFE_LINEAR test layer requires tensor c and d")
            if c_value.dim() == 3:
                batch_size = int(c_value.shape[0])
                m_rows = int(c_value.shape[1])
                merged["C"] = c_value.reshape(batch_size * m_rows, n_out)
            elif c_value.dim() == 2:
                m_rows = int(c_value.shape[0])
                merged["C"] = c_value
            else:
                raise ValueError(f"UNSAFE_LINEAR test c dim {c_value.dim()} unsupported")
            merged["thresholds"] = d_value.reshape(batch_size, m_rows)
            merged["M"] = m_rows
        else:
            merged["C"] = torch.zeros(batch_size, n_out)
            merged["thresholds"] = torch.zeros(batch_size, 1)
            merged["M"] = 1
    return Layer(
        id=99,
        kind=LayerKind.ASSERT.value,
        params=merged,
        in_vars=list(range(n_out)),
        out_vars=list(range(n_out)),
    )


def _test_check_violations_batched_per_kind():  # pragma: no cover
    y = torch.tensor(
        [
            [3.0, 1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0, -1.0],
            [0.0, 3.0, 1.0, -1.0],
            [0.0, 1.0, 3.0, -1.0],
            [0.0, 1.0, 4.0, -1.0],
            [0.0, 1.0, 2.0, 5.0],
            [0.0, 1.0, 2.0, 6.0],
            [4.0, 1.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    net = _IdentityOutput()
    n_batch, n_out = y.shape

    top1 = _make_assert_layer(
        OutKind.TOP1_ROBUST,
        {"y_true": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])},
        n_out,
    )
    y_true_top1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    expected_top1 = y.argmax(dim=1) != y_true_top1
    assert torch.equal(check_violations_batched(net, y, top1), expected_top1)

    margin = _make_assert_layer(
        OutKind.MARGIN_ROBUST,
        {
            "y_true": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            "margin": torch.full((n_batch,), 1.5, dtype=y.dtype),
        },
        n_out,
    )
    y_true = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    true_scores = y.gather(1, y_true.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(y, dtype=torch.bool)
    _ = mask.scatter_(1, y_true.unsqueeze(1), False)
    expected_margin = (y.masked_fill(~mask, -float("inf")).max(dim=1).values - true_scores) >= 1.5
    assert torch.equal(check_violations_batched(net, y, margin), expected_margin)

    linear = _make_assert_layer(
        OutKind.LINEAR_LE,
        {"c": torch.ones(n_batch, n_out, dtype=y.dtype), "d": torch.full((n_batch,), 4.0, dtype=y.dtype)},
        n_out,
    )
    expected_linear = y.sum(dim=1) >= 4.0 + 1e-8
    assert torch.equal(check_violations_batched(net, y, linear), expected_linear)

    range_layer = _make_assert_layer(
        OutKind.RANGE,
        {
            "lb": torch.full((n_batch, n_out), -0.5, dtype=y.dtype),
            "ub": torch.full((n_batch, n_out), 4.5, dtype=y.dtype),
        },
        n_out,
    )
    expected_range = ((y < -0.5 - 1e-8) | (y > 4.5 + 1e-8)).any(dim=1)
    assert torch.equal(check_violations_batched(net, y, range_layer), expected_range)

    c = torch.eye(n_out, dtype=y.dtype).unsqueeze(0).expand(n_batch, -1, -1).contiguous()
    d = torch.full((n_batch, n_out), 3.5, dtype=y.dtype)
    unsafe = _make_assert_layer(
        OutKind.UNSAFE_LINEAR,
        {"c": c, "d": d, "C": c.reshape(n_batch * n_out, n_out), "thresholds": d, "M": n_out},
        n_out,
    )
    expected_unsafe = (y <= 3.5 + 1e-8).all(dim=1)
    assert torch.equal(check_violations_batched(net, y, unsafe), expected_unsafe)



def _test_check_violations_batched_b1_scalar_params():  # pragma: no cover
    net = _IdentityOutput()
    x = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float64)
    assert_layer = _make_assert_layer(
        OutKind.TOP1_ROBUST,
        {"y_true": torch.tensor([0], dtype=torch.long)},
        n_out=3,
    )
    result = check_violations_batched(net, x, assert_layer)
    assert tuple(result.shape) == (1,)
    assert bool(result[0].item()) is True



# ---------------------------------------------------------------------------
# C12: K-batched verify_bab_batched test fixtures
# ---------------------------------------------------------------------------


def _load_bab_deep_net() -> Optional[Net]:  # pragma: no cover
    """Load layer_testing_bab_deep.json from examples/nets, or None if absent.

    Returns None silently when the fixture is missing so tests can skip rather
    than hard-fail in isolated environments. Forces CPU device for hermetic
    test execution: the BaB integration tests must not depend on GPU
    availability or device-manager global state.
    """
    from pathlib import Path

    from act.back_end.serialization.serialization import load_net_from_file
    from act.util.device_manager import initialize_device

    here = Path(__file__).resolve()
    candidate = here.parents[1] / "examples" / "nets" / "layer_testing_bab_deep.json"
    if not candidate.exists():
        return None
    initialize_device("cpu", "float64")
    return load_net_from_file(str(candidate), target_device="cpu")


class _UnknownSolver(Solver):  # pragma: no cover
    """Mock solver: returns UNKNOWN on every lane (forces BaB to branch)."""

    def solve_batch(self, problem, timelimit=None):
        from act.back_end.solver.solver_base import BatchLPSolution

        n = problem.N
        return BatchLPSolution(
            statuses=tuple([SolveStatus.UNKNOWN] * n),
            x=torch.zeros(
                (n, problem.nvars), device=problem.lb.device, dtype=problem.lb.dtype,
            ),
            max_viol=torch.full(
                (n,), float("nan"), device=problem.lb.device, dtype=problem.lb.dtype,
            ),
        )


class _OOMSolver(Solver):  # pragma: no cover
    """Mock solver: raises an OOM-like exception on every solve_batch call."""

    def solve_batch(self, problem, timelimit=None):
        raise RuntimeError("CUDA out of memory: mocked for OOM-fails-loud test")


def _test_bab_kbatch_status_parity():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_kbatch_status_parity: layer_testing_bab_deep.json absent")
        return
    from act.back_end.solver.solver_torchlp import TorchLPSolver

    config = BaBConfig(max_depth=6, max_nodes=32, verbose=False)
    statuses_by_k: dict[int, VerifyStatus] = {}
    for k in (1, 2, 4, 8):
        result = verify_bab_batched(
            net=net,
            solver_factory=lambda: TorchLPSolver(),
            config=config,
            max_batch_size=k,
            time_budget_s=60.0,
        )
        statuses_by_k[k] = result.status
    distinct = set(statuses_by_k.values())
    assert len(distinct) == 1, (
        f"K-batch status parity violated: {statuses_by_k}"
    )


def _test_bab_budget_exhaustion_returns_unknown():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_budget_exhaustion_returns_unknown: fixture absent")
        return
    config = BaBConfig(max_depth=10, max_nodes=2, verbose=False)
    result = verify_bab_batched(
        net=net,
        solver_factory=lambda: _UnknownSolver(),
        config=config,
        max_batch_size=1,
        time_budget_s=30.0,
    )
    assert result.status == VerifyStatus.UNKNOWN, (
        f"Expected UNKNOWN under-budget with mock-UNKNOWN solver, got "
        f"{result.status}; metadata={result.metadata}"
    )
    assert result.metadata.get("reason") == "budget_exhausted_with_unproven_subboxes", (
        f"Missing soundness-reason metadata: {result.metadata}"
    )


def _test_bab_oom_fails_loud():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_oom_fails_loud: fixture absent")
        return
    config = BaBConfig(max_depth=5, max_nodes=10, verbose=False)
    raised = False
    try:
        verify_bab_batched(
            net=net,
            solver_factory=lambda: _OOMSolver(),
            config=config,
            max_batch_size=4,
            time_budget_s=10.0,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        assert "out of memory" in msg, f"Unexpected RuntimeError message: {e}"
        raised = True
    assert raised, "OOM exception was swallowed — silent fallback present"


def _test_bab_k_fluctuates():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_k_fluctuates: fixture absent")
        return
    config = BaBConfig(max_depth=8, max_nodes=20, verbose=False)
    k_log: List[int] = []
    _ = verify_bab_batched(
        net=net,
        solver_factory=lambda: _UnknownSolver(),
        config=config,
        max_batch_size=8,
        time_budget_s=30.0,
        _k_log=k_log,
    )
    distinct = set(k_log)
    assert len(distinct) >= 2, (
        f"K did not fluctuate across iterations (got {k_log}); dynamic K-batching "
        f"requires at least 2 distinct K values per D4."
    )


_TESTS = [  # pragma: no cover
    test_imports,
    test_config_yaml_roundtrip,
    test_subproblem_batch,
    test_split_subproblems,
    test_random_branching,
    test_random_branching_with_mask,
    test_random_bounding,
    test_babnode_compat,
    _test_check_violations_batched_per_kind,
    _test_check_violations_batched_b1_scalar_params,
    _test_bab_kbatch_status_parity,
    _test_bab_budget_exhaustion_returns_unknown,
    _test_bab_oom_fails_loud,
    _test_bab_k_fluctuates,
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
