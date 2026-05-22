#===- act/back_end/verifier.py - Spec-free Verification Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free verification. Assumes the ACT Net already encodes
#   both input and output specifications via INPUT_SPEC and ASSERT layers
#   (produced by torch2act.TorchToACT).
#
# Architecture — verify_once:
#   1. Seed [B, *input_shape] bounds from INPUT_SPEC layers (no CSP).
#   2. analyze() propagates batched bounds through every TF op.
#   3. Read pre-encoded [B*M, n_out] linear-form C / [B, M] thresholds / M
#      from the ASSERT layer params (produced upstream by
#      OutputSpec.encode_linear at FE construction time).
#   4. INTERVAL CERTIFICATION: one tensor pass computes margin_max under
#      output bounds; sample b is CERTIFIED iff every M lane passes.
#   5. CONCRETE FALSIFICATION (when model_fn given): one batched forward at
#      box centre; samples whose concrete output meets-or-exceeds threshold
#      become FALSIFIED. Remaining samples are UNKNOWN.
#   6. Return List[VerifyResult] of length B (one per input lane).
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, *, model_fn=None) -> List[VerifyResult]
#       Pure-tensor batched single-shot verifier.
#   - setup_and_solve_batch(net, input_bounds_per_b, solver, timelimit=None)
#       Batch-native CSP setup helper used by LP and BaB refinement.
#   - find_entry_layer_id / get_input_ids / get_output_ids /
#     gather_input_spec_layers / get_assert_layer / seed_from_input_specs /
#     add_all_input_specs (helpers).
#
# Notes:
#   * Spec-free verification: all constraints extracted from ACT Net layers.
#   * verify_once returns one VerifyResult per lane (len(result) == B).
#   * INPUT_SPEC constraints (including LIN_POLY) are propagated through
#     analyze(); they enter via add_all_input_specs into entry_fact.cons.
#     LIN_POLY constraints are not consumed by verify_once's interval
#     certification; they are preserved for the batch-native solver path.

from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any, TYPE_CHECKING, cast

import torch
import copy

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet, Fact, Net
from act.back_end.solver.solver_base import Solver, SolveStatus, BatchLPSolution
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import validate_constraints

if TYPE_CHECKING:
    from act.back_end.analyze import AnalyzeCache

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind

# Verification types (canonical location: act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult

# -----------------------------------------------------------------------------
# Sequential per-sample slicing (for B>1 BaB)
# -----------------------------------------------------------------------------

def _slice_first_dim(value: Any, sample_idx: int, expected_b: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == expected_b:
        return value[sample_idx:sample_idx + 1]
    return value


def slice_net_to_sample(net: Net, sample_idx: int) -> Net:
    from act.front_end.spec_creator_base import LabeledInputTensor

    mutable_kinds = {
        LayerKind.INPUT.value,
        LayerKind.INPUT_SPEC.value,
        LayerKind.ASSERT.value,
    }
    layers = []
    for layer in net.layers:
        if layer.kind not in mutable_kinds:
            layers.append(layer)
            continue
        layer2 = copy.copy(layer)
        layer2.params = dict(layer.params)
        layer2.in_vars = list(layer.in_vars)
        layer2.out_vars = list(layer.out_vars)
        layer2.cache = dict(layer.cache)
        layers.append(layer2)
    net2 = copy.copy(net)
    net2.layers = layers
    net2.preds = net.preds
    net2.succs = net.succs
    net2.by_id = {layer.id: layer for layer in layers}

    entry_id = find_entry_layer_id(net2)
    input_layer = net2.by_id[entry_id]
    shape = input_layer.params.get("shape") or []
    shape_t = tuple(shape) if isinstance(shape, (list, tuple)) else ()
    B = int(shape_t[0]) if shape_t else 1
    if shape_t and int(shape_t[0]) == B:
        input_layer.params["shape"] = (1,) + tuple(shape_t[1:])
    li = input_layer.params.get("labeled_input")
    if isinstance(li, LabeledInputTensor):
        new_tensor = _slice_first_dim(li.tensor, sample_idx, B)
        new_label = _slice_first_dim(li.label, sample_idx, B) if li.label is not None else None
        input_layer.__dict__["params"]["labeled_input"] = LabeledInputTensor(
            tensor=new_tensor, label=new_label,
        )

    for spec_layer in gather_input_spec_layers(net2):
        for key in ("center", "eps", "lb", "ub", "A", "b"):
            val = spec_layer.params.get(key)
            if val is not None:
                spec_layer.params[key] = _slice_first_dim(val, sample_idx, B)

    assert_layer = get_assert_layer(net2)
    m_raw = assert_layer.params.get("M", 1)
    if isinstance(m_raw, torch.Tensor):
        m_rows = int(m_raw.item())
    elif isinstance(m_raw, int):
        m_rows = m_raw
    else:
        raise ValueError(f"ASSERT M must be int or tensor, got {m_raw!r}")
    for key in ("y_true", "margin", "c", "d", "lb", "ub"):
        val = assert_layer.params.get(key)
        if val is not None:
            assert_layer.params[key] = _slice_first_dim(val, sample_idx, B)
    # C is [B*M, n_out] — first dim is B*M not B, so slice rows manually
    c_big = assert_layer.params.get("C")
    if isinstance(c_big, torch.Tensor) and c_big.shape[0] == B * m_rows:
        assert_layer.params["C"] = c_big[sample_idx * m_rows:(sample_idx + 1) * m_rows]
    thresholds = assert_layer.params.get("thresholds")
    if isinstance(thresholds, torch.Tensor) and thresholds.shape[0] == B:
        assert_layer.params["thresholds"] = thresholds[sample_idx:sample_idx + 1]

    return net2


# -----------------------------------------------------------------------------
# ACT Net extraction helpers
# -----------------------------------------------------------------------------

def find_entry_layer_id(net) -> int:
    """Return the id of the single INPUT layer."""
    candidates = [L.id for L in net.layers if L.kind == LayerKind.INPUT.value]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(candidates)}.")
    return candidates[0]

def get_input_ids(net) -> List[int]:
    """Return input variable IDs (out_vars of INPUT layer)."""
    entry = find_entry_layer_id(net)
    return list(net.by_id[entry].out_vars)

def get_output_ids(net) -> List[int]:
    """Return output variable IDs (in_vars of ASSERT layer)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return list(assert_layer.in_vars)

def gather_input_spec_layers(net):
    """Return list of INPUT_SPEC layers."""
    return [L for L in net.layers if L.kind == LayerKind.INPUT_SPEC.value]

def get_assert_layer(net):
    """Return the ASSERT layer (must be last)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return assert_layer

# -----------------------------------------------------------------------------
# Seed and input spec helpers
# -----------------------------------------------------------------------------

def seed_from_input_specs(spec_layers) -> Bounds:
    """
    Create seed Bounds from INPUT_SPEC layers.
    Prefers BOX, then LINF_BALL, raises if only LIN_POLY exists.
    
    Note: This extracts only box bounds for seeding abstract interpretation.
    All constraints (including LIN_POLY) are added via add_all_input_specs().
    """
    # BOX first
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.BOX and "lb" in spec_layer.params and "ub" in spec_layer.params:
            return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
    
    # LINF_BALL next
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LINF_BALL:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is not None and eps is not None:
                e = torch.tensor(eps)
                return Bounds(center - e, center + e)
    
    # LIN_POLY only -> error
    if any(spec_layer.params.get("kind") == InKind.LIN_POLY for spec_layer in spec_layers):
        raise ValueError("LIN_POLY requires a seed box (BOX or LINF_BALL).")
    
    raise ValueError("No valid input specification found for seeding.")

def add_all_input_specs(globalC: ConSet, input_ids: List[int], spec_layers) -> None:
    """
    Add all INPUT_SPEC constraints to constraint set.
    
    This function adds:
    - BOX constraints (box bounds)
    - LINF_BALL constraints (converted to box)
    - LIN_POLY constraints (linear polytope A·x ≤ b)
    
    The LIN_POLY constraints are tagged with "in:linpoly" and will be
    exported by export_to_batch_problem() in cons_exportor.py.
    """
    for L in spec_layers:
        k = L.params.get("kind")
        if k == InKind.BOX:
            globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
        elif k == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = torch.tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LIN_POLY:
            A, b = L.params["A"], L.params["b"]
            globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": A, "b": b}))
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")




@torch.no_grad()
def setup_and_solve_batch(
    net,
    input_bounds_per_b: Bounds,
    solver: Solver,
    timelimit: Optional[float] = None,
    *,
    cache: Optional["AnalyzeCache"] = None,
) -> BatchLPSolution:
    """[BATCHED-API] Orchestrate analyze → export_to_batch_problem → solve_batch.

    ``input_bounds_per_b`` must already be a tensor-view batch
    ``[B, *input_shape]``; B=1 is just
    the length-one batch case, not a scalar special case.
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_batch_problem

    if input_bounds_per_b.lb.dim() < 2 or input_bounds_per_b.ub.dim() < 2:
        raise ValueError(
            f"setup_and_solve_batch: input_bounds_per_b must be batched "
            f"[B, *input_shape], got lb={tuple(input_bounds_per_b.lb.shape)} "
            f"ub={tuple(input_bounds_per_b.ub.shape)}"
        )

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    entry_fact = Fact(bounds=input_bounds_per_b, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)

    _before, after, globalC = analyze(net, entry_id, entry_fact, cache=cache)
    validate_constraints(globalC, after, net)

    problem = export_to_batch_problem(
        net=net,
        globalC=globalC,
        assert_layer=assert_layer,
        input_box_per_b=input_bounds_per_b,
    )
    solution = solver.solve_batch(problem, timelimit=timelimit)

    expected_n = int(input_bounds_per_b.lb.shape[0])
    if len(solution.statuses) != expected_n:
        raise ValueError(
            f"setup_and_solve_batch: solver returned {len(solution.statuses)} "
            f"statuses for B={expected_n}"
        )
    valid_statuses = {SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN}
    unexpected = [status for status in solution.statuses if status not in valid_statuses]
    if unexpected:
        raise ValueError(
            f"setup_and_solve_batch: unexpected solver statuses {unexpected}"
        )
    if solution.max_viol.shape != (expected_n,):
        raise ValueError(
            f"setup_and_solve_batch: max_viol shape "
            f"{tuple(solution.max_viol.shape)} != ({expected_n},)"
        )
    return solution


@torch.no_grad()
def verify_lp_batched(
    net,
    solver_factory: Callable[[], Solver],
    timelimit: Optional[float] = None,
) -> List[VerifyResult]:
    """[BATCHED-API] Run one native batched LP verification pass.

    The ACT net supplies a batched INPUT_SPEC seed ``[B, *input_shape]`` and a
    batched ASSERT layer. ``setup_and_solve_batch`` solves all B LPs at once;
    this function decodes each solver lane to a ``VerifyResult`` and validates
    SAT candidates concretely before reporting FALSIFIED.
    """
    import importlib

    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2 or seed_bounds.ub.dim() < 2:
        raise ValueError(
            f"verify_lp_batched: seed bounds must be [B, *input_shape], "
            f"got lb={tuple(seed_bounds.lb.shape)} ub={tuple(seed_bounds.ub.shape)}"
        )
    batch_size = int(seed_bounds.lb.shape[0])
    solver = solver_factory()
    solution = setup_and_solve_batch(
        net,
        Bounds(seed_bounds.lb.clone(), seed_bounds.ub.clone()),
        solver,
        timelimit=timelimit,
    )
    if len(solution.statuses) != batch_size:
        raise ValueError(
            f"verify_lp_batched: solver returned {len(solution.statuses)} "
            f"statuses for B={batch_size}"
        )
    if solution.x.dim() != 2 or solution.x.shape[0] != batch_size:
        raise ValueError(
            f"verify_lp_batched: solution.x must be [B, nvars], got "
            f"shape={tuple(solution.x.shape)} for B={batch_size}"
        )

    input_ids = get_input_ids(net)
    input_index = torch.tensor(input_ids, device=solution.x.device, dtype=torch.long)
    x_candidates = solution.x.index_select(1, input_index).reshape_as(seed_bounds.lb)
    assert_layer = get_assert_layer(net)

    sat_mask = torch.tensor(
        [status in (SolveStatus.SAT, "FEASIBLE") for status in solution.statuses],
        device=x_candidates.device,
        dtype=torch.bool,
    )
    violations = torch.zeros(batch_size, device=x_candidates.device, dtype=torch.bool)
    if bool(sat_mask.any().item()):
        bab_module = importlib.import_module("act.back_end.bab.bab")
        sat_idx = torch.where(sat_mask)[0]
        checked_sat = bab_module.check_violations_batched(
            net, x_candidates.index_select(0, sat_idx), assert_layer,
        )
        if checked_sat.shape != (int(sat_idx.numel()),):
            raise ValueError(
                f"verify_lp_batched: check_violations_batched returned "
                f"shape={tuple(checked_sat.shape)} expected ({int(sat_idx.numel())},)"
            )
        violations.scatter_(
            0, sat_idx, checked_sat.to(device=x_candidates.device, dtype=torch.bool),
        )

    results: List[VerifyResult] = []
    x_cpu = x_candidates.detach().cpu()
    max_viol_cpu = solution.max_viol.detach().cpu()
    for lane, status in enumerate(solution.statuses):
        metadata: Dict[str, Any] = {
            "lane": lane,
            "B": batch_size,
            "solver_status": status,
            "max_viol": float(max_viol_cpu[lane].item()),
        }
        if status in (SolveStatus.SAT, "FEASIBLE"):
            if bool(violations[lane].item()):
                results.append(
                    VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_cpu[lane].clone(),
                        metadata=metadata,
                    )
                )
            else:
                metadata["validation"] = "no_verified_violation"
                results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        elif status in (SolveStatus.UNSAT, "INFEASIBLE"):
            results.append(VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata))
        elif status == "TIMEOUT":
            results.append(VerifyResult(VerifyStatus.TIMEOUT, metadata=metadata))
        elif status == SolveStatus.UNKNOWN:
            results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        else:
            raise ValueError(f"verify_lp_batched: unexpected solver status {status!r}")
    return results


# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------


def _get_output_layer_bounds(net, after: Dict[int, Fact]) -> Bounds:
    """Return the Bounds tensor produced by the network's output layer.

    The output layer is the unique predecessor of the ASSERT layer; the
    returned Bounds is shaped ``[B, n_out]``.
    """
    assert_layer = get_assert_layer(net)
    pred_ids = net.preds.get(assert_layer.id, [])
    if len(pred_ids) != 1:
        raise ValueError(
            f"ASSERT layer {assert_layer.id} must have exactly one "
            f"predecessor (the network output), got predecessors={pred_ids}"
        )
    return after[pred_ids[0]].bounds


@torch.no_grad()
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[VerifyResult]:
    """Single-shot, pure-tensor batched verifier.

    Pipeline:

      1. Seed bounds from INPUT_SPEC layers (already shaped ``[B, *input_shape]``).
      2. ``analyze`` propagates batched bounds through every layer.
      3. Read pre-encoded ``C`` / ``thresholds`` / ``M`` from the ASSERT
         layer params (encoding lives in ``OutputSpec.encode_linear`` on the
         front-end; verify_once does no kind-dispatch).
      4. INTERVAL CERTIFICATION: in one tensor pass, compute the
         per-row interval upper bound of ``C @ y`` and compare to the
         per-lane threshold; ALL of a sample's M lanes must pass for that
         sample to be CERTIFIED.
      5. CONCRETE FALSIFICATION (only if ``model_fn`` given): evaluate the
         model at the box centre; any sample where a lane's concrete
         margin meets-or-exceeds the threshold is FALSIFIED.
      6. Remaining samples are UNKNOWN.

    Args:
        net: an ACT ``Net`` whose first layer is INPUT, last layer is ASSERT,
            and whose INPUT_SPEC layers carry already-batchified
            ``[B, *input_shape]`` lb/ub.
        model_fn: optional callable mapping ``x: [B, *input_shape] ->
            [B, n_out]`` for concrete falsification. If omitted, the
            FALSIFIED status is never produced (FALSIFIED requires evidence).

    Returns:
        ``List[VerifyResult]`` of length ``B`` (one per input lane). Each
        result carries ``status`` plus a ``metadata['lane'] = i`` and any
        ``counterexample`` (a ``torch.Tensor`` of shape ``[*input_shape]``)
        for FALSIFIED lanes.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import get_transfer_function

    # 1. Extract structure and seed.
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        raise ValueError(
            f"verify_once: INPUT_SPEC seed must be batched [B, *input_shape], "
            f"got dim={seed_bounds.lb.dim()} shape={tuple(seed_bounds.lb.shape)}. "
            f"Use VerifiableModel._merge_specs_to_batch (front-end) or manually "
            f"expand INPUT_SPEC lb/ub to [B, ...] before calling verify_once."
        )
    B = seed_bounds.lb.shape[0]

    # Dual standalone dispatch: when ``--solver dual`` is set (dual moved
    # dual from the --tf-mode axis to the --solver axis), route through
    # DualSolver.evaluate_spec instead of analyze() + interval cert. LP/Gurobi
    # path remains authoritative for the LP-feeding TFs (interval/hybridz).
    # ``ensure_active_tf`` still self-heals the TF default for interval/hybridz
    # callers; ``is_dual_solver_active`` reads the orthogonal solver-mode global.
    from act.back_end.transfer_functions import ensure_active_tf, is_dual_solver_active
    active_tf = ensure_active_tf("interval")

    if is_dual_solver_active():
        from act.back_end.solver.solver_dual import DualSolver
        from act.front_end.specs import OutputSpec

        def _unbatch(val: Any) -> Any:
            # ASSERT params are pre-batchified ([B, ...]) by FE; OutputSpec
            # constructor expects unbroadcasted scalar/1-D form. Single-property
            # batch verification: all rows share the same spec, so row 0 is the
            # canonical form. Per-sample-varying spec support is a future task.
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == B:
                return val[0]
            return val

        out_spec = OutputSpec(
            kind=assert_layer.params.get("kind"),
            c=_unbatch(assert_layer.params.get("c")),
            d=_unbatch(assert_layer.params.get("d")),
            y_true=assert_layer.params.get("y_true"),
            margin=_unbatch(assert_layer.params.get("margin")),
            lb=_unbatch(assert_layer.params.get("lb")),
            ub=_unbatch(assert_layer.params.get("ub")),
        )
        num_classes = len(output_ids)
        # DualSolver is now self-contained: no tf parameter, evaluate_spec
        # computes its own forward bounds internally from the net.
        result = DualSolver().evaluate_spec(net, out_spec, num_classes=num_classes)
        return result.to_verify_results()

    # 2. Build entry_fact (with all INPUT_SPEC constraints) and analyze.
    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    _before, after, _globalC = analyze(net, entry_id, entry_fact)

    # 3. Pull output bounds (pre-ASSERT layer's Fact).
    output_bounds = _get_output_layer_bounds(net, after)
    output_lb = output_bounds.lb
    output_ub = output_bounds.ub
    if output_lb.dim() != 2 or output_lb.shape[0] != B:
        raise ValueError(
            f"verify_once: output bounds must be [B={B}, n_out], got "
            f"shape={tuple(output_lb.shape)}. Some TF op on this network's "
            f"path collapsed the leading batch dimension."
        )
    n_out = output_lb.shape[1]
    if n_out != len(output_ids):
        raise ValueError(
            f"verify_once: output_lb has n_out={n_out} but ASSERT.in_vars "
            f"has length {len(output_ids)}"
        )
    device = output_lb.device
    dtype = output_lb.dtype

    # 4. Read pre-encoded ASSERT params (produced by OutputSpec.encode_linear
    # at FE construction time). Dispatch on ``kind`` because UNSAFE_LINEAR
    # has EXISTS-row safety semantics while the four other kinds (LINEAR_LE,
    # TOP1_ROBUST, MARGIN_ROBUST, RANGE) share an ALL-rows form.
    C = assert_layer.params["C"].to(device=device, dtype=dtype)
    thresholds = assert_layer.params["thresholds"].to(device=device, dtype=dtype)
    M = int(assert_layer.params["M"])
    kind = assert_layer.params.get("kind")
    is_unsafe_linear = kind == OutKind.UNSAFE_LINEAR
    assert C.dim() == 2 and C.shape == (B * M, n_out), (
        f"verify_once: ASSERT params['C'].shape={tuple(C.shape)} "
        f"expected ({B * M}, {n_out})"
    )
    assert thresholds.shape == (B, M), (
        f"verify_once: ASSERT params['thresholds'].shape="
        f"{tuple(thresholds.shape)} expected ({B}, {M})"
    )

    C_pos = C.clamp(min=0)
    C_neg = C.clamp(max=0)
    lb_exp = output_lb.repeat_interleave(M, dim=0)
    ub_exp = output_ub.repeat_interleave(M, dim=0)

    if is_unsafe_linear:
        # UNSAFE polytope = {y : C y <= d}. Property is SAFE iff for all y in
        # the box, EXISTS row i with c_i @ y > d_i (i.e. y leaves the polytope
        # on row i). Sound under-approximation: EXISTS row i such that
        # min_{y in box} (c_i @ y) > d_i. min(c_i @ y) = c_i_pos @ lb + c_i_neg @ ub.
        margin_min = (C_pos * lb_exp + C_neg * ub_exp).sum(dim=-1)
        certified = (margin_min.view(B, M) > thresholds).any(dim=-1)
    else:
        # LINEAR_LE / TOP1_ROBUST / MARGIN_ROBUST / RANGE: certified iff for
        # all y in the box, ALL rows max_y (c_i @ y) < d_i.
        margin_max = (C_pos * ub_exp + C_neg * lb_exp).sum(dim=-1)
        certified = (margin_max.view(B, M) < thresholds).all(dim=-1)

    # 5. Concrete falsification (optional).
    falsified = torch.zeros(B, dtype=torch.bool, device=device)
    counterexamples: List[Optional[torch.Tensor]] = [None] * B
    if model_fn is not None:
        x_center = 0.5 * (seed_bounds.lb + seed_bounds.ub)
        y_concrete = model_fn(x_center)
        if y_concrete.dim() != 2 or y_concrete.shape != (B, n_out):
            raise ValueError(
                f"verify_once: model_fn returned shape "
                f"{tuple(y_concrete.shape)}, expected ({B}, {n_out})"
            )
        y_concrete = y_concrete.to(device=device, dtype=dtype)
        C_view = C.view(B, M, n_out)
        concrete_violation = torch.einsum("bmn,bn->bm", C_view, y_concrete)
        if is_unsafe_linear:
            # Concrete y is in the UNSAFE polytope iff ALL rows c_i @ y <= d_i;
            # that is the violation condition for UNSAFE_LINEAR.
            falsified = (~certified) & (
                (concrete_violation <= thresholds).all(dim=-1)
            )
        else:
            # ALL-rows kinds: FALSIFIED iff ANY lane's concrete margin
            # meets-or-exceeds threshold.
            falsified = (~certified) & (
                (concrete_violation >= thresholds).any(dim=-1)
            )
        if falsified.any():
            x_center_cpu = x_center.detach().cpu()
            # B1 (oracle-verified): single sync via .tolist() replaces B per-element .item() syncs.
            # torch.where returns ascending indices; lane order is preserved.
            for i in torch.where(falsified)[0].tolist():
                counterexamples[i] = x_center_cpu[i].clone()

    # 6. Assemble per-lane results.
    results: List[VerifyResult] = []
    cert_list = certified.tolist()
    fals_list = falsified.tolist()
    for i in range(B):
        meta: Dict[str, Any] = {"lane": i, "B": B, "M": M}
        if cert_list[i]:
            results.append(
                VerifyResult(VerifyStatus.CERTIFIED, metadata=meta)
            )
        elif fals_list[i]:
            results.append(
                VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=counterexamples[i],
                    metadata=meta,
                )
            )
        else:
            results.append(
                VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)
            )
    return results


#===---------------------------------------------------------------------===#
# Self-contained ASSERT-encoding + verify_once test battery.
# Run via: python -m act.back_end.verifier
#===---------------------------------------------------------------------===#


def _test_build_top1_robust_drops_y_true_row() -> None:  # pragma: no cover
    # Encoding is row-deletion, not masking: every row is e_j - e_{y_true}
    # for j != y_true, hence M = K-1 and Frobenius row norm = sqrt(2).
    from act.front_end.specs import OutputSpec, OutKind

    K = 5
    out = OutputSpec(
        kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0, 2, 4])
    ).encode_linear(
        B=3, n_out=K, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == K - 1, f"expected M=K-1=4, got {out['M']}"
    assert out["C"].shape == (3 * (K - 1), K), (
        f"expected C.shape == (B*M, K) == (12, 5), got "
        f"{tuple(out['C'].shape)}"
    )
    row_norms = out["C"].norm(dim=1)
    assert (row_norms > 0).all(), (
        f"found a zero row in C (y_true row was masked, not dropped): "
        f"norms={row_norms.tolist()}"
    )
    expected_norm = torch.full_like(row_norms, 2.0).sqrt()
    assert torch.allclose(row_norms, expected_norm), (
        f"every row should be e_j - e_{{y_true}} with ||.||=sqrt(2); "
        f"got norms={row_norms.tolist()}"
    )


def _test_build_linear_le_threshold_is_d_unchanged() -> None:  # pragma: no cover
    from act.front_end.specs import OutputSpec, OutKind

    out = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0, -1.0]),
        d=torch.tensor(0.5),
    ).encode_linear(
        B=3, n_out=2, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == 1
    assert tuple(out["C"].shape) == (3, 2)
    assert tuple(out["thresholds"].shape) == (3, 1)
    assert torch.allclose(
        out["thresholds"],
        torch.full((3, 1), 0.5, dtype=torch.float32),
    ), f"thresholds mismatch: {out['thresholds'].tolist()}"


def _test_build_margin_robust_threshold_is_negated_margin() -> None:  # pragma: no cover
    from act.front_end.specs import OutputSpec, OutKind

    out = OutputSpec(
        kind=OutKind.MARGIN_ROBUST,
        y_true=torch.tensor([1]),
        margin=torch.tensor(0.1),
    ).encode_linear(
        B=1, n_out=4, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == 3
    expected = torch.full((1, 3), -0.1, dtype=torch.float32)
    assert torch.allclose(out["thresholds"], expected), (
        f"thresholds should be -margin; got {out['thresholds'].tolist()}"
    )


def _test_build_range_interleaves_pm_e_rows() -> None:  # pragma: no cover
    from act.front_end.specs import OutputSpec, OutKind

    out = OutputSpec(
        kind=OutKind.RANGE,
        lb=torch.tensor([-1.0, -1.0, -1.0]),
        ub=torch.tensor([1.0, 1.0, 1.0]),
    ).encode_linear(
        B=2, n_out=3, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == 6
    expected = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    C_per_sample = out["C"].view(2, 6, 3)
    for b in range(2):
        assert torch.allclose(C_per_sample[b], expected), (
            f"sample {b}: rows mismatch.\n got={C_per_sample[b].tolist()}\n"
            f" want={expected.tolist()}"
        )


def _test_interval_margin_certification_shape() -> None:  # pragma: no cover
    # margin_max = sum_k (max(C_k,0)*ub_k + min(C_k,0)*lb_k) over [B*M, n_out];
    # per-sample CERTIFIED iff every M lane satisfies margin_max < threshold.
    from act.front_end.specs import OutputSpec, OutKind

    B, n_out = 2, 5
    output_lb = torch.zeros(B, n_out)
    output_ub = torch.ones(B, n_out)
    out = OutputSpec(
        kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0, 1])
    ).encode_linear(
        B=B, n_out=n_out, device=torch.device("cpu"), dtype=torch.float32,
    )
    M = out["M"]
    assert M == 4

    C = out["C"]
    C_pos = C.clamp(min=0)
    C_neg = C.clamp(max=0)
    lb_exp = output_lb.repeat_interleave(M, dim=0)
    ub_exp = output_ub.repeat_interleave(M, dim=0)
    margin_max = (C_pos * ub_exp + C_neg * lb_exp).sum(dim=-1)
    assert tuple(margin_max.shape) == (B * M,), (
        f"margin_max shape {tuple(margin_max.shape)} != (B*M,) == ({B * M},)"
    )

    cert_per_sample = (
        margin_max.view(B, M) < out["thresholds"]
    ).all(dim=-1)
    assert tuple(cert_per_sample.shape) == (B,), (
        f"per-sample cert shape {tuple(cert_per_sample.shape)} != ({B},)"
    )


def _make_dense_net_box_test(  # pragma: no cover
    B: int,
    n_in: int,
    n_out: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lb_in: torch.Tensor,
    ub_in: torch.Tensor,
    assert_params: Dict[str, Any],
):
    # assert_params is high-level (kind + y_true/margin/c/d/lb/ub); lift to
    # encoded form via OutputSpec.encode_linear to match the production
    # OutputSpecLayer.to_act_layers path.
    from act.back_end.core import Layer, Net
    from act.front_end.specs import OutputSpec

    in_v = list(range(n_in))
    out_v = list(range(n_in, n_in + n_out))

    spec_kwargs = {
        k: assert_params[k] for k in ("y_true", "margin", "c", "d", "lb", "ub")
        if k in assert_params
    }
    out_spec = OutputSpec(kind=assert_params["kind"], **spec_kwargs)
    encoded = out_spec.encode_linear(
        B=B, n_out=n_out, device=weight.device, dtype=weight.dtype,
    )

    layers = [
        Layer(
            id=0,
            kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(weight.dtype)},
            in_vars=[],
            out_vars=in_v,
        ),
        Layer(
            id=1,
            kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v,
            out_vars=in_v,
        ),
        Layer(
            id=2,
            kind=LayerKind.DENSE.value,
            params={
                "weight": weight,
                "in_features": n_in,
                "out_features": n_out,
                "weight_pos": weight.clamp(min=0),
                "weight_neg": weight.clamp(max=0),
                "bias": bias,
                "input_shape": (n_in,),
            },
            in_vars=in_v,
            out_vars=out_v,
        ),
        Layer(
            id=3,
            kind=LayerKind.ASSERT.value,
            params=encoded,
            in_vars=out_v,
            out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)



def _test_setup_and_solve_batch_b1_smoke() -> None:  # pragma: no cover
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_device, get_default_dtype

    device = get_default_device()
    dtype = get_default_dtype()
    weight = torch.ones(1, 1, device=device, dtype=dtype)
    bias = torch.zeros(1, device=device, dtype=dtype)
    lb_in = torch.full((1, 1), 1.0, device=device, dtype=dtype)
    ub_in = torch.full((1, 1), 2.0, device=device, dtype=dtype)
    net = _make_dense_net_box_test(
        B=1, n_in=1, n_out=1, weight=weight, bias=bias,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": OutKind.LINEAR_LE,
            "c": torch.ones(1, device=device, dtype=dtype),
            "d": torch.tensor(0.0, device=device, dtype=dtype),
        },
    )

    solution = setup_and_solve_batch(
        net,
        Bounds(lb_in.clone(), ub_in.clone()),
        TorchLPSolver(),
    )
    assert solution.statuses == (SolveStatus.SAT,), f"got {solution.statuses}"
    assert tuple(solution.max_viol.shape) == (1,)
    assert float(solution.max_viol[0].item()) <= 1e-4


def _test_setup_and_solve_batch_b_greater_than_1() -> None:  # pragma: no cover
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_device, get_default_dtype

    device = get_default_device()
    dtype = get_default_dtype()

    B = 4
    weight = torch.ones(1, 1, device=device, dtype=dtype)
    bias = torch.zeros(1, device=device, dtype=dtype)
    lb_in = torch.tensor([[1.0], [1.25], [1.5], [1.75]], device=device, dtype=dtype)
    ub_in = torch.tensor([[2.0], [2.25], [2.5], [2.75]], device=device, dtype=dtype)
    net = _make_dense_net_box_test(
        B=B, n_in=1, n_out=1, weight=weight, bias=bias,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": OutKind.LINEAR_LE,
            "c": torch.ones(1, device=device, dtype=dtype),
            "d": torch.tensor(0.0, device=device, dtype=dtype),
        },
    )

    solution = setup_and_solve_batch(
        net,
        Bounds(lb_in.clone(), ub_in.clone()),
        TorchLPSolver(),
    )

    assert solution.statuses == (SolveStatus.SAT,) * B, (
        f"expected {B} SAT statuses, got {solution.statuses}"
    )
    assert tuple(solution.x.shape) == (B, solution.x.shape[1]), (
        f"solution.x should retain leading batch B={B}, got "
        f"{tuple(solution.x.shape)}"
    )
    for i in range(B):
        assert float(solution.max_viol[i].item()) <= 1e-4, (
            f"batch lane {i}: max_viol "
            f"{float(solution.max_viol[i].item())} > 1e-4"
        )



def _test_verify_once_b3_all_certified() -> None:  # pragma: no cover
    # Zero DENSE -> abstract output is singleton {0}, well below d=10.
    # End-to-end check that the [B*M, n_out] cert pass folds to per-sample.
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()

    B, n_in, n_out = 3, 4, 2
    W = torch.zeros(n_out, n_in, device=device, dtype=dtype)
    b = torch.zeros(n_out, device=device, dtype=dtype)
    lb_in = torch.full((B, n_in), -1.0, device=device, dtype=dtype)
    ub_in = torch.full((B, n_in), 1.0, device=device, dtype=dtype)

    net = _make_dense_net_box_test(
        B=B, n_in=n_in, n_out=n_out, weight=W, bias=b,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": "LINEAR_LE",
            "c": torch.tensor([1.0, 1.0], device=device, dtype=dtype),
            "d": 10.0,
        },
    )

    results = verify_once(net)
    assert len(results) == B, f"expected {B} results, got {len(results)}"
    for i, r in enumerate(results):
        assert r.status == VerifyStatus.CERTIFIED, (
            f"sample {i}: expected CERTIFIED, got {r.status}"
        )


def _test_verify_once_b8_mixed_outcomes() -> None:  # pragma: no cover
    # 8 input boxes designed to produce CERT/FALS/UNK mix in one run,
    # proving the cert pass + concrete falsification operate sample-wise
    # rather than collapsing the batch.
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()

    B, n_in, n_out = 8, 2, 2
    W = torch.eye(n_out, device=device, dtype=dtype)
    b = torch.zeros(n_out, device=device, dtype=dtype)
    lb_in = torch.tensor(
        [
            [2.0, -2.0],
            [1.0, -2.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-2.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ],
        device=device, dtype=dtype,
    )
    ub_in = torch.tensor(
        [
            [3.0, -1.0],
            [2.0, -1.5],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.0, 0.0],
            [1.0, 1.0],
        ],
        device=device, dtype=dtype,
    )
    net = _make_dense_net_box_test(
        B=B, n_in=n_in, n_out=n_out, weight=W, bias=b,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": "TOP1_ROBUST",
            "y_true": torch.zeros(B, dtype=torch.long, device=device),
        },
    )

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    results = verify_once(net, model_fn=model_fn)
    assert len(results) == B, f"expected {B} results, got {len(results)}"

    valid = {
        VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN,
    }
    statuses = [r.status for r in results]
    assert all(s in valid for s in statuses), (
        f"unexpected status enum value in {statuses}"
    )
    assert any(s == VerifyStatus.CERTIFIED for s in statuses), (
        f"no CERTIFIED lane in {statuses}"
    )
    assert any(s == VerifyStatus.FALSIFIED for s in statuses), (
        f"no FALSIFIED lane in {statuses}"
    )
    assert any(s == VerifyStatus.UNKNOWN for s in statuses), (
        f"no UNKNOWN lane in {statuses}"
    )


def _test_verify_lp_batched_multi_b1() -> None:  # pragma: no cover
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.stats import VerifyStatus

    net = load_net_from_file(
        "act/back_end/examples/nets/layer_testing_top1_robust.json",
        target_device="cpu",
    )
    results = verify_lp_batched(net, TorchLPSolver, timelimit=1.0)
    valid = {VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN}
    assert len(results) == 1, f"expected one result, got {len(results)}"
    assert results[0].status in valid, f"unexpected status {results[0].status}"


def _test_verify_lp_batched_batch_b4() -> None:  # pragma: no cover
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()
    B = 4
    weight = torch.ones(1, 1, device=device, dtype=dtype)
    bias = torch.zeros(1, device=device, dtype=dtype)
    lb_in = torch.tensor([[1.0], [1.25], [1.5], [1.75]], device=device, dtype=dtype)
    ub_in = torch.tensor([[2.0], [2.25], [2.5], [2.75]], device=device, dtype=dtype)
    net = _make_dense_net_box_test(
        B=B, n_in=1, n_out=1, weight=weight, bias=bias,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": OutKind.LINEAR_LE,
            "c": torch.ones(1, device=device, dtype=dtype),
            "d": torch.tensor(0.0, device=device, dtype=dtype),
        },
    )

    results = verify_lp_batched(net, TorchLPSolver, timelimit=1.0)
    valid = {VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN}
    assert len(results) == B, f"expected {B} results, got {len(results)}"
    for i, result in enumerate(results):
        assert result.status in valid, f"lane {i}: unexpected status {result.status}"


_TESTS = [  # pragma: no cover
    _test_build_top1_robust_drops_y_true_row,
    _test_build_linear_le_threshold_is_d_unchanged,
    _test_build_margin_robust_threshold_is_negated_margin,
    _test_build_range_interleaves_pm_e_rows,
    _test_interval_margin_certification_shape,
    _test_setup_and_solve_batch_b1_smoke,
    _test_setup_and_solve_batch_b_greater_than_1,
    _test_verify_once_b3_all_certified,
    _test_verify_once_b8_mixed_outcomes,
    _test_verify_lp_batched_multi_b1,
    _test_verify_lp_batched_batch_b4,
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
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    # Pin device/dtype to CPU/float64 so hosts where CUDA is visible but
    # no kernel matches the runtime's compute capability don't raise on
    # the default GPU init path in act.util.device_manager.
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    print("Running verifier self-tests (act.back_end.verifier)\n")
    return run_all_tests()


if __name__ == "__main__":
    import sys

    sys.exit(main())
