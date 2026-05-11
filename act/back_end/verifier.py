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
# BaB-only architecture — setup_and_solve / add_negated_assert_to_solver:
#   Single-spec CSP path used by the BaB driver:
#   interval propagation -> export to solver -> add negated ASSERT
#   -> SAT/UNSAT. verify_once does not call this path.
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, *, model_fn=None) -> List[VerifyResult]
#       Pure-tensor batched single-shot verifier.
#   - setup_and_solve(net, input_bounds, solver, timelimit=None)
#       Single-spec CSP setup helper used by the BaB driver.
#   - add_negated_assert_to_solver / find_entry_layer_id / get_input_ids /
#     get_output_ids / gather_input_spec_layers / get_assert_layer /
#     seed_from_input_specs / add_all_input_specs (helpers).
#
# Notes:
#   * Spec-free verification: all constraints extracted from ACT Net layers.
#   * verify_once returns one VerifyResult per lane (len(result) == B).
#   * INPUT_SPEC constraints (including LIN_POLY) are propagated through
#     analyze(); they enter via add_all_input_specs into entry_fact.cons.
#     LIN_POLY constraints are not consumed by verify_once's interval
#     certification; they are preserved for the BaB / setup_and_solve path
#     which is solver-aware.

from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import torch

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet, Fact
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import validate_constraints

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind

# Verification types (canonical location: act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult

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
    for L in spec_layers:
        if L.params.get("kind") == InKind.BOX and "lb" in L.params and "ub" in L.params:
            return Bounds(L.params["lb"].clone(), L.params["ub"].clone())
    
    # LINF_BALL next
    for L in spec_layers:
        if L.params.get("kind") == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                return Bounds(L.params["lb"].clone(), L.params["ub"].clone())
            center = L.params.get("center")
            eps = L.params.get("eps")
            if center is not None and eps is not None:
                e = torch.tensor(eps)
                return Bounds(center - e, center + e)
    
    # LIN_POLY only -> error
    if any(L.params.get("kind") == InKind.LIN_POLY for L in spec_layers):
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
    exported to the solver via export_to_solver() in cons_exportor.py.
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

def _b1_scalar(val: Any, name: str) -> float:
    """Extract a Python scalar from a BaB-path ASSERT param.

    BaB MILP encoding is single-instance by contract: every high-level
    field is either a Python scalar (legacy form) or a ``torch.Tensor[B=1]``
    (post-``OutputSpec.encode_linear``). Tensors with ``B > 1`` are
    rejected because the existential encoding for TOP1 / MARGIN
    intrinsically requires a single ``y_true`` / ``margin``.
    """
    if isinstance(val, torch.Tensor):
        if val.numel() != 1:
            raise ValueError(
                f"BaB MILP requires B=1; param '{name}' has "
                f"numel={val.numel()}, shape={tuple(val.shape)}"
            )
        return float(val.item())
    return float(val)


def _b1_int(val: Any, name: str) -> int:
    """Same as ``_b1_scalar`` but returns int (for ``y_true``)."""
    if isinstance(val, torch.Tensor):
        if val.numel() != 1:
            raise ValueError(
                f"BaB MILP requires B=1; param '{name}' has "
                f"numel={val.numel()}, shape={tuple(val.shape)}"
            )
        return int(val.item())
    return int(val)


def _b1_vec(val: Any, name: str, expected_len: int) -> torch.Tensor:
    """Extract a ``[expected_len]`` 1-D tensor from a BaB-path ASSERT param.

    Accepts either ``Tensor[expected_len]`` (legacy) or ``Tensor[B=1, expected_len]``
    (post-encode_linear) — rejects ``B > 1``.
    """
    t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
    if t.dim() == 2:
        if t.shape[0] != 1:
            raise ValueError(
                f"BaB MILP requires B=1; param '{name}' has "
                f"shape={tuple(t.shape)}"
            )
        t = t[0]
    if t.dim() != 1 or t.shape[0] != expected_len:
        raise ValueError(
            f"BaB MILP: param '{name}' must reduce to [{expected_len}]; "
            f"got shape={tuple(t.shape)}"
        )
    return t


def add_negated_assert_to_solver(solver: Solver, out_ids: List[int], assert_layer) -> None:
    """Add the negation of ASSERT property as constraints to solver."""
    from act.back_end.cons_exportor import to_numpy
    k = assert_layer.params.get("kind")
    n_out = len(out_ids)

    if k == OutKind.LINEAR_LE:
        c_vec = _b1_vec(assert_layer.params["c"], "c", n_out)
        d = _b1_scalar(assert_layer.params["d"], "d")
        coeffs = list(to_numpy(c_vec))
        solver.add_lin_ge(out_ids, coeffs, d + 1e-6)

    elif k == OutKind.UNSAFE_LINEAR:
        c_raw = assert_layer.params["c"]
        c_t = c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)
        if c_t.dim() == 3:
            if c_t.shape[0] != 1:
                raise ValueError(
                    f"BaB MILP requires B=1; UNSAFE_LINEAR c has "
                    f"shape={tuple(c_t.shape)}"
                )
            c_t = c_t[0]
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(0)
        d_raw = assert_layer.params["d"]
        d_t = d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)
        if d_t.dim() == 2:
            if d_t.shape[0] != 1:
                raise ValueError(
                    f"BaB MILP requires B=1; UNSAFE_LINEAR d has "
                    f"shape={tuple(d_t.shape)}"
                )
            d_t = d_t[0]
        C = to_numpy(c_t)
        d_vec = to_numpy(d_t).reshape(-1)
        for i in range(C.shape[0]):
            row = list(C[i])
            solver.add_lin_le(out_ids, row, float(d_vec[i]) + 1e-6)

    elif k == OutKind.TOP1_ROBUST:
        t = _b1_int(assert_layer.params["y_true"], "y_true")
        v = solver.n
        solver.add_vars(1)
        for j, oj in enumerate(out_ids):
            if j != t:
                solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], 0.0)
        solver.add_lin_ge([v], [1.0], 0.0)

    elif k == OutKind.MARGIN_ROBUST:
        t = _b1_int(assert_layer.params["y_true"], "y_true")
        margin = _b1_scalar(assert_layer.params["margin"], "margin")
        v = solver.n
        solver.add_vars(1)
        for j, oj in enumerate(out_ids):
            if j != t:
                solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], -margin)
        solver.add_lin_ge([v], [1.0], 0.0)

    elif k == OutKind.RANGE:
        lb_raw = assert_layer.params.get("lb", None)
        ub_raw = assert_layer.params.get("ub", None)
        if lb_raw is None and ub_raw is None:
            raise ValueError("RANGE assert requires lb and/or ub.")

        lb = to_numpy(_b1_vec(lb_raw, "lb", n_out)) if lb_raw is not None else None
        ub = to_numpy(_b1_vec(ub_raw, "ub", n_out)) if ub_raw is not None else None

        v = solver.n
        solver.add_vars(1)
        v_max_terms: List[float] = []

        v_max = max(v_max_terms) if v_max_terms else 1e6
        if (not np.isfinite(v_max)) or v_max < 1e-3:
            v_max = 1e6

        solver.add_lin_ge([v], [1.0], 0.0)
        solver.add_lin_ge([v], [-1.0], -v_max)

        for i, yi in enumerate(out_ids):
            if lb is not None: solver.add_lin_ge([v, yi], [1.0, 1.0], float(lb[i]))
            if ub is not None: solver.add_lin_ge([v, yi], [1.0, -1.0], float(-ub[i]))

# -----------------------------------------------------------------------------
# Core solver workflow (shared by verify_once and BaB)
# -----------------------------------------------------------------------------

def _reshape_input_bounds(input_bounds: Bounds, net) -> Bounds:
    """Reshape 1D bounds to the network's INPUT shape.

    BaB sub-problems arrive flat ([D]); analyze needs [B, *input_shape].
    Bounds already 2-D+ pass through unchanged.
    """
    if input_bounds.lb.dim() >= 2:
        return input_bounds
    entry_id = find_entry_layer_id(net)
    target = net.by_id[entry_id].params.get("shape")
    if target is None:
        raise ValueError(
            f"_reshape_input_bounds: 1-D input_bounds requires "
            f"INPUT.params['shape'] on the entry layer, but layer "
            f"{entry_id} has no 'shape' param."
        )
    target_t = tuple(int(d) for d in target)
    prod = 1
    for d in target_t:
        prod *= d
    flat = input_bounds.lb
    if flat.numel() != prod:
        raise ValueError(
            f"_reshape_input_bounds: 1-D input_bounds numel "
            f"{flat.numel()} does not match prod(INPUT.shape)={prod} "
            f"(shape={target_t})."
        )
    return Bounds(
        lb=input_bounds.lb.reshape(*target_t).contiguous(),
        ub=input_bounds.ub.reshape(*target_t).contiguous(),
    )


@torch.no_grad()
def setup_and_solve(
    net,
    input_bounds: Bounds,
    solver: Solver,
    timelimit: Optional[float] = None
) -> tuple[str, Optional[np.ndarray], Dict[str, Any]]:
    """
    Core verification workflow: setup constraints and solve.
    
    This function encapsulates the common verification pattern:
    1. Extract network structure (entry layer, input/output IDs, specs)
    2. Create entry_fact with input_bounds and all INPUT_SPEC constraints
    3. Run abstract interpretation (analyze)
    4. Export constraints to solver
    5. Add negated ASSERT property
    6. Solve and return status + counterexample (if found)
    
    Args:
        net: ACT network
        input_bounds: Input region bounds (seed box or refinement box)
        solver: Solver instance
        timelimit: Optional timeout in seconds
    
    Returns:
        Tuple of (status, counterexample_input, stats):
        - status: SolveStatus.SAT/UNSAT/UNKNOWN
        - counterexample_input: np.ndarray if SAT, else None
        - stats: Dict with metadata (ncons, status, etc.)
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_solver
    
    # Extract network structure
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    input_bounds = _reshape_input_bounds(input_bounds, net)

    # Create entry_fact with ALL input constraints
    entry_fact = Fact(bounds=input_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    
    # Analyze with full input specification (propagates constraints)
    before, after, globalC = analyze(net, entry_id, entry_fact)
    
    # Validate constraints (validation runs if enabled, logging only if debug_tf also enabled)
    validate_constraints(globalC, after, net)
    
    # Export all constraints to solver (including LIN_POLY)
    export_to_solver(globalC, solver, objective=None, sense="min")
    add_negated_assert_to_solver(solver, output_ids, assert_layer)
    
    # Solve (feasibility check only)
    solver.set_objective_linear([], [], 0.0, sense="min")
    solver.optimize(timelimit)
    
    # Extract result
    st = solver.status()
    ce_input = None
    if st == SolveStatus.SAT and solver.has_solution():
        ce_input = solver.get_values(input_ids)
    
    stats = {"status": st, "ncons": len(globalC)}
    return st, ce_input, stats


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
    # at FE construction time). No runtime kind-dispatch / encoding happens
    # in verify_once — the encoding lives in act/front_end/specs.py.
    C = assert_layer.params["C"].to(device=device, dtype=dtype)
    thresholds = assert_layer.params["thresholds"].to(device=device, dtype=dtype)
    M = int(assert_layer.params["M"])
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
        # Cert uses strict <; falsification uses >=. A sample is FALSIFIED
        # iff ANY of its M lanes' concrete margin meets-or-exceeds threshold.
        falsified = (~certified) & (
            (concrete_violation >= thresholds).any(dim=-1)
        )
        if falsified.any():
            x_center_cpu = x_center.detach().cpu()
            for i in range(B):
                if bool(falsified[i].item()):
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


def _test_build_top1_robust_drops_y_true_row() -> None:
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


def _test_build_linear_le_threshold_is_d_unchanged() -> None:
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


def _test_build_margin_robust_threshold_is_negated_margin() -> None:
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


def _test_build_range_interleaves_pm_e_rows() -> None:
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


def _test_interval_margin_certification_shape() -> None:
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


def _make_dense_net_box_test(
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


def _test_verify_once_b3_all_certified() -> None:
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


def _test_verify_once_b8_mixed_outcomes() -> None:
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


_TESTS = [
    _test_build_top1_robust_drops_y_true_row,
    _test_build_linear_le_threshold_is_d_unchanged,
    _test_build_margin_robust_threshold_is_negated_margin,
    _test_build_range_interleaves_pm_e_rows,
    _test_interval_margin_certification_shape,
    _test_verify_once_b3_all_certified,
    _test_verify_once_b8_mixed_outcomes,
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

