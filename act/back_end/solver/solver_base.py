
#===- act/back_end/solver/solver_base.py - Base Solver Interface -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Base Solver Interface. Defines abstract base class and common interfaces
#   for constraint satisfaction problem solvers.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from typing import Optional

class SolveStatus:
    """Solver status codes (SAT/UNSAT terminology for verification)."""
    SAT = "SAT"              # Satisfiable - solution found
    UNSAT = "UNSAT"          # Unsatisfiable - no solution exists
    UNKNOWN = "UNKNOWN"      # Inconclusive (timeout/spurious/error)

class SolverCaps:
    def __init__(self, supports_gpu: bool = False, supports_csp: bool = True, supports_hz: bool = False):
        self.supports_gpu = supports_gpu
        self.supports_csp = supports_csp
        self.supports_hz = supports_hz

class Solver:
    """Abstract solver interface used by the exporter and verification pipeline."""

    # --- Capabilities / lifecycle ---
    def capabilities(self) -> SolverCaps:  # pragma: no cover - abstract
        return SolverCaps(False)

    # --- Domain-based bounds ---
    def compute_bounds(self, domain_obj) -> object:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- Batched API [BATCHED-API] ---
    def solve_batch(
        self,
        problem: "BatchLPProblem",
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":
        raise NotImplementedError(f"{type(self).__name__}.solve_batch")


# --- Batched API types [BATCHED-API] ---

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass(frozen=True)
class BatchLPProblem:
    """N independent linear programs sharing variable-id schema.

    All tensors have leading batch dim N. nvars is shared across N
    (caller aligns). Constraint counts m_eq, m_le are uniform across N
    via canonical row emission in cons_exportor (no per-N ragged shapes).

    Sparse storage convention:
        A_eq_blockdiag and A_le_blockdiag are 2-D torch.sparse_coo_tensors
        with shape (N*m_eq, N*nvars) and (N*m_le, N*nvars) respectively,
        constructed so that block (i, i) holds the i-th instance's
        constraint rows. This is the only sparse layout torch.sparse.mm
        handles efficiently as of PyTorch 2.x.
    """

    nvars: int
    m_eq: int
    m_le: int
    lb: torch.Tensor
    ub: torch.Tensor
    A_eq_blockdiag: torch.Tensor
    b_eq: torch.Tensor
    A_le_blockdiag: torch.Tensor
    b_le: torch.Tensor
    obj_c: torch.Tensor
    obj_const: torch.Tensor
    sense: str = "min"

    @property
    def N(self) -> int:
        return int(self.lb.shape[0])

    def __post_init__(self) -> None:
        N = self.N
        if self.lb.shape != (N, self.nvars):
            raise ValueError(f"lb shape {tuple(self.lb.shape)} != ({N}, {self.nvars})")
        if self.ub.shape != (N, self.nvars):
            raise ValueError(f"ub shape {tuple(self.ub.shape)} != ({N}, {self.nvars})")
        if self.b_eq.shape != (N, self.m_eq):
            raise ValueError(f"b_eq shape {tuple(self.b_eq.shape)} != ({N}, {self.m_eq})")
        if self.b_le.shape != (N, self.m_le):
            raise ValueError(f"b_le shape {tuple(self.b_le.shape)} != ({N}, {self.m_le})")
        if self.obj_c.shape != (N, self.nvars):
            raise ValueError(f"obj_c shape {tuple(self.obj_c.shape)} != ({N}, {self.nvars})")
        if self.obj_const.shape != (N,):
            raise ValueError(f"obj_const shape {tuple(self.obj_const.shape)} != ({N},)")
        if not self.A_eq_blockdiag.is_sparse:
            raise ValueError("A_eq_blockdiag must be sparse")
        if self.A_eq_blockdiag.shape != (N * self.m_eq, N * self.nvars):
            raise ValueError(
                f"A_eq_blockdiag shape {tuple(self.A_eq_blockdiag.shape)} "
                f"!= ({N * self.m_eq}, {N * self.nvars})"
            )
        if not self.A_le_blockdiag.is_sparse:
            raise ValueError("A_le_blockdiag must be sparse")
        if self.A_le_blockdiag.shape != (N * self.m_le, N * self.nvars):
            raise ValueError(
                f"A_le_blockdiag shape {tuple(self.A_le_blockdiag.shape)} "
                f"!= ({N * self.m_le}, {N * self.nvars})"
            )
        if self.sense not in ("min", "max"):
            raise ValueError(f"sense={self.sense!r}")


@dataclass(frozen=True)
class BatchLPSolution:
    """Per-N status + iterate.

    statuses[i] in {SAT, UNSAT, UNKNOWN}. SAT means x[i] is feasible
    within max_viol[i] <= tol_feas; iterate clamped to [lb, ub]. UNSAT
    means the backend proved infeasibility (only Gurobi N=1 produces
    UNSAT; TorchLPSolver cannot conclude UNSAT and uses UNKNOWN per
    soundness invariant — penalty-on-Adam cannot certify infeasibility).
    """

    statuses: Tuple[str, ...]
    x: torch.Tensor
    max_viol: torch.Tensor

    def __post_init__(self) -> None:
        N = int(self.x.shape[0])
        if len(self.statuses) != N:
            raise ValueError(f"statuses len {len(self.statuses)} != N={N}")
        if self.max_viol.shape != (N,):
            raise ValueError(f"max_viol shape {tuple(self.max_viol.shape)} != ({N},)")
        valid = {"SAT", "UNSAT", "UNKNOWN"}
        for i, s in enumerate(self.statuses):
            if s not in valid:
                raise ValueError(f"statuses[{i}]={s!r} not in {valid}")


# --- Self-tests ---


def _empty_blockdiag(N: int, m: int, nvars: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros(0),
        (N * m, N * nvars),
    )


def _problem(N: int, nvars: int, m_eq: int = 0, m_le: int = 0) -> BatchLPProblem:
    return BatchLPProblem(
        nvars=nvars,
        m_eq=m_eq,
        m_le=m_le,
        lb=torch.zeros(N, nvars),
        ub=torch.ones(N, nvars),
        A_eq_blockdiag=_empty_blockdiag(N, m_eq, nvars),
        b_eq=torch.zeros(N, m_eq),
        A_le_blockdiag=_empty_blockdiag(N, m_le, nvars),
        b_le=torch.zeros(N, m_le),
        obj_c=torch.zeros(N, nvars),
        obj_const=torch.zeros(N),
    )


def _test_batch_lp_problem_valid():
    p = _problem(N=2, nvars=3)
    assert p.N == 2
    assert p.sense == "min"


def _test_batch_lp_problem_with_constraints():
    p = _problem(N=4, nvars=5, m_eq=2, m_le=3)
    assert p.N == 4
    assert p.A_eq_blockdiag.shape == (8, 20)
    assert p.A_le_blockdiag.shape == (12, 20)


def _test_batch_lp_problem_invalid_shape_raises():
    N, nvars = 2, 3
    try:
        BatchLPProblem(
            nvars=nvars, m_eq=0, m_le=0,
            lb=torch.zeros(N, nvars + 1),
            ub=torch.ones(N, nvars),
            A_eq_blockdiag=_empty_blockdiag(N, 0, nvars),
            b_eq=torch.zeros(N, 0),
            A_le_blockdiag=_empty_blockdiag(N, 0, nvars),
            b_le=torch.zeros(N, 0),
            obj_c=torch.zeros(N, nvars), obj_const=torch.zeros(N),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad lb shape")


def _test_batch_lp_solution_valid():
    s = BatchLPSolution(
        statuses=("SAT", "UNKNOWN"),
        x=torch.zeros(2, 3),
        max_viol=torch.zeros(2),
    )
    assert len(s.statuses) == 2


def _test_batch_lp_solution_invalid_status_raises():
    try:
        BatchLPSolution(
            statuses=("SAT", "BOGUS"),
            x=torch.zeros(2, 3),
            max_viol=torch.zeros(2),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad status")


if __name__ == "__main__":
    import sys
    tests = [
        _test_batch_lp_problem_valid,
        _test_batch_lp_problem_with_constraints,
        _test_batch_lp_problem_invalid_shape_raises,
        _test_batch_lp_solution_valid,
        _test_batch_lp_solution_invalid_status_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
