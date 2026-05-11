#===- act/front_end/specs.py - Specification Data Types ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Defines InputSpec and OutputSpec data structures for verification
#   specifications including safety, robustness, and constraint types.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

class InKind:
    BOX = "BOX"
    LINF_BALL = "LINF_BALL"
    LIN_POLY = "LIN_POLY"

@dataclass
class InputSpec:
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[torch.Tensor] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Ensure all numeric fields are tensors for architecture."""
        # Convert eps (scalar → 1-D tensor)
        if self.eps is not None and not isinstance(self.eps, torch.Tensor):
            self.eps = torch.tensor([float(self.eps)])
        
        # Convert d (scalar → 1-D tensor)
        if hasattr(self, 'd') and self.d is not None and not isinstance(self.d, torch.Tensor):
            self.d = torch.tensor([float(self.d)])
        
        # Convert lb, ub, center (list or scalar → tensor)
        for field in ['lb', 'ub', 'center']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))
                else:
                    setattr(self, field, torch.tensor([float(val)]))
        
        # Convert A, b (list → tensor, keep None as is)
        for field in ['A', 'b']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))

class OutKind:
    LINEAR_LE   = "LINEAR_LE"
    TOP1_ROBUST = "TOP1_ROBUST"
    MARGIN_ROBUST = "MARGIN_ROBUST"
    RANGE = "RANGE"
    UNSAFE_LINEAR = "UNSAFE_LINEAR"

@dataclass
class OutputSpec:
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[torch.Tensor] = None
    y_true: Optional[torch.Tensor] = None
    margin: Optional[torch.Tensor] = None
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Ensure all numeric fields are tensors for batch-native architecture."""
        # Convert y_true (int/list → tensor)
        if self.y_true is not None and not isinstance(self.y_true, torch.Tensor):
            if isinstance(self.y_true, (list, tuple)):
                self.y_true = torch.tensor(self.y_true, dtype=torch.int64)
            else:
                self.y_true = torch.tensor([int(self.y_true)], dtype=torch.int64)
        
        # Convert margin (scalar → 1-D tensor)
        if self.margin is not None and not isinstance(self.margin, torch.Tensor):
            self.margin = torch.tensor([float(self.margin)])
        
        # Convert d (scalar → 1-D tensor)
        if self.d is not None and not isinstance(self.d, torch.Tensor):
            self.d = torch.tensor([float(self.d)])
        
        # Convert c, lb, ub (list or scalar → tensor)
        for field in ['c', 'lb', 'ub']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))
                else:
                    setattr(self, field, torch.tensor([float(val)]))

    def encode_linear(
        self,
        B: int,
        n_out: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """Produce ASSERT-layer params with both high-level fields (BaB) and
        pre-encoded linear form (verify_once).

        Output dict layout (all numeric values are ``torch.Tensor`` with a
        leading batch axis of size B; B=1 is the trivial case):

            {
              "kind":       str,                # for BaB MILP dispatch
              "y_true":     Tensor[B] long,     # TOP1_ROBUST, MARGIN_ROBUST
              "margin":     Tensor[B] float,    # MARGIN_ROBUST only
              "c":          Tensor[B, n_out],   # LINEAR_LE
                            Tensor[B, N, n_out],# UNSAFE_LINEAR
              "d":          Tensor[B],          # LINEAR_LE
                            Tensor[B, N],       # UNSAFE_LINEAR
              "lb":         Tensor[B, n_out],   # RANGE (if lb given)
              "ub":         Tensor[B, n_out],   # RANGE (if ub given)
              "C":          Tensor[B*M, n_out], # for verify_once
              "thresholds": Tensor[B, M],       # for verify_once
              "M":          int,                # for verify_once
            }

        Encoding orientation is "violation form": for each row
        ``C[b*M+j, :]``, the lane ``(b, j)`` is CERTIFIED iff
        ``margin_max(C[b*M+j] @ y) < thresholds[b, j]``. This method is the
        single source of truth for the five-kind row layouts consumed by
        ``verifier.verify_once`` (back-end).

        Args:
            B: batch size. Must be >= 1.
            n_out: number of output classes / output dimension.
                For TOP1/MARGIN this is also K (num_classes).
            device: target tensor device.
            dtype: target floating tensor dtype (``y_true`` uses ``torch.long``
                regardless).

        Returns:
            A dict with the layout above. All tensors carry the leading B
            axis. ``M`` is an int — kind-dependent (1, K-1, N, n_out, or
            2*n_out).

        Raises:
            ValueError: if required dataclass fields are missing or
                shape-incompatible with ``B`` / ``n_out``.
            NotImplementedError: if ``self.kind`` is not one of the five
                supported kinds.
        """
        params: Dict[str, Any] = {"kind": self.kind}

        if self.kind == OutKind.LINEAR_LE:
            if self.c is None or self.d is None:
                raise ValueError("LINEAR_LE requires both c and d")
            c_vec = self.c.to(device=device, dtype=dtype).flatten()
            if c_vec.shape[0] != n_out:
                raise ValueError(
                    f"LINEAR_LE: c length {c_vec.shape[0]} != n_out {n_out}"
                )
            d_t = self.d.to(device=device, dtype=dtype).flatten()
            if d_t.numel() != 1:
                raise ValueError(
                    f"LINEAR_LE: d must be scalar (got numel={d_t.numel()})"
                )
            d_scalar = float(d_t.item())
            c_batched = c_vec.unsqueeze(0).expand(B, -1).contiguous()
            d_batched = torch.full((B,), d_scalar, device=device, dtype=dtype)
            params["c"] = c_batched
            params["d"] = d_batched
            c_rows = c_batched
            thresholds = d_batched.unsqueeze(1)
            m_specs = 1

        elif self.kind == OutKind.UNSAFE_LINEAR:
            if self.c is None or self.d is None:
                raise ValueError("UNSAFE_LINEAR requires both c and d")
            c_mat = self.c.to(device=device, dtype=dtype)
            if c_mat.dim() == 1:
                c_mat = c_mat.unsqueeze(0)
            N = c_mat.shape[0]
            if c_mat.shape[1] != n_out:
                raise ValueError(
                    f"UNSAFE_LINEAR: c cols {c_mat.shape[1]} != n_out {n_out}"
                )
            d_vec = self.d.to(device=device, dtype=dtype).flatten()
            if d_vec.shape[0] != N:
                raise ValueError(
                    f"UNSAFE_LINEAR: d length {d_vec.shape[0]} != N {N}"
                )
            params["c"] = c_mat.unsqueeze(0).expand(B, -1, -1).contiguous()
            params["d"] = d_vec.unsqueeze(0).expand(B, -1).contiguous()
            c_rows = c_mat.unsqueeze(0).expand(B, -1, -1).reshape(
                B * N, n_out
            ).contiguous()
            thresholds = d_vec.unsqueeze(0).expand(B, -1).contiguous()
            m_specs = N

        elif self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if self.y_true is None:
                raise ValueError(f"{self.kind} requires y_true")
            K = n_out
            if K < 2:
                raise ValueError(
                    f"{self.kind}: requires K >= 2 classes, got K={K}"
                )
            y_true_t = self.y_true.to(
                device=device, dtype=torch.long
            ).reshape(-1)
            if y_true_t.numel() == 1 and B > 1:
                y_true_t = y_true_t.repeat(B)
            if y_true_t.numel() != B:
                raise ValueError(
                    f"{self.kind}: y_true length {y_true_t.numel()} != B {B}"
                )
            if (y_true_t < 0).any() or (y_true_t >= K).any():
                raise ValueError(
                    f"{self.kind}: y_true contains out-of-range index "
                    f"(must lie in [0, K)); y_true={y_true_t.tolist()}, K={K}"
                )
            params["y_true"] = y_true_t
            # TOP1/MARGIN share row structure: drop the y_true row entirely
            # (m_specs = K-1) rather than masking with an active_mask.
            m_specs = K - 1
            j_full = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
            one_hot_y = torch.nn.functional.one_hot(
                y_true_t, num_classes=K
            ).bool()
            keep_mask = ~one_hot_y
            j_kept = j_full[keep_mask].view(B, m_specs)
            eye_K = torch.eye(K, device=device, dtype=dtype)
            e_true = eye_K[y_true_t]
            e_others = eye_K[j_kept]
            # Violation form: row = e_j - e_{y_true}.
            c_per_sample = e_others - e_true.unsqueeze(1)
            c_rows = c_per_sample.reshape(B * m_specs, K).contiguous()
            if self.kind == OutKind.TOP1_ROBUST:
                thresholds = torch.zeros(
                    B, m_specs, device=device, dtype=dtype
                )
            else:
                if self.margin is None:
                    raise ValueError(
                        "MARGIN_ROBUST requires margin; use TOP1_ROBUST for "
                        "zero-margin semantics."
                    )
                margin_t = self.margin.to(
                    device=device, dtype=dtype
                ).reshape(-1)
                if margin_t.numel() == 1 and B > 1:
                    margin_t = margin_t.repeat(B)
                if margin_t.numel() != B:
                    raise ValueError(
                        f"MARGIN_ROBUST: margin length {margin_t.numel()} "
                        f"!= 1 or B {B}"
                    )
                params["margin"] = margin_t
                thresholds = (-margin_t).unsqueeze(1).expand(
                    B, m_specs
                ).contiguous()

        elif self.kind == OutKind.RANGE:
            if self.lb is None and self.ub is None:
                raise ValueError("RANGE requires lb and/or ub")
            eye = torch.eye(n_out, device=device, dtype=dtype)
            rows: List[torch.Tensor] = []
            thresh_rows: List[torch.Tensor] = []
            lb_vec: Optional[torch.Tensor] = None
            ub_vec: Optional[torch.Tensor] = None
            if self.lb is not None:
                lb_vec = self.lb.to(device=device, dtype=dtype).flatten()
                if lb_vec.shape[0] != n_out:
                    raise ValueError(
                        f"RANGE: lb length {lb_vec.shape[0]} != n_out {n_out}"
                    )
                rows.append(-eye)
                thresh_rows.append(-lb_vec)
            if self.ub is not None:
                ub_vec = self.ub.to(device=device, dtype=dtype).flatten()
                if ub_vec.shape[0] != n_out:
                    raise ValueError(
                        f"RANGE: ub length {ub_vec.shape[0]} != n_out {n_out}"
                    )
                rows.append(eye)
                thresh_rows.append(ub_vec)
            both_sides = lb_vec is not None and ub_vec is not None
            if both_sides:
                # Interleave [-e_0, +e_0, -e_1, +e_1, ...].
                stacked = torch.stack(
                    [rows[0], rows[1]], dim=1
                ).reshape(2 * n_out, n_out)
                thresh_stacked = torch.stack(
                    [thresh_rows[0], thresh_rows[1]], dim=1
                ).reshape(2 * n_out)
            else:
                stacked = rows[0]
                thresh_stacked = thresh_rows[0]
            m_specs = 2 * n_out if both_sides else n_out
            c_rows = stacked.unsqueeze(0).expand(B, -1, -1).reshape(
                B * m_specs, n_out
            ).contiguous()
            thresholds = thresh_stacked.unsqueeze(0).expand(
                B, -1
            ).contiguous()
            if lb_vec is not None:
                params["lb"] = lb_vec.unsqueeze(0).expand(
                    B, -1
                ).contiguous()
            if ub_vec is not None:
                params["ub"] = ub_vec.unsqueeze(0).expand(
                    B, -1
                ).contiguous()

        else:
            raise NotImplementedError(
                f"Unsupported ASSERT kind: {self.kind!r}. Supported: "
                f"LINEAR_LE, UNSAFE_LINEAR, TOP1_ROBUST, MARGIN_ROBUST, "
                f"RANGE."
            )

        assert c_rows.shape == (B * m_specs, n_out), (
            f"C.shape={tuple(c_rows.shape)} != ({B * m_specs}, {n_out})"
        )
        assert thresholds.shape == (B, m_specs), (
            f"thresholds.shape={tuple(thresholds.shape)} != ({B}, {m_specs})"
        )
        params["C"] = c_rows
        params["thresholds"] = thresholds
        params["M"] = m_specs
        return params
