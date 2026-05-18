#===- act/back_end/solver/solver_dual.py - Dual Bounds Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualSolver: Wong-Kolter / CROWN-style certified lower-bound solver.
# STRICT batched API ([B, *shape] only). Raises ValueError on 1-D input.
# Mirrors HZSolver precedent in solver_hz.py.
#===---------------------------------------------------------------------===#
# pyright: reportMissingImports=false

from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast
from act.back_end.core import Bounds, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_base import Solver, SolverCaps
from act.front_end.specs import OutputSpec, OutKind
from act.util.device_manager import get_default_device, get_default_dtype
from act.util.stats import SpecBatchResult

if TYPE_CHECKING:
    from act.back_end.dual_tf.dual_tf import DualTF


def expand_bounds_dict(bounds_dict: Dict[int, Bounds], M: int) -> Dict[int, Bounds]:
    """Expand each batched Bounds entry from [B, *shape] to [B*M, *shape].

    repeat_interleave aligns with row b*M+j sharing sample b's bounds. All
    entries must already be batched (lb.dim() >= 2). M=1 returns the dict
    unchanged.
    """
    if M <= 0:
        raise ValueError(f"expand_bounds_dict: M must be positive, got {M}")
    if M == 1:
        return dict(bounds_dict)
    out: Dict[int, Bounds] = {}
    for lid, bounds in bounds_dict.items():
        if bounds.lb.dim() < 2:
            raise ValueError(
                f"expand_bounds_dict: layer {lid} bounds must be batched "
                f"[B, *shape], got dim={bounds.lb.dim()} shape={tuple(bounds.lb.shape)}"
            )
        out[lid] = Bounds(
            lb=bounds.lb.repeat_interleave(M, dim=0),
            ub=bounds.ub.repeat_interleave(M, dim=0),
        )
    return out


def _reverse_topological_sort(net: Net) -> List[int]:
    """Kahn's algorithm on net.succs.

    Returns layer IDs in reverse-topological order: every layer appears
    after all its successors.

    Raises:
        ValueError: If the graph contains a cycle or disconnected layers.
    """
    in_deg: Dict[int, int] = {layer.id: len(net.succs.get(layer.id, [])) for layer in net.layers}
    queue: List[int] = [lid for lid, degree in in_deg.items() if degree == 0]
    order: List[int] = []
    while queue:
        lid = queue.pop(0)
        order.append(lid)
        for pred in net.preds.get(lid, []):
            in_deg[pred] -= 1
            if in_deg[pred] == 0:
                queue.append(pred)
    if len(order) != len(net.layers):
        raise ValueError(
            f"DualSolver: graph has cycle or disconnected layers "
            f"({len(order)}/{len(net.layers)} sorted)"
        )
    return order


class DualSolver(Solver):
    """Dual (Wong-Kolter) certified bounds solver. Strict [B, *shape] API."""

    _AFFINE_CONTRIB_KINDS = {
        LayerKind.DENSE.value,
        LayerKind.CONV2D.value,
        "BIAS",
        "BN",
        "ADD",
    }

    def __init__(self, tf: "DualTF", n_iters: int = 0):
        self.tf = tf
        self.n_iters = n_iters
        self._last_bounds: Optional[Bounds] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    def compute_bound(self, net: Net, bounds_dict: Dict[int, Bounds],
                      c: torch.Tensor, return_sce: bool = False,
                      enable_grad: bool = False
                      ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        ν is propagated backward through a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).
        Args:
            c: Tensor[B, num_classes] — REQUIRED batched. Raises ValueError if 1-D.
            return_sce: if True, also return per-sample concrete input extremum.
            enable_grad: if True, allow gradients to flow through the computation
                         (for robust training). Default False (inference/verification).
        Returns:
            Tensor[B] or (Tensor[B], Tensor[B, *in_shape]) when return_sce=True.
        """
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B, num_classes], got shape {tuple(c.shape)}. "
                "Use c.unsqueeze(0) for single instance.")
        with torch.set_grad_enabled(enable_grad):
            assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
            device, dtype = get_default_device(), get_default_dtype()
            if c.dtype != dtype or c.device != device:
                c = c.to(device=device, dtype=dtype)
            B = c.shape[0]

            for _ in range(self.n_iters):
                pass

            assert_layer = None
            for layer in net.layers:
                k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
                if k == LayerKind.ASSERT.value:
                    assert_layer = layer
                    break
            if assert_layer is None:
                raise ValueError("DualSolver.compute_bound: net has no ASSERT layer")

            assert_preds = net.preds.get(assert_layer.id, [])
            if len(assert_preds) != 1:
                raise ValueError(
                    f"DualSolver.compute_bound: ASSERT layer {assert_layer.id} must have "
                    f"exactly 1 predecessor, got {len(assert_preds)}"
                )

            output_lid = assert_preds[0]
            nu_accum: Dict[int, torch.Tensor] = {output_lid: c.clone()}
            obj = torch.zeros(B, dtype=c.dtype, device=c.device)

            topo_order = _reverse_topological_sort(net)
            registry = self.tf._BACKWARD_REGISTRY

            for lid in topo_order:
                layer = net.by_id[lid]
                k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind

                if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
                    continue

                if lid not in nu_accum:
                    continue

                nu_here = nu_accum.pop(lid)
                handler = registry.get(k)
                if handler is None:
                    raise ValueError(
                        f"DualSolver.compute_bound: unknown layer kind '{k}' at layer {lid}; "
                        f"soundness requires explicit backward handler. "
                        f"Supported kinds: {sorted(registry.keys())}"
                    )

                preds = list(net.preds.get(lid, []))
                pred_nus, contrib = handler(layer, nu_here, bounds_dict, preds)

                if len(pred_nus) != len(preds):
                    raise ValueError(
                        f"handler {k} at layer {lid} returned {len(pred_nus)} pred_nus, "
                        f"expected {len(preds)}"
                    )
                if contrib.shape != (B,):
                    raise ValueError(
                        f"handler {k} at layer {lid} contrib shape {tuple(contrib.shape)}, "
                        f"expected ({B},)"
                    )

                if k in self._AFFINE_CONTRIB_KINDS:
                    contrib = -contrib

                obj = obj + contrib
                for pred_id, pred_nu in zip(preds, pred_nus):
                    if pred_id in nu_accum:
                        nu_accum[pred_id] = nu_accum[pred_id] + pred_nu
                    else:
                        nu_accum[pred_id] = pred_nu.clone()

            input_lid = self._find_input_layer_id(net)
            if input_lid is None:
                return (obj, None) if return_sce else obj

            nu_final = nu_accum.get(input_lid)
            if nu_final is None:
                return (obj, None) if return_sce else obj

            input_contrib, sce = self._input_contribution_from_nu(
                net,
                input_lid,
                nu_final,
                bounds_dict,
                return_sce=return_sce,
                enable_grad=enable_grad,
            )
            obj = obj + input_contrib
            return (obj, sce) if return_sce else obj

    def evaluate_spec(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        out_spec: OutputSpec,
        num_classes: Optional[int] = None,
        chunk_size: Optional[int] = None,
        enable_grad: bool = False,
    ) -> SpecBatchResult:
        """Dual bound evaluation for any OutputSpec, using
        ``OutputSpec.encode_linear`` as the single source of truth.

        Strategy: ``encode_linear`` produces (C, thresholds) in UB-cert
        form (CERTIFIED iff ``UB(C @ y) < threshold``). DualSolver returns
        LB(C @ y) from ``compute_bound``. Equivalence via sign flip: pass
        ``-C`` to compute_bound and compare against ``-threshold``; slack
        ``>= 0`` means certified.

        Args:
            net: ACT Net with ASSERT layer.
            bounds_dict: layer bounds from forward analysis. MUST contain
                batched bounds for all relevant layers including the ASSERT
                predecessor.
            out_spec: the property to evaluate. For TOP1/MARGIN robust kinds,
                out_spec.y_true must be populated by the caller.
            num_classes: K; required for TOP1_ROBUST / MARGIN_ROBUST.
            chunk_size: if set and M > chunk_size, process specs in chunks of
                chunk_size at a time (memory-saving for large K).
            enable_grad: if True, allow gradient flow (e.g. for Adam).

        Returns:
            SpecBatchResult with margins/slack/active_mask/certified tensors.

        Raises:
            ValueError: if net lacks ASSERT layer, ASSERT has != 1 predecessor,
                or the output layer's bounds are missing / unbatched.
            NotImplementedError: if out_spec.kind == UNSAFE_LINEAR (EXISTS
                quantifier not supported by sound dual certificate).
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        dtype = sample.lb.dtype
        if sample.lb.dim() < 2:
            raise ValueError(
                "DualSolver.evaluate_spec: bounds_dict entries must be batched "
                f"[B, *shape]; got dim={sample.lb.dim()}"
            )
        B = sample.lb.shape[0]

        assert_layer = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                assert_layer = layer
                break
        if assert_layer is None:
            raise ValueError("DualSolver.evaluate_spec: net has no ASSERT layer")
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer must have exactly 1 predecessor, got {len(assert_preds)}"
            )
        output_lid = assert_preds[0]
        if output_lid not in bounds_dict:
            raise ValueError(
                f"DualSolver.evaluate_spec: bounds_dict missing output layer "
                f"{output_lid} (ASSERT predecessor); run forward analysis first."
            )
        out_bounds = bounds_dict[output_lid]
        if out_bounds.lb.dim() < 2:
            raise ValueError(
                f"DualSolver.evaluate_spec: output layer {output_lid} bounds "
                f"must be batched; got dim={out_bounds.lb.dim()}"
            )
        n_out = int(out_bounds.lb.flatten(start_dim=1).shape[-1])

        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            raise NotImplementedError(
                "DualSolver: UNSAFE_LINEAR uses EXISTS semantics (any-row "
                "certification) not supported by sound dual lower bounds. "
                "Use the LP/Gurobi path for UNSAFE_LINEAR specs."
            )

        fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
        C_neg = -fe_params["C"].contiguous()
        thresholds_neg = -fe_params["thresholds"].contiguous()
        M = int(fe_params["M"])
        active_mask = torch.ones(B, M, dtype=torch.bool, device=device)

        with torch.set_grad_enabled(enable_grad):
            if chunk_size is None or M <= chunk_size:
                bounds_k = expand_bounds_dict(bounds_dict, M)
                margins_flat = self.compute_bound(
                    net, bounds_k, C_neg, enable_grad=enable_grad,
                )
            else:
                margins_flat = self._chunked_eval(
                    net, bounds_dict, C_neg, B, M, n_out, chunk_size, enable_grad,
                )

            if isinstance(margins_flat, tuple):
                margins_flat = margins_flat[0]

            margins = margins_flat.view(B, M)
            slack = margins - thresholds_neg
            violations = (slack < 0) & active_mask
            certified = ~violations.any(dim=-1)

        return SpecBatchResult(
            margins=margins,
            slack=slack,
            active_mask=active_mask,
            certified=certified,
        )

    def _chunked_eval(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        C_neg: torch.Tensor, B: int, M: int, n_out: int,
        chunk_size: int, enable_grad: bool,
    ) -> torch.Tensor:
        """Evaluate sign-flipped C in chunks along the M dimension.

        For large M (e.g. CIFAR-100 K=100), trades time for memory by
        processing chunk_size specs per sample at a time.
        """
        C_view = C_neg.view(B, M, n_out)
        chunks: List[torch.Tensor] = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            m_chunk = end - start
            C_chunk = C_view[:, start:end, :].reshape(B * m_chunk, n_out).contiguous()
            bounds_chunk = expand_bounds_dict(bounds_dict, m_chunk)
            margins_chunk = self.compute_bound(
                net, bounds_chunk, C_chunk, enable_grad=enable_grad,
            )
            if isinstance(margins_chunk, tuple):
                margins_chunk = margins_chunk[0]
            chunks.append(margins_chunk.view(B, m_chunk))
        return torch.cat(chunks, dim=1).reshape(B * M)

    def compute_robust_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        y_true: Union[int, torch.Tensor], num_classes: int,
        margin: float = 0.0,
        return_full: bool = False,
        enable_grad: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], SpecBatchResult]:
        """Dual certified robust bound for classification (top-1 or margin).

        Unified via evaluate_spec(). Retained as a first-class API for robust
        training loops and existing verification callers.

        Args:
            net: the ACT Net with an ASSERT layer.
            bounds_dict: layer bounds from forward analysis.
            y_true: [B] true class labels, or scalar for uniform label.
            num_classes: K (output dim of network's ASSERT predecessor).
            margin: if > 0 use MARGIN_ROBUST semantics (require y_t - y_j >= margin);
                    else use TOP1_ROBUST (require y_t - y_j >= 0).
            return_full: if True, return the full SpecBatchResult (has per-class
                         [B, K] margins useful for training losses). If False,
                         return legacy tuple (min_slack: Tensor[B], certified: Tensor[B] bool).
            enable_grad: if True, allow gradients to flow through the computation
                         (for robust training). Default False (inference/verification).

        Returns:
            SpecBatchResult if return_full else (Tensor[B], Tensor[B] bool).
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        if isinstance(y_true, int):
            B = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1
            y_true_t = torch.full((B,), y_true, dtype=torch.long, device=device)
        else:
            y_true_t = y_true.to(device=device, dtype=torch.long)

        kind = OutKind.MARGIN_ROBUST if margin > 0 else OutKind.TOP1_ROBUST
        out_spec = OutputSpec(
            kind=kind,
            y_true=y_true_t,
            margin=margin if margin > 0 else None,
        )
        result = self.evaluate_spec(
            net, bounds_dict, out_spec,
            num_classes=num_classes,
            enable_grad=enable_grad,
        )
        if return_full:
            return result
        return result.min_slack, result.certified

    def _find_input_layer_id(self, net: Net) -> Optional[int]:
        """Return the INPUT_SPEC layer id if present, else INPUT's id, else None."""
        input_spec_id = None
        input_id = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.INPUT_SPEC.value:
                input_spec_id = layer.id
            elif k == LayerKind.INPUT.value:
                input_id = layer.id
        return input_spec_id if input_spec_id is not None else input_id

    def _input_contribution_from_nu(self, net: Net, input_lid: int,
                                    nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                                    return_sce: bool = False,
                                    enable_grad: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute lb·[nu]_+ + ub·[nu]_- over the input box (batched)."""
        with torch.set_grad_enabled(enable_grad):
            B = nu.shape[0]
            input_layer = net.by_id[input_lid]

            bounds = bounds_dict.get(input_lid)
            if bounds is None:
                if "lb" in input_layer.params and "ub" in input_layer.params:
                    lb = cast(torch.Tensor, input_layer.params["lb"])
                    ub = cast(torch.Tensor, input_layer.params["ub"])
                else:
                    raise ValueError(
                        f"_input_contribution_from_nu: input layer {input_lid} has no "
                        f"bounds in bounds_dict and no lb/ub params"
                    )
            else:
                lb = bounds.lb
                ub = bounds.ub

            orig_shape = lb.shape
            if lb.dim() < 2:
                lb_b = lb.flatten().unsqueeze(0).expand(B, -1)
                ub_b = ub.flatten().unsqueeze(0).expand(B, -1)
            else:
                lb_b = lb.flatten(start_dim=1)
                ub_b = ub.flatten(start_dim=1)
            v = nu.flatten(start_dim=1)

            n = min(v.shape[-1], lb_b.shape[-1])
            if v.shape[-1] != lb_b.shape[-1]:
                lb_b, ub_b, v = lb_b[..., :n], ub_b[..., :n], v[..., :n]

            assert (lb_b <= ub_b).all(), "Invalid input bounds: lb > ub"
            contrib = (lb_b * v.clamp(min=0)).sum(dim=-1) + (ub_b * v.clamp(max=0)).sum(dim=-1)

            sce = None
            if return_sce:
                sce_flat = torch.where(v > 0, lb_b, ub_b)
                if lb.dim() < 2 and sce_flat.shape[-1] == lb.flatten().numel():
                    sce = sce_flat.view(B, *orig_shape)
                elif lb.dim() >= 2:
                    total = int(torch.tensor(orig_shape[1:]).prod().item())
                    sce = sce_flat.view(B, *orig_shape[1:]) if sce_flat.shape[-1] == total else sce_flat
                else:
                    sce = sce_flat
            return contrib, sce
