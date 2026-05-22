#===- act/back_end/solver/solver_dual.py - Dual Bounds Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualSolver: Wong-Kolter dual certified lower-bound solver.
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

    .. deprecated:: superseded by lazy M-broadcast
        ``DualSolver.evaluate_spec`` and ``_chunked_eval`` no longer call this
        helper — they thread ``M`` through ``compute_certified_bound`` and
        broadcast inside activation handlers instead, avoiding the M× memory
        blowup. Retained for transitional numerical-equivalence testing and
        external callers (e.g. legacy BaB code paths); will be removed once
        all callers migrate.

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

    def __init__(self, n_iters: int = 0):
        # DualTF is now a backward-registry holder (not a TransferFunction);
        # instantiate internally so callers don't need to know about it.
        # Recent refactor moved dual from the --tf-mode axis to the --solver axis,
        # making DualSolver fully self-contained — no external TF coupling.
        from act.back_end.dual_tf.dual_tf import DualTF
        self.tf = DualTF()
        self.n_iters = n_iters
        self._last_bounds: Optional[Bounds] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    def compute_certified_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        c: torch.Tensor, M: int = 1,
        return_sce: bool = False,
        enable_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        Implements ``Solver.compute_certified_bound``; see base for the
        full contract. DualSolver realises this via reverse-topological
        backward propagation of a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).

        Lazy M-broadcast: ``c`` has shape ``[B*M, num_classes]`` packed
        sample-major (row ``b*M+j`` = sample b's j-th spec row), but
        ``bounds_dict`` entries stay at ``[B, *shape]``. Activation handlers
        (RELU/SIGMOID/TANH) view nu as ``[B, M, n]`` and broadcast bounds
        ``[B, 1, n]`` against it — mathematically equivalent to the legacy
        M-expanded path, with M× lower bounds memory.
        """
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, num_classes], got shape {tuple(c.shape)}. "
                "Use c.unsqueeze(0) for single instance.")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if c.shape[0] % M != 0:
            raise ValueError(
                f"c batch dim {c.shape[0]} not divisible by M={M}; "
                f"expected c.shape[0] == B*M for some integer B"
            )
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
                raise ValueError("DualSolver.compute_certified_bound: net has no ASSERT layer")

            assert_preds = net.preds.get(assert_layer.id, [])
            if len(assert_preds) != 1:
                raise ValueError(
                    f"DualSolver.compute_certified_bound: ASSERT layer {assert_layer.id} must have "
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
                        f"DualSolver.compute_certified_bound: unknown layer kind '{k}' at layer {lid}; "
                        f"soundness requires explicit backward handler. "
                        f"Supported kinds: {sorted(registry.keys())}"
                    )

                preds = list(net.preds.get(lid, []))
                pred_nus, contrib = handler(layer, nu_here, bounds_dict, preds, M)

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
                M=M,
                return_sce=return_sce,
                enable_grad=enable_grad,
            )
            obj = obj + input_contrib
            return (obj, sce) if return_sce else obj

    def evaluate_spec(
        self, net: Net,
        out_spec: OutputSpec,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        num_classes: Optional[int] = None,
        chunk_size: Optional[int] = None,
        enable_grad: bool = False,
    ) -> SpecBatchResult:
        """Dual bound evaluation for any OutputSpec — self-contained entry point.

        Refactor note: ``bounds_dict`` is optional. When omitted (the typical
        case), the solver gathers the net's INPUT_SPEC seed bounds and computes
        per-layer forward bounds internally via ``compute_forward_bounds``.
        Callers who already have a bounds_dict (e.g. BaB refinement loops) may
        pass it explicitly to skip the recomputation.

        Strategy: dispatch on ``out_spec.kind`` into two branches that share
        ``compute_certified_bound`` but use opposite sign conventions and
        opposite row aggregators.

        - ALL-rows kinds (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE):
          ``encode_linear`` emits (C, thresholds) in UB-cert form (CERTIFIED
          iff ``UB(C @ y) < threshold``). Pass ``-C`` / ``-thresholds`` to
          ``compute_certified_bound`` and compare; ``slack >= 0`` means the
          row passes. Certified iff every row passes (``.all()``).
        - EXISTS-row kind (UNSAFE_LINEAR): the unsafe polytope is
          ``P = {y : c_i^T y <= d_i for ALL i}``. SAFE iff for all reachable
          y, some row i satisfies ``c_i^T y > d_i`` (escape). Sound
          strengthening via quantifier swap (mirrors ``verifier.py:574-580``):
          certify SAFE iff there exists a row i with ``LB_dual(c_i^T y) > d_i``.
          ``encode_linear`` emits UNSAFE_LINEAR in LB-cert form, so pass ``+C``
          / ``+thresholds`` directly (no sign flip). Certified iff any row
          escapes (``.any()``).

        Raises:
            ValueError: if net lacks ASSERT layer, ASSERT has != 1 predecessor,
                or (when bounds_dict is supplied) the output layer's bounds are
                missing / unbatched.
        """
        if bounds_dict is None:
            from act.back_end.dual_tf.tf_forward import compute_forward_bounds
            from act.back_end.verifier import (
                gather_input_spec_layers,
                seed_from_input_specs,
            )
            spec_layers = gather_input_spec_layers(net)
            seed_bounds = seed_from_input_specs(spec_layers)
            bounds_dict = compute_forward_bounds(
                net, seed_bounds.lb, seed_bounds.ub, post_activation=False,
            )

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
            # EXISTS-row branch. encode_linear emits LB-cert form for
            # UNSAFE_LINEAR (specs.py:179-201) — pass +C / +thresholds
            # directly. Certified iff any row escapes the unsafe polytope.
            # Slack semantics is ASYMMETRIC vs ALL-rows kinds below:
            # here ``slack > 0`` means the row certifies; ``min_slack`` is
            # NOT a meaningful summary (use ``slack.max(dim=-1)`` instead).
            fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
            C = fe_params["C"].contiguous()
            thresholds = fe_params["thresholds"].contiguous()
            N = int(fe_params["M"])
            active_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            with torch.set_grad_enabled(enable_grad):
                if chunk_size is None or N <= chunk_size:
                    margins_flat = self.compute_certified_bound(
                        net, bounds_dict, C, M=N, enable_grad=enable_grad,
                    )
                else:
                    margins_flat = self._chunked_eval(
                        net, bounds_dict, C, B, N, n_out, chunk_size, enable_grad,
                    )
                if isinstance(margins_flat, tuple):
                    margins_flat = margins_flat[0]
                margins = margins_flat.view(B, N)
                slack = margins - thresholds
                certified = ((slack > 0) & active_mask).any(dim=-1)

            return SpecBatchResult(
                margins=margins,
                slack=slack,
                active_mask=active_mask,
                certified=certified,
            )

        fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
        C_neg = -fe_params["C"].contiguous()
        thresholds_neg = -fe_params["thresholds"].contiguous()
        M = int(fe_params["M"])
        active_mask = torch.ones(B, M, dtype=torch.bool, device=device)

        with torch.set_grad_enabled(enable_grad):
            if chunk_size is None or M <= chunk_size:
                margins_flat = self.compute_certified_bound(
                    net, bounds_dict, C_neg, M=M, enable_grad=enable_grad,
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
            margins_chunk = self.compute_certified_bound(
                net, bounds_dict, C_chunk, M=m_chunk, enable_grad=enable_grad,
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
                                    M: int = 1,
                                    return_sce: bool = False,
                                    enable_grad: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute lb·[nu]_+ + ub·[nu]_- over the input box (batched).

        Lazy M-broadcast: ``nu`` has leading dim ``B*M`` (sample-major)
        while batched ``bounds_dict[input_lid]`` is ``[B, *shape]``. The
        contribution per (b, m) reuses the same bounds for all m via
        ``[B, 1, n]`` broadcast against ``[B, M, n]``. Bit-identical to
        legacy M-expanded path.

        The unbatched (``lb.dim() < 2``) and missing-bounds (lb/ub from
        ``input_layer.params``) paths are preserved: they broadcast a single
        ``[n]`` tensor against ``[BM, n]`` nu — the same as legacy with B=BM.
        """
        with torch.set_grad_enabled(enable_grad):
            BM = nu.shape[0]
            assert BM % M == 0, (
                f"_input_contribution_from_nu: nu batch {BM} not divisible by M={M}"
            )
            B = BM // M
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
            v_flat = nu.flatten(start_dim=1)                       # [BM, n_in]

            if lb.dim() < 2:
                lb_b = lb.flatten().unsqueeze(0).expand(BM, -1)
                ub_b = ub.flatten().unsqueeze(0).expand(BM, -1)
                n = min(v_flat.shape[-1], lb_b.shape[-1])
                if v_flat.shape[-1] != lb_b.shape[-1]:
                    lb_b, ub_b, v_flat = lb_b[..., :n], ub_b[..., :n], v_flat[..., :n]
                assert (lb_b <= ub_b).all(), "Invalid input bounds: lb > ub"
                contrib = ((lb_b * v_flat.clamp(min=0)).sum(dim=-1)
                           + (ub_b * v_flat.clamp(max=0)).sum(dim=-1))
                sce = None
                if return_sce:
                    sce_flat = torch.where(v_flat > 0, lb_b, ub_b)
                    if sce_flat.shape[-1] == lb.flatten().numel():
                        sce = sce_flat.view(BM, *orig_shape)
                    else:
                        sce = sce_flat
                return contrib, sce

            lb_B = lb.flatten(start_dim=1)                         # [B, n_in]
            ub_B = ub.flatten(start_dim=1)                         # [B, n_in]
            n = min(v_flat.shape[-1], lb_B.shape[-1])
            if v_flat.shape[-1] != lb_B.shape[-1]:
                lb_B = lb_B[..., :n]
                ub_B = ub_B[..., :n]
                v_flat = v_flat[..., :n]
            assert (lb_B <= ub_B).all(), "Invalid input bounds: lb > ub"

            v = v_flat.view(B, M, n)                               # [B, M, n] view
            lb_bc = lb_B.unsqueeze(1)                              # [B, 1, n]
            ub_bc = ub_B.unsqueeze(1)                              # [B, 1, n]
            contrib_BM = ((lb_bc * v.clamp(min=0)).sum(dim=-1)
                          + (ub_bc * v.clamp(max=0)).sum(dim=-1))  # [B, M]
            contrib = contrib_BM.view(BM)

            sce = None
            if return_sce:
                sce_BMn = torch.where(v > 0, lb_bc, ub_bc)         # [B, M, n]
                sce_flat = sce_BMn.view(BM, n)
                total = int(torch.tensor(orig_shape[1:]).prod().item())
                sce = sce_flat.view(BM, *orig_shape[1:]) if sce_flat.shape[-1] == total else sce_flat
            return contrib, sce
