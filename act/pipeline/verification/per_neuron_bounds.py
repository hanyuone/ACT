#!/usr/bin/env python3
#===- act/pipeline/verification/per_neuron_bounds.py - Per-Neuron Bounds --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Level-2 (per-neuron) numerical validation. This module checks that ACT’s
#   abstract bounds (lb/ub) over-approximate the concrete activations produced
#   by a reference PyTorch forward pass, neuron-by-neuron, for a single input.
#
# Key Features:
#   - Abstract bounds extraction:
#       Runs ACT analyze() to obtain per-layer Bounds(lb, ub) for all layers.
#   - Concrete activation tracing (hook-based):
#       Captures intermediate module outputs via forward hooks on “hookable”
#       PyTorch modules (Linear/Conv/ReLU/Pool/Flatten/...).
#   - Built-in alignment to ACT layer IDs:
#       Aligns hook events to ACT layers using a strict hookable-order strategy
#       (with optional shape sanity checks from ACT layer params).
#   - Per-neuron violation detection:
#       A neuron is flagged only if it exceeds [lb, ub] beyond tolerance:
#           tol = atol + rtol * |a|   (a = concrete activation)
#   - Debug-oriented reporting:
#       Computes per-layer statistics and returns the top-K worst violations
#       (largest gaps) for fast bug localization.
#
# Pipeline (single sample):
#   (input_tensor, entry_fact)
#     → compute_abstract_bounds()              : ACT analysis → bounds_by_layer
#     → collect_concrete_activations()         : hooks → concrete_by_layer + meta
#     → compare_bounds_per_neuron()            : gaps/violations/topk report
#     → run_per_neuron_bounds_check()          : single entry point
#
# Numerical Policy:
#   - Tolerance: tol = atol + rtol * |a|
#       Mitigates false positives due to floating-point roundoff.
#   - nan_policy="error":
#       Any NaN/Inf encountered in concrete or bounds yields ERROR status.
#   - topk:
#       When violations occur, returns the K most severe violating neurons
#       (largest gap) to simplify debugging.
#
# Outputs (dict):
#   - status: PASS / FAIL / ERROR
#   - violations_total: total number of violating neurons
#   - violations_topk: list of worst-K violations (layer_id, neuron_index, gap, ...)
#   - layerwise_stats: per-layer summary (num_violations, max_gap, mean_gap, ranges)
#   - alignment: meta describing the alignment mode and event/layer counts
#   - total_checks: total number of neurons compared
#   - worst_gap: maximum gap observed across all layers
#
# Usage:
#   result = run_per_neuron_bounds_check(
#       act_net=act_net,
#       model=torch_model,
#       input_tensor=x,
#       entry_fact=entry_fact,
#       tf_mode="interval",
#       config=PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10),
#   )
#
# Design Notes:
#   - Alignment is strict by design: mismatches are surfaced as explicit errors
#     (kind/type/shape) rather than silently producing incorrect matches.
#   - Only “hookable” modules are traced to keep the activation stream stable
#     and comparable to ACT layer kinds.
#
#===---------------------------------------------------------------------===#


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from act.back_end.analyze import analyze
from act.back_end.core import Bounds, Layer
from act.back_end.verifier import find_entry_layer_id
from act.back_end.transfer_functions import set_transfer_function_mode


_ACT_KIND_TO_MODULE = {
    "DENSE": "Linear",
    "CONV1D": "Conv1d",
    "CONV2D": "Conv2d",
    "CONV3D": "Conv3d",
    "RELU": "ReLU",
    "SIGMOID": "Sigmoid",
    "TANH": "Tanh",
    "SILU": "SiLU",
    "LRELU": "LeakyReLU",
    "FLATTEN": "Flatten",
    "MAXPOOL1D": "MaxPool1d",
    "MAXPOOL2D": "MaxPool2d",
    "MAXPOOL3D": "MaxPool3d",
    "AVGPOOL1D": "AvgPool1d",
    "AVGPOOL2D": "AvgPool2d",
    "AVGPOOL3D": "AvgPool3d",
    "ADAPTIVEAVGPOOL1D": "AdaptiveAvgPool1d",
    "ADAPTIVEAVGPOOL2D": "AdaptiveAvgPool2d",
    "ADAPTIVEAVGPOOL3D": "AdaptiveAvgPool3d",
}


def compute_abstract_bounds(
    act_net,
    entry_fact,
    *,
    tf_mode: str = "interval",
) -> Tuple[Dict[int, Bounds], List[str]]:
    """Compute abstract bounds for all layers in the ACT net.

    For B>1 inputs under ``hybridz`` / ``dual`` (which are single-instance
    by construction), runs the analysis once per sample and stacks the
    per-layer bounds along a new batch axis. ``interval`` is batch-native
    (post batch1) and runs a single batched pass.
    """
    from act.back_end.core import Fact

    errors: List[str] = []
    bounds_by_layer: Dict[int, Bounds] = {}
    set_transfer_function_mode(tf_mode)
    entry_id = find_entry_layer_id(act_net)

    seed_lb = entry_fact.bounds.lb
    seed_ub = entry_fact.bounds.ub
    is_batched_input = seed_lb.dim() >= 2 and seed_lb.shape[0] > 1
    needs_per_sample = is_batched_input and tf_mode in ("hybridz", "dual")

    if needs_per_sample:
        B = seed_lb.shape[0]
        per_sample_after: List[Dict[int, Any]] = []
        for b_idx in range(B):
            # Preserve the leading B=1 axis (batch-native TFs require >=2D).
            single_fact = Fact(
                bounds=Bounds(
                    lb=seed_lb[b_idx:b_idx + 1],
                    ub=seed_ub[b_idx:b_idx + 1],
                ),
                cons=entry_fact.cons,
            )
            _before, after_b, _globalC = analyze(
                act_net, entry_id, single_fact
            )
            per_sample_after.append(after_b)

        for layer in getattr(act_net, "layers", []):
            lid = layer.id
            if not all(lid in a for a in per_sample_after):
                errors.append(
                    f"Missing bounds for layer_id={lid} (kind={layer.kind}) "
                    f"in at least one sample under per-sample {tf_mode} analysis"
                )
                continue
            lbs = [a[lid].bounds.lb for a in per_sample_after]
            ubs = [a[lid].bounds.ub for a in per_sample_after]
            # Per-sample TF backends differ: batched ones (interval) leave the
            # B=1 axis intact (shape (1, *)); single-instance ones (dual)
            # return 1-D (numel,). Normalize to at-least-2D then cat on dim 0
            # so the result is always (B, *).
            lbs = [t.unsqueeze(0) if t.dim() < 2 else t for t in lbs]
            ubs = [t.unsqueeze(0) if t.dim() < 2 else t for t in ubs]
            try:
                stacked_lb = torch.cat(lbs, dim=0)
                stacked_ub = torch.cat(ubs, dim=0)
            except RuntimeError as e:
                errors.append(
                    f"Failed to stack per-sample bounds at layer_id={lid}: {e}"
                )
                continue
            if stacked_lb.shape != stacked_ub.shape:
                errors.append(
                    f"Bounds shape mismatch at layer_id={lid}: "
                    f"lb={tuple(stacked_lb.shape)} ub={tuple(stacked_ub.shape)}"
                )
                continue
            if not torch.isfinite(stacked_lb).all() or not torch.isfinite(stacked_ub).all():
                errors.append(f"Non-finite bounds at layer_id={lid}")
                continue
            bounds_by_layer[lid] = Bounds(lb=stacked_lb, ub=stacked_ub)
        return bounds_by_layer, errors

    _before, after, _globalC = analyze(act_net, entry_id, entry_fact)

    for layer in getattr(act_net, "layers", []):
        lid = layer.id
        if lid not in after:
            errors.append(f"Missing bounds for layer_id={lid} (kind={layer.kind})")
            continue
        fact = after[lid]
        lb = fact.bounds.lb
        ub = fact.bounds.ub
        if lb.shape != ub.shape:
            errors.append(
                f"Bounds shape mismatch at layer_id={lid}: lb={tuple(lb.shape)} ub={tuple(ub.shape)}"
            )
            continue
        if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
            errors.append(f"Non-finite bounds at layer_id={lid}")
            continue
        bounds_by_layer[lid] = Bounds(lb=lb, ub=ub)

    return bounds_by_layer, errors


def collect_concrete_activations(
    act_net,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    strict_single_call_per_module: bool = False,
) -> Tuple[Dict[int, torch.Tensor], List[str], List[str], Dict[str, Any]]:
    """
    Collect concrete activations and align them to ACT layer IDs.
    """
    errors: List[str] = []
    warnings: List[str] = []
    call_counts: Dict[int, int] = {}
    hookable_events: List[Tuple[str, torch.Tensor]] = []
    hooks = []

    def _hook(module, inputs, output):
        module_id = id(module)
        call_counts[module_id] = call_counts.get(module_id, 0) + 1
        if strict_single_call_per_module and call_counts[module_id] > 1:
            errors.append(f"Module called multiple times: {module.__class__.__name__}")
        if not torch.is_tensor(output):
            warnings.append(f"Non-tensor output from {module.__class__.__name__}")
            return
        tensor = output.detach()
        hookable_events.append((module.__class__.__name__, tensor))

    hookable_kinds = set(_ACT_KIND_TO_MODULE.values())

    for module in model.modules():
        if module is model:
            continue
        if module.__class__.__name__ in hookable_kinds:
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for h in hooks:
            h.remove()

    hookable_layers = [
        L for L in getattr(act_net, "layers", [])
        if _ACT_KIND_TO_MODULE.get(L.kind) in hookable_kinds
    ]

    if len(hookable_events) != len(hookable_layers):
        errors.append(
            f"Hookable count mismatch: events={len(hookable_events)} layers={len(hookable_layers)}"
        )

    def _numel(shape: Tuple[int, ...]) -> int:
        prod = 1
        for s in shape:
            prod *= int(s)
        return int(prod)

    def _drop_batch_if_and_only_if_batch1(
        raw_shape: Tuple[int, ...],
        expected_shape: Tuple[int, ...] | None,
    ) -> Tuple[Tuple[int, ...], bool, str]:
        """Match per-sample shape: raw and expected may carry any leading
        batch dim (B=1 or B>1). Comparison strips the leading dim from both
        sides if their per-sample ranks line up."""
        if expected_shape is None:
            return raw_shape, False, "expected_shape_missing"
        if not raw_shape:
            return raw_shape, False, "raw_shape_empty"
        if len(raw_shape) == len(expected_shape) + 1:
            candidate = tuple(raw_shape[1:])
            if candidate == expected_shape:
                return candidate, True, "dropped_batch"
            return raw_shape, False, "drop_would_not_match_expected"
        if len(raw_shape) == len(expected_shape):
            if tuple(raw_shape[1:]) == tuple(expected_shape[1:]):
                return raw_shape, True, "per_sample_matched"
            return raw_shape, False, "per_sample_shape_mismatch"
        return raw_shape, False, "rank_mismatch"

    mapping: Dict[int, torch.Tensor] = {}

    for idx, layer in enumerate(hookable_layers):
        if idx >= len(hookable_events):
            break
        module_type, tensor = hookable_events[idx]
        expected = _ACT_KIND_TO_MODULE.get(layer.kind)
        if expected is None:
            errors.append(
                f"Unsupported ACT kind at position {idx}: act_kind={layer.kind}"
            )
        elif expected != module_type:
            errors.append(
                f"Kind/type mismatch at position {idx}: act_kind={layer.kind} event_type={module_type}"
            )
        expected_shape = None

        params = getattr(layer, "params", {}) or {}
        if "output_shape" in params:
            expected_shape = tuple(int(x) for x in params["output_shape"])
        elif "shape" in params:
            expected_shape = tuple(int(x) for x in params["shape"])
        if expected_shape is not None:
            raw_shape = tuple(int(x) for x in tensor.shape)
            no_batch_shape, dropped, drop_reason = _drop_batch_if_and_only_if_batch1(
                raw_shape,
                expected_shape,
            )
            if not dropped:
                ev_numel = _numel(raw_shape)
                exp_numel = _numel(expected_shape)
                if ev_numel != exp_numel:
                    errors.append(
                        f"Shape mismatch at layer_id={layer.id}: "
                        f"event_raw={raw_shape} event_no_batch={no_batch_shape} "
                        f"expected={expected_shape} "
                        f"dropped_batch={dropped} drop_reason={drop_reason} "
                        f"event_numel={ev_numel} expected_numel={exp_numel}"
                    )
        mapping[layer.id] = tensor

    info = {
        "mode": "hookable_order_strict",
        "hookable_events": len(hookable_events),
        "hookable_layers": len(hookable_layers),
    }

    return mapping, errors, warnings, info


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def compare_bounds_per_neuron(
    *,
    bounds_by_layer: Dict[int, Bounds],
    concrete_by_layer: Dict[int, torch.Tensor],
    layer_by_id: Dict[int, Layer],
    atol: float = 1e-6,
    rtol: float = 0.0,
    topk: int = 10,
    nan_policy: str = "error",
) -> Dict[str, Any]:
    """
    Compare per-neuron concrete activations against abstract bounds.
    """
    errors: List[str] = []
    warnings: List[str] = []
    violations_topk: List[Dict[str, Any]] = []
    layerwise_stats: List[Dict[str, Any]] = []
    violations_total = 0

    if set(bounds_by_layer.keys()) != set(concrete_by_layer.keys()):
        missing = set(bounds_by_layer.keys()) - set(concrete_by_layer.keys())
        extra = set(concrete_by_layer.keys()) - set(bounds_by_layer.keys())
        errors.append(f"Layer key mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    if errors:
        return {
            "status": "ERROR",
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "errors": errors,
            "warnings": warnings,
        }

    candidates: List[Dict[str, Any]] = []

    for layer_id, bounds in bounds_by_layer.items():
        concrete = concrete_by_layer[layer_id]
        layer = layer_by_id.get(layer_id)
        kind = layer.kind if layer is not None else "UNKNOWN"
        lb = bounds.lb
        ub = bounds.ub

        if nan_policy == "error":
            if not _is_finite(concrete) or not _is_finite(lb) or not _is_finite(ub):
                errors.append(f"Non-finite value at layer_id={layer_id}")
                continue

        concrete_flat = concrete.reshape(-1)
        lb_flat = lb.reshape(-1)
        ub_flat = ub.reshape(-1)
        if concrete_flat.numel() != lb_flat.numel():
            errors.append(
                f"Shape mismatch at layer_id={layer_id}: "
                f"concrete_numel={concrete_flat.numel()} bounds_numel={lb_flat.numel()}"
            )
            continue
        tol = atol + rtol * concrete_flat.abs()

        diff_low = (lb_flat - tol) - concrete_flat
        diff_high = concrete_flat - (ub_flat + tol)
        gap = torch.maximum(diff_low, diff_high)
        gap = torch.clamp(gap, min=0.0)

        violations_mask = gap > 0
        num_violations = int(violations_mask.sum().item())
        violations_total += num_violations

        if num_violations > 0:
            gap_vals = gap[violations_mask]
            max_gap = float(gap_vals.max().item())
            mean_gap = float(gap_vals.mean().item())
        else:
            max_gap = 0.0
            mean_gap = 0.0

        layerwise_stats.append(
            {
                "layer_id": int(layer_id),
                "kind": kind,
                "shape": list(lb.shape),
                "num_neurons": int(concrete_flat.numel()),
                "num_violations": int(num_violations),
                "max_gap": float(max_gap),
                "mean_gap": float(mean_gap),
                "lb_min": float(lb_flat.min().item()) if lb_flat.numel() > 0 else 0.0,
                "lb_max": float(lb_flat.max().item()) if lb_flat.numel() > 0 else 0.0,
                "ub_min": float(ub_flat.min().item()) if ub_flat.numel() > 0 else 0.0,
                "ub_max": float(ub_flat.max().item()) if ub_flat.numel() > 0 else 0.0,
                "concrete_min": float(concrete_flat.min().item()) if concrete_flat.numel() > 0 else 0.0,
                "concrete_max": float(concrete_flat.max().item()) if concrete_flat.numel() > 0 else 0.0,
                "layer_status": "FAIL" if num_violations > 0 else "PASS",
            }
        )

        if topk > 0:
            k = min(int(topk), int(concrete_flat.numel()))
            if k > 0:
                vals, idxs = torch.topk(gap, k=k)
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    if v <= 0:
                        continue
                    i = int(i)
                    candidates.append(
                        {
                            "layer_id": int(layer_id),
                            "kind": kind,
                            "neuron_index": i,
                            "gap": float(v),
                            "concrete": float(concrete_flat[i].item()),
                            "lb": float(lb_flat[i].item()),
                            "ub": float(ub_flat[i].item()),
                        }
                    )

    if errors:
        return {
            "status": "ERROR",
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "errors": errors,
            "warnings": warnings,
        }

    candidates.sort(key=lambda x: x["gap"], reverse=True)
    violations_topk = candidates[: int(topk)]

    status = "FAIL" if violations_total > 0 else "PASS"
    return {
        "status": status,
        "violations_total": int(violations_total),
        "violations_topk": violations_topk,
        "layerwise_stats": layerwise_stats,
        "errors": errors,
        "warnings": warnings,
    }

"""
A standard absolute-plus-relative tolerance used in numerical computing to 
avoid flagging floating-point roundoff as “unsoundness”; 

A neuron as violating only if:
    it falls outside [lb, ub] by more than atol + rtol*|a|.

topk: 
    When violations occur, topk returns the K most severe violation cases, 
    making it easier to quickly pinpoint the bug.
"""
@dataclass(frozen=True)
class PerNeuronCheckConfig:
    atol: float = 1e-6
    rtol: float = 0.0
    topk: int = 10
    nan_policy: str = "error"


def run_per_neuron_bounds_check(
    *,
    act_net,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    entry_fact,
    tf_mode: str,
    config: PerNeuronCheckConfig,
) -> Dict[str, Any]:
    """
    Full per-neuron bounds validation pipeline for a single input sample.
    """
    errors: List[str] = []
    warnings: List[str] = []

    bounds_by_layer, bounds_errors = compute_abstract_bounds(
        act_net,
        entry_fact,
        tf_mode=tf_mode,
    )
    if bounds_errors:
        errors.extend(bounds_errors)

    concrete_by_layer, event_errors, event_warnings, alignment_meta = collect_concrete_activations(
        act_net,
        model,
        input_tensor,
    )
    if event_errors:
        errors.extend(event_errors)
    if event_warnings:
        warnings.extend(event_warnings)

    if errors:
        return {
            "status": "ERROR",
            "errors": errors,
            "warnings": warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment_meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    missing_bounds = [lid for lid in concrete_by_layer.keys() if lid not in bounds_by_layer]
    if missing_bounds:
        return {
            "status": "ERROR",
            "errors": [f"Missing bounds for layer_ids={sorted(missing_bounds)}"],
            "warnings": warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment_meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    bounds_for_compare = {lid: bounds_by_layer[lid] for lid in concrete_by_layer.keys()}
    compare = compare_bounds_per_neuron(
        bounds_by_layer=bounds_for_compare,
        concrete_by_layer=concrete_by_layer,
        layer_by_id=getattr(act_net, "by_id", {}),
        atol=config.atol,
        rtol=config.rtol,
        topk=config.topk,
        nan_policy=config.nan_policy,
    )

    if compare.get("status") == "ERROR":
        return {
            "status": "ERROR",
            "errors": compare.get("errors", []),
            "warnings": warnings + compare.get("warnings", []),
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment_meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    layerwise_stats = compare.get("layerwise_stats", [])
    total_checks = sum(int(s.get("num_neurons", 0)) for s in layerwise_stats)
    worst_gap = 0.0
    for s in layerwise_stats:
        worst_gap = max(worst_gap, float(s.get("max_gap", 0.0)))

    compare["alignment"] = alignment_meta
    compare["warnings"] = warnings + compare.get("warnings", [])
    compare["total_checks"] = int(total_checks)
    compare["worst_gap"] = float(worst_gap)
    return compare
