"""
Property violation detection for ACTFuzzer.

Checks if model outputs violate OutputSpec properties and records counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import torch

from act.front_end.specs import OutputSpec, OutKind


@dataclass
class Counterexample:
    """
    Counterexample with full details.
    
    Represents an input that violates the OutputSpec property.
    This is the primary output of ACTFuzzer.
    
    Attributes:
        input: Input tensor that caused violation (perturbed)
        output: Model's output on this input
        expected: Expected value (e.g., true label)
        actual: Actual value (e.g., predicted label)
        kind: Type of violation (TOP1_ROBUST, MARGIN_ROBUST, etc.)
        confidence: Confidence score of the prediction
        timestamp: When the counterexample was found
        seed_index: Which VNNLib instance (0..N-1) this counterexample belongs to
        seed_input: Original unperturbed input (L∞ projection anchor and visualization baseline)
    """
    input: torch.Tensor
    output: torch.Tensor
    expected: int
    actual: int
    kind: str
    confidence: float
    timestamp: float
    seed_index: Optional[int] = None
    seed_input: Optional[torch.Tensor] = None
    
    def summary(self) -> str:
        """One-line summary of the counterexample."""
        return f"{self.kind}: expected {self.expected}, got {self.actual} (conf={self.confidence:.3f})"
    
    def save(self, path):
        """Save counterexample to disk."""
        torch.save({
            "input": self.input,
            "output": self.output,
            "expected": self.expected,
            "actual": self.actual,
            "kind": self.kind,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "seed_index": self.seed_index,
            "seed_input": self.seed_input,
        }, path)
    
    @staticmethod
    def load(path):
        """Load counterexample from disk."""
        data = torch.load(path)
        return Counterexample(
            input=data["input"],
            output=data["output"],
            expected=data["expected"],
            actual=data["actual"],
            kind=data["kind"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            seed_index=data.get("seed_index"),
            seed_input=data.get("seed_input")
        )


# =============================================================================
# Property Checking
# =============================================================================

class PropertyChecker:
    """
    Vectorized property checker for violation detection.
    
    Supports all OutKind types:
    - TOP1_ROBUST: Top prediction must equal true label
    - MARGIN_ROBUST: Margin to runner-up must exceed threshold
    - RANGE: Output must be within [lb, ub]
    - LINEAR_LE: Linear constraint c^T y <= d must hold
    
    Example:
        >>> checker = PropertyChecker(output_spec)
        >>> violations = checker.check(inputs, outputs, labels)
        >>> # violations is List[Counterexample | None] of length B
    """
    
    def __init__(self, output_spec: Optional[OutputSpec]):
        """Initialize property checker."""
        self.spec = output_spec
        
        # Dispatch table for spec kinds
        self._dispatch = {
            OutKind.TOP1_ROBUST: self._check_top1,
            OutKind.MARGIN_ROBUST: self._check_margin,
            OutKind.RANGE: self._check_range,
            OutKind.LINEAR_LE: self._check_linear,
        }
    
    def check(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        seeds: 'FuzzingSeed'
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """
        Check B samples for violations in parallel.
        
        Args:
            inputs: Input tensors [B, C, H, W] or [B, D]
            outputs: Model outputs [B, num_classes]
            seeds: FuzzingSeed batch with labels, original tensors, and indices
        
        Returns:
            Tuple of (violations_mask BoolTensor[B], List[Counterexample] for violations only)
        """
        B = inputs.shape[0]
        device = outputs.device
        
        if self.spec is None:
            return (torch.zeros(B, dtype=torch.bool, device=device), [])
        
        handler = self._dispatch.get(self.spec.kind)
        if handler is None:
            return (torch.zeros(B, dtype=torch.bool, device=device), [])
        
        # Pre-compute label tensors (used by all check methods)
        y_true = seeds.label.to(device)
        valid_mask = (y_true >= 0)
        
        return handler(inputs, outputs, y_true, valid_mask, seeds=seeds)
    
    def _build_results(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        violations_mask: torch.Tensor,
        kind: str,
        actual_values: torch.Tensor,
        confidence_values: torch.Tensor,
        seeds: 'FuzzingSeed',
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """Build Counterexample list from violation mask and return (mask, list)."""
        timestamp = time.time()
        violation_indices = violations_mask.nonzero(as_tuple=True)[0]
        
        counterexamples: List[Counterexample] = []
        
        for idx in violation_indices:
            i = idx.item()
            counterexamples.append(Counterexample(
                input=inputs[i].detach().cpu(),
                output=outputs[i].detach().cpu(),
                expected=int(seeds.label[i].item()),
                actual=int(actual_values[i].item()),
                kind=kind,
                confidence=float(confidence_values[i].item()),
                timestamp=timestamp,
                seed_index=int(seeds.original_index[i].item()),
                seed_input=seeds.original_tensor[i].detach().cpu(),
            ))
        
        return violations_mask, counterexamples
    
    def _check_top1(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        seeds: 'FuzzingSeed' = None,
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """Check if top prediction != y_true for B samples."""
        pred_classes = outputs.argmax(dim=1)
        violations_mask = valid_mask & (pred_classes != y_true)
        
        probs = torch.softmax(outputs, dim=1)
        confidences = probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        
        return self._build_results(
            inputs, outputs, violations_mask, "TOP1_ROBUST",
            pred_classes, confidences, seeds=seeds)
    
    def _check_margin(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        **kw,
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """Check if margin(y_true) < threshold for B samples."""
        B = inputs.shape[0]
        device = outputs.device
        num_classes = outputs.shape[1]
        y_safe = y_true.clamp(min=0)
        
        true_logits = outputs.gather(1, y_safe.unsqueeze(1)).squeeze(1)
        
        mask = torch.ones(B, num_classes, dtype=torch.bool, device=device)
        mask.scatter_(1, y_safe.unsqueeze(1), False)
        runner_up_logits = outputs.masked_fill(~mask, float('-inf')).max(dim=1).values
        
        margins = true_logits - runner_up_logits
        threshold = getattr(self.spec, 'margin', None)
        if threshold is None:
            threshold = 0.0
        elif torch.is_tensor(threshold):
            threshold = threshold.to(device)
        violations_mask = valid_mask & (margins < threshold)
        
        actual = torch.full((B,), -1, dtype=torch.long, device=device)
        return self._build_results(
            inputs, outputs, violations_mask, "MARGIN_ROBUST",
            actual, margins, **kw)
    
    def _check_range(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        **kw,
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """Check if outputs are outside [lb, ub] bounds for B samples."""
        B = inputs.shape[0]
        device = outputs.device
        
        if self.spec.lb is None or self.spec.ub is None:
            return (torch.zeros(B, dtype=torch.bool, device=device), [])
        
        lb = self._to_tensor(self.spec.lb, device)
        ub = self._to_tensor(self.spec.ub, device)
        
        violations_mask = ((outputs < lb) | (outputs > ub)).any(dim=1)
        
        lb_viol = (lb - outputs).clamp(min=0).max(dim=1).values
        ub_viol = (outputs - ub).clamp(min=0).max(dim=1).values
        confidences = torch.maximum(lb_viol, ub_viol)
        
        actual = torch.full((B,), -1, dtype=torch.long, device=device)
        return self._build_results(
            inputs, outputs, violations_mask, "RANGE",
            actual, confidences, **kw)
    
    def _check_linear(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        **kw,
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """Check if linear constraint c^T y <= d is violated for B samples."""
        B = inputs.shape[0]
        device = outputs.device
        
        if self.spec.c is None or self.spec.d is None:
            return (torch.zeros(B, dtype=torch.bool, device=device), [])
        
        c = self.spec.c.to(device)
        d = float(self.spec.d)
        
        values = (outputs * c).sum(dim=1)
        violations_mask = (values > d)
        confidences = values - d
        
        actual = torch.full((B,), -1, dtype=torch.long, device=device)
        return self._build_results(
            inputs, outputs, violations_mask, "LINEAR_LE",
            actual, confidences, **kw)
    
    @staticmethod
    def _to_tensor(val, device: torch.device) -> torch.Tensor:
        """Convert value to tensor on device."""
        if isinstance(val, torch.Tensor):
            return val.to(device)
        return torch.tensor(val, device=device)
