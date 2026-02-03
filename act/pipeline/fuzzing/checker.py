"""
Property violation detection for ACTFuzzer.

Checks if model outputs violate OutputSpec properties and records counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
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
        seed_index: Index of the original seed (optional, for tracking)
        seed_input: Original unperturbed input (optional, for visualization)
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
        labels: List[Optional[int]],
        seed_tensors: Optional[List[torch.Tensor]] = None,
        seed_indices: Optional[List[int]] = None
    ) -> List[Optional[Counterexample]]:
        """
        Check B samples for violations in parallel.
        
        Args:
            inputs: Input tensors [B, C, H, W] or [B, D]
            outputs: Model outputs [B, num_classes]
            labels: List of B ground truth labels (None entries skip checking)
            seed_tensors: Optional list of original seed tensors
            seed_indices: Optional list of seed indices
        
        Returns:
            List of B elements, each Counterexample or None
        """
        B = inputs.shape[0]
        
        if self.spec is None:
            return [None] * B
        
        handler = self._dispatch.get(self.spec.kind)
        if handler is None:
            return [None] * B
        
        # Pre-compute label tensors (used by all check methods)
        device = outputs.device
        y_true = torch.tensor(
            [l if l is not None else -1 for l in labels],
            dtype=torch.long, device=device
        )
        valid_mask = (y_true >= 0)
        
        return handler(inputs, outputs, y_true, valid_mask,
                       labels=labels, seed_tensors=seed_tensors,
                       seed_indices=seed_indices)
    
    def _build_results(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        violations_mask: torch.Tensor,
        kind: str,
        actual_values: torch.Tensor,
        confidence_values: torch.Tensor,
        labels: List[Optional[int]],
        seed_tensors: Optional[List[torch.Tensor]],
        seed_indices: Optional[List[int]],
    ) -> List[Optional[Counterexample]]:
        """Build Counterexample list from violation mask."""
        B = inputs.shape[0]
        timestamp = time.time()
        violation_indices = violations_mask.nonzero(as_tuple=True)[0]
        
        results: List[Optional[Counterexample]] = [None] * B
        
        for idx in violation_indices:
            i = idx.item()
            results[i] = Counterexample(
                input=inputs[i].detach().cpu(),
                output=outputs[i].detach().cpu(),
                expected=labels[i],
                actual=int(actual_values[i].item()),
                kind=kind,
                confidence=float(confidence_values[i].item()),
                timestamp=timestamp,
                seed_index=seed_indices[i] if seed_indices else None,
                seed_input=seed_tensors[i].detach().cpu() if seed_tensors else None,
            )
        
        return results
    
    def _check_top1(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        **kw,
    ) -> List[Optional[Counterexample]]:
        """Check if top prediction != y_true for B samples."""
        pred_classes = outputs.argmax(dim=1)
        violations_mask = valid_mask & (pred_classes != y_true)
        
        probs = torch.softmax(outputs, dim=1)
        confidences = probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        
        return self._build_results(
            inputs, outputs, violations_mask, "TOP1_ROBUST",
            pred_classes, confidences, **kw)
    
    def _check_margin(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        **kw,
    ) -> List[Optional[Counterexample]]:
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
        threshold = getattr(self.spec, 'margin', 0.0) or 0.0
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
    ) -> List[Optional[Counterexample]]:
        """Check if outputs are outside [lb, ub] bounds for B samples."""
        B = inputs.shape[0]
        device = outputs.device
        
        if self.spec.lb is None or self.spec.ub is None:
            return [None] * B
        
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
    ) -> List[Optional[Counterexample]]:
        """Check if linear constraint c^T y <= d is violated for B samples."""
        B = inputs.shape[0]
        device = outputs.device
        
        if self.spec.c is None or self.spec.d is None:
            return [None] * B
        
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
