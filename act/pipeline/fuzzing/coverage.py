"""
Coverage tracking for ACTFuzzer.

Tracks neuron coverage (method-level metrics) during fuzzing to guide exploration.
Implements two coverage strategies:
1. BestInputCov (BIC): per-input mutation threshold coverage (stores only per-input coverage values; a neuron is marked covered once |activation| > threshold. Coverage = (#covered) / (#total neurons)).
2. GlobalCov (GLC): global union threshold coverage (a neuron stays covered once it exceeds the threshold across all mutated inputs).

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple, List
import torch
import torch.nn as nn


NeuronId = Tuple[str, int]


class CoverageStrategy(ABC):
    """
    Coverage strategy interface.

    A strategy owns its own coverage state (covered set, totals, stats) and can be
    plugged into a tracker/engine (similar to `MutationStrategy` + `MutationEngine`).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        self.model = model
        self.threshold = threshold

    @abstractmethod
    def update(self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]) -> float:
        """Update coverage with new activations; returns coverage delta (0..1)."""
        raise NotImplementedError

    @abstractmethod
    def get_coverage(self) -> float:
        """Return coverage in [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return strategy-specific coverage stats (JSON friendly)."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal coverage state."""
        raise NotImplementedError

    # Optional capabilities (implemented only by some strategies)
    def get_uncovered_neurons(self) -> Set[NeuronId]:
        raise NotImplementedError

    def get_covered_neurons(self) -> Set[NeuronId]:
        raise NotImplementedError


def _activation_to_neuron_matrix(activation: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,...) activation tensor to (N, neurons) matrix for coverage.
    
    - (N, neurons) [dim==2]: already correct → return as-is
    - (N, C, H, W) [dim==4]: max-pool spatial dims → (N, C)
    - Other: flatten all but batch dim → (N, K)
    """
    if activation.dim() == 2:
        return activation  # (N, neurons)
    if activation.dim() == 4:
        return activation.abs().amax(dim=(2, 3))  # (N, C)
    return activation.flatten(start_dim=1)  # (N, K)


class BestInputCov(CoverageStrategy):
    """
    Per-input neuron coverage.

    - Each update() computes coverage for that specific input only.
    - We store per-input coverage values (history), not a global union of covered neurons.
    - get_coverage() returns the best (max) per-input coverage seen so far (monotonic).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)
        self._layer_neuron_counts: Dict[str, int] = {}
        self.coverage_history: List[float] = []
        self.last_input_coverage: float = 0.0
        self.best_input_coverage: float = 0.0

    def update(self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]) -> float:
        """
        Update per-input neuron coverage with batch activations.
        
        Computes coverage per sample, tracks best (max) seen so far.
        Handles both single (N=1) and batch (N>1) activations.
        """
        # Register layers and count neurons
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)
            if mat.numel() == 0:
                continue
            self._layer_neuron_counts.setdefault(layer_name, mat.shape[1])
        
        total_neurons = int(sum(self._layer_neuron_counts.values()))
        if total_neurons == 0:
            return 0.0
        
        old_best = float(self.best_input_coverage)
        N = input_tensor.shape[0]
        
        for i in range(N):
            covered_count = 0
            for layer_name, activation in activations.items():
                mat = _activation_to_neuron_matrix(activation)
                fired = (mat[i].abs() > float(self.threshold)).sum().item()
                covered_count += int(fired)
            
            sample_cov = covered_count / total_neurons
            self.coverage_history.append(float(sample_cov))
            self.last_input_coverage = float(sample_cov)
            if sample_cov > self.best_input_coverage:
                self.best_input_coverage = float(sample_cov)
        
        return max(0.0, float(self.best_input_coverage) - old_best)

    def get_coverage(self) -> float:
        return float(self.best_input_coverage)

    def get_stats(self) -> Dict[str, Any]:
        n = len(self.coverage_history)
        avg = (sum(self.coverage_history) / n) if n else 0.0
        return {
            "coverage": float(self.get_coverage()),
            "inputs_seen": int(n),
            "last_input_coverage": float(self.last_input_coverage),
            "best_input_coverage": float(self.best_input_coverage),
            "avg_input_coverage": float(avg),
            "total_neurons_seen": int(sum(self._layer_neuron_counts.values())),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def reset(self) -> None:
        self._layer_neuron_counts.clear()
        self.coverage_history.clear()
        self.last_input_coverage = 0.0
        self.best_input_coverage = 0.0


class GlobalCov(CoverageStrategy):
    """
    Global union neuron coverage.

    A neuron is covered if it has fired at least once across all inputs.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)

        self.all_neurons: Set[NeuronId] = set()
        self._layer_neuron_counts: Dict[str, int] = {}
        self.covered_neurons: Set[NeuronId] = set()
        self.last_newly_covered_count: int = 0

    def _ensure_layer_registered(self, layer_name: str, neuron_count: int) -> None:
        neuron_count = int(neuron_count)
        if neuron_count <= 0:
            return
        if layer_name in self._layer_neuron_counts:
            return
        self._layer_neuron_counts[layer_name] = neuron_count
        for idx in range(neuron_count):
            self.all_neurons.add((layer_name, idx))

    def update(self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]) -> float:
        """
        Update global union neuron coverage with batch activations.
        
        A neuron is covered if it fired in ANY sample across ALL inputs seen.
        Handles both single (N=1) and batch (N>1) activations.
        """
        old_count = len(self.covered_neurons)
        
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)  # (N, neurons)
            if mat.numel() == 0:
                continue
            n_neurons = mat.shape[1]
            self._ensure_layer_registered(layer_name, n_neurons)
            # Union across batch: any sample firing counts
            fired_mask = mat.abs() > float(self.threshold)  # (N, neurons)
            fired_any = fired_mask.any(dim=0)  # (neurons,)
            fired_indices = fired_any.nonzero(as_tuple=True)[0].tolist()
            for idx in fired_indices:
                self.covered_neurons.add((layer_name, int(idx)))
        
        new_count = len(self.covered_neurons)
        self.last_newly_covered_count = new_count - old_count
        total_neurons = len(self.all_neurons)
        return (self.last_newly_covered_count / total_neurons) if total_neurons > 0 else 0.0

    def get_coverage(self) -> float:
        total_neurons = len(self.all_neurons)
        if total_neurons == 0:
            return 0.0
        return len(self.covered_neurons) / total_neurons

    def get_uncovered_neurons(self) -> Set[NeuronId]:
        return self.all_neurons - self.covered_neurons

    def get_covered_neurons(self) -> Set[NeuronId]:
        return self.covered_neurons.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "coverage": float(self.get_coverage()),
            "covered_neurons": int(len(self.covered_neurons)),
            "total_neurons": int(len(self.all_neurons)),
            "last_newly_covered": int(self.last_newly_covered_count),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def reset(self) -> None:
        self.covered_neurons.clear()
        self.all_neurons.clear()
        self._layer_neuron_counts.clear()
        self.last_newly_covered_count = 0

class CoverageTracker:
    """
    Coverage engine that delegates to coverage strategies.
    
    Supports runtime strategy switching via update(strategy=...).
    Strategies are lazily initialized on first use.
    """

    _REGISTRY = {"BestInputCov": BestInputCov, "GlobalCov": GlobalCov}

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.1,
        strategy: str = "BestInputCov"
    ):
        self.model = model
        self.threshold = threshold
        self.strategy = strategy
        self._strategies: Dict[str, CoverageStrategy] = {}
        
        if strategy not in self._REGISTRY:
            raise ValueError(f"Unknown coverage strategy '{strategy}'. Valid: {list(self._REGISTRY.keys())}")

    def _get_strategy(self, name: str) -> CoverageStrategy:
        """Get or create strategy by name (lazy init)."""
        if name not in self._strategies:
            if name not in self._REGISTRY:
                raise ValueError(f"Unknown coverage strategy '{name}'. Valid: {list(self._REGISTRY.keys())}")
            self._strategies[name] = self._REGISTRY[name](model=self.model, threshold=self.threshold)
        return self._strategies[name]

    def update(
        self,
        input_tensor: torch.Tensor,
        activations: Dict[str, torch.Tensor],
        strategy: Optional[str] = None
    ) -> float:
        """Update coverage. Use strategy param to override default."""
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).update(input_tensor, activations)

    def get_coverage(self, strategy: Optional[str] = None) -> float:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_coverage()

    def get_stats(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        s = strategy if strategy is not None else self.strategy
        return {"strategy": s, **self._get_strategy(s).get_stats()}

    def get_uncovered_neurons(self, strategy: Optional[str] = None) -> Set[NeuronId]:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_uncovered_neurons()

    def get_covered_neurons(self, strategy: Optional[str] = None) -> Set[NeuronId]:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_covered_neurons()

