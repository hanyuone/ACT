"""
Seed corpus management for ACTFuzzer.

Implements AFL-style seed scheduling with energy-based prioritization.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

from act.front_end.spec_creator_base import LabeledInputTensor


@dataclass
class FuzzingSeed:
    """
    Fuzzing seed with metadata.
    
    Attributes:
        tensor: Input tensor (current, may be mutated)
        original_tensor: Original clean image (NEVER mutated, for visualization)
        original_index: Index in the original dataset (for tracking)
        label: Ground truth label (if available)
        energy: Seed energy (higher = more interesting)
        depth: How many mutations from original seed
        parent_id: ID of parent seed (for provenance tracking)
    """
    tensor: torch.Tensor
    original_tensor: Optional[torch.Tensor] = None  # Original clean image
    original_index: Optional[int] = None  # Index in loader.images
    label: Optional[int] = None
    energy: float = 1.0
    depth: int = 0
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique ID and set original_tensor if not provided."""
        self.id = f"seed_{id(self)}"
        # If original_tensor not set, use tensor (for initial seeds)
        if self.original_tensor is None:
            self.original_tensor = self.tensor.clone()


class SeedCorpus:
    """
    AFL-style seed corpus with energy-based scheduling.
    
    Features:
    - Energy-based seed selection (higher energy = more likely to select)
    - Random selection as fallback
    - Automatic seed deduplication (by tensor hash)
    
    Example:
        >>> corpus = SeedCorpus(initial_seeds, strategy="energy")
        >>> 
        >>> # Select seed for mutation
        >>> seed = corpus.select()
        >>> 
        >>> # Add interesting seed back
        >>> new_seed = FuzzingSeed(mutated_tensor, label=5, energy=10.0)
        >>> corpus.add(new_seed)
    """
    
    def __init__(self, 
                 initial_seeds: List[LabeledInputTensor],
                 strategy: str = "energy"):
        """
        Initialize seed corpus.
        
        Args:
            initial_seeds: Initial seeds from spec creators
            strategy: Selection strategy ("energy" or "random")
        """
        self.strategy = strategy
        self.seeds: List[FuzzingSeed] = []
        self.seen_hashes: set = set()
        
        # Convert LabeledInputTensor to FuzzingSeed
        for i, labeled_tensor in enumerate(initial_seeds):
            tensor_cpu = labeled_tensor.tensor.cpu()
            seed = FuzzingSeed(
                tensor=tensor_cpu,
                original_tensor=tensor_cpu.clone(),  # Store original
                original_index=i,  # Track index in dataset
                label=int(labeled_tensor.label.item()) if isinstance(labeled_tensor.label, torch.Tensor) else labeled_tensor.label,
                energy=1.0,
                depth=0
            )
            self._add_internal(seed)
    
    def _add_internal(self, seed: FuzzingSeed):
        """Internal add without deduplication check."""
        self.seeds.append(seed)
        # Track hash for deduplication
        tensor_hash = self._hash_tensor(seed.tensor)
        self.seen_hashes.add(tensor_hash)
    
    def _hash_tensor(self, tensor: torch.Tensor) -> int:
        """Compute hash of tensor for deduplication."""
        # Simple hash based on tensor values
        return hash(tensor.flatten().cpu().numpy().tobytes())
    
    def select(self) -> FuzzingSeed:
        """
        Select next seed based on strategy.
        
        Returns:
            Selected FuzzingSeed
        """
        if not self.seeds:
            raise ValueError("Corpus is empty!")
        
        if self.strategy == "energy":
            # Weighted random selection by energy
            energies = np.array([s.energy for s in self.seeds])
            
            # Avoid division by zero
            if energies.sum() == 0:
                energies = np.ones_like(energies)
            
            # Normalize to probabilities
            probs = energies / energies.sum()
            
            # Sample
            idx = np.random.choice(len(self.seeds), p=probs)
            return self.seeds[idx]
        
        else:  # random
            return np.random.choice(self.seeds)
    
    def add(self, seed: FuzzingSeed):
        """
        Add seed to corpus if it's interesting (not duplicate).
        
        Args:
            seed: FuzzingSeed to add
        """
        # Check for duplicate
        tensor_hash = self._hash_tensor(seed.tensor)
        
        if tensor_hash in self.seen_hashes:
            # Duplicate, skip
            return
        
        # Add new seed
        self._add_internal(seed)
    
    def __len__(self) -> int:
        """Return corpus size."""
        return len(self.seeds)
    
    def __iter__(self):
        """Iterate over seeds."""
        return iter(self.seeds)
    
    def get_stats(self) -> dict:
        """Get corpus statistics."""
        if not self.seeds:
            return {
                "total_seeds": 0,
                "avg_energy": 0.0,
                "max_depth": 0,
            }
        
        energies = [s.energy for s in self.seeds]
        depths = [s.depth for s in self.seeds]
        
        return {
            "total_seeds": len(self.seeds),
            "avg_energy": np.mean(energies),
            "max_energy": np.max(energies),
            "max_depth": max(depths),
            "strategy": self.strategy,
        }
