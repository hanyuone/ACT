#===- act/front_end/verifiable_model.py - PyTorch Wrapper Layers -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Batch-native PyTorch wrapper layers for spec-free verification.
#   Embeds constraints directly into models, enabling efficient batched
#   verification where N samples are checked in a single forward pass.
#
# Key Features:
#   - Batch-native: Processes (N, ...) tensors, verifies N samples 
#   - Spec-free: Constraints embedded in model, not external files
#   - PyTorch-native: Full nn.Module compatibility
#   - Automatic verification: Returns per-sample constraint satisfaction
#
# Core Layers:
#
#   InputLayer(labeled_input, shape=(N, C, H, W), dtype=...)
#     Declares batched input with metadata. No-op at inference.
#
#   InputSpecLayer(spec=InputSpec(lb=(N,...), ub=(N,...)))
#     Checks N input constraints. Returns (x, satisfied(N,), explanation).
#
#   OutputSpecLayer(spec=OutputSpec(y_true=(N,), margin=(N,)))
#     Checks N output properties. Returns (y, satisfied(N,), explanation).
#
# Batched Workflow:
#   # 1. Build model with batched specs (N=3 samples)
#   model = VerifiableModel(
#       InputLayer(labeled_input, shape=(3, 1, 28, 28), dtype=torch.float32),
#       InputSpecLayer(InputSpec(kind=InKind.BOX, lb=(3,1,28,28), ub=(3,1,28,28))),
#       nn.Flatten(start_dim=1),
#       nn.Linear(784, 10),
#       OutputSpecLayer(OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[5,3,7]))
#   )
#
#   # 2. Run: 3 samples verified in one forward pass
#   result = model(batch_input)  # batch_input: (3, 1, 28, 28)
#   # Returns: {output: (3,10), input_satisfied: bool, output_satisfied: bool}
#
#   # 3. Convert to ACT for formal verification
#   from act.pipeline.torch2act import TorchToACT
#   act_net = TorchToACT(model).run()
#
# Supported Constraints:
#   InKind: BOX, LINF_BALL, LIN_POLY
#   OutKind: TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.layer_util import create_layer


def prod(seq: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in seq:
        p *= s
    return p


class VerifiableModel(nn.Sequential):
    """
    verification wrapper for PyTorch models.
    
    Processes batched tensors (N, ...) where N samples are verified independently
    in a single forward pass. Collects constraint results from spec layers.
    
    Batched Architecture:
        Input: (N, C, H, W) or (N, features)
        → InputSpecLayer: checks each sample's input constraints
        → Model layers: processes entire batch
        → OutputSpecLayer: checks each sample's output properties
        Output: Aggregated bool (all samples satisfied)
    
    Args:
        *args: Layers (InputLayer, InputSpecLayer, model, OutputSpecLayer)
    
    Returns:
        Dict with:
        - output: (N, ...) tensor
        - input_satisfied: bool (all N samples satisfy input constraints)
        - output_satisfied: bool (all N samples satisfy output properties)
        - input_explanation/output_explanation: Human-readable results
    """
    
    # Class-level strict mode setting (shared across all instances)
    _strict_mode: bool = False
    
    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def set_strict_mode(cls, enabled: bool) -> None:
        """
        When enabled, forward() raises ValueError on input/output violations.
        """
        cls._strict_mode = enabled
    
    @classmethod
    def get_strict_mode(cls) -> bool:
        return cls._strict_mode
    
    def forward(self, x):
        """
        Forward pass with automatic constraint checking.
        
        Intercepts tuple returns from InputSpecLayer/OutputSpecLayer
        and collects verification results.
        """
        input_satisfied = True
        input_explanation = "No INPUT_SPEC layer"
        output_satisfied = True
        output_explanation = "No OUTPUT_SPEC layer"
        
        # Process through all layers
        for i, module in enumerate(self):
            result = module(x)
            
            # Check if layer returned constraint checking tuple
            if isinstance(result, tuple) and len(result) == 3:
                x, satisfied, explanation = result
                
                # Identify if this is input or output spec layer
                # Input spec layers typically appear early (first few layers)
                # Output spec layers typically appear at the end
                if i < len(self) // 2:  # First half = likely INPUT_SPEC
                    input_satisfied = satisfied
                    input_explanation = explanation
                else:  # Second half = likely OUTPUT_SPEC
                    output_satisfied = satisfied
                    output_explanation = explanation
            else:
                # Regular layer, just pass through
                x = result
        
        # Convert tensor satisfied to bool for dict output (backward compatible)
        # Spec layers now always return tensor, convert here
        if isinstance(input_satisfied, torch.Tensor):
            input_satisfied = bool(input_satisfied.all().item())
        if isinstance(output_satisfied, torch.Tensor):
            output_satisfied = bool(output_satisfied.all().item())
        
        # Strict mode: raise on constraint violations
        if self._strict_mode:
            if not input_satisfied:
                print(f"[STRICT MODE] {input_explanation}")
                raise ValueError(
                    f"Input constraint violated in strict mode: {input_explanation}"
                )
            if not output_satisfied:
                print(f"[STRICT MODE] {output_explanation}")
                raise ValueError(
                    f"Output constraint violated in strict mode: {output_explanation}"
                )
        
        # Return comprehensive verification result
        return {
            'output': x,
            'input_satisfied': input_satisfied,
            'input_explanation': input_explanation,
            'output_satisfied': output_satisfied,
            'output_explanation': output_explanation
        }


class InputLayer(nn.Module):
    """
    Declares batched input with metadata. No-op during forward pass.
    
    Batch Format:
        shape: (N, C, H, W) or (N, features) where N is batch size
        Allocates N * per_sample_vars for ACT verification
        Stores labeled_input: (tensor, label) with N samples
    
    Args:
        labeled_input: LabeledInputTensor with (N, ...) tensor and labels
        shape: Input shape including batch dimension
        dtype: REQUIRED - affects verification precision/range
        dataset_name, layout, etc.: Optional metadata
    
    Forward:
        x → x (pass-through, metadata only)
    """
    def __init__(
        self,
        labeled_input: "LabeledInputTensor",  # Complete input sample with label
        shape: Tuple[int, ...],
        dtype: torch.dtype,  # REQUIRED: Critical for verification soundness
        desc: str = "input",
        # Tier 1: Essential metadata (strongly recommended)
        layout: Optional[str] = None,
        dataset_name: Optional[str] = None,
        # Tier 2: Important metadata
        num_classes: Optional[int] = None,
        value_range: Optional[Tuple[float, float]] = None,
        scale_hint: Optional[str] = None,
        distribution: Optional[str] = None,  # "uniform", "normal", "normalized", "unknown", or custom
        # Tier 3: Optional metadata
        sample_id: Optional[Union[int, str]] = None,
        domain: Optional[str] = None,
        channels: Optional[int] = None,
    ):
        super().__init__()
        # Allow any batch size >= 1 for verification
        if shape[0] < 1:
            raise ValueError(f"Batch size must be >= 1, got {shape[0]}")
        
        # Core attributes (dtype now required)
        self.shape = tuple(shape)
        self._batched = shape[0] > 1  # Track if batched for to_act_layers
        self.dtype = dtype  # REQUIRED
        self.desc = desc
        
        # Tier 1: Essential metadata
        self.layout = layout
        self.dataset_name = dataset_name
        
        # Tier 2: Important metadata
        self.num_classes = num_classes
        self.value_range = tuple(value_range) if value_range else None
        self.scale_hint = scale_hint
        self.distribution = distribution
        
        # Tier 3: Optional metadata
        self.sample_id = sample_id
        self.domain = domain
        self.channels = channels
        
        # Store labeled input as PyTorch buffers
        # Assumes device/dtype already initialized by caller (e.g., via initialize_device)
        self.register_buffer("_input_tensor", labeled_input.tensor)
        
        # Store label (tensor or None)
        # Note: Labels use int64 dtype
        if labeled_input.label is not None:
            if not isinstance(labeled_input.label, torch.Tensor):
                raise TypeError(
                    f"labeled_input.label must be torch.Tensor or None, "
                    f"got {type(labeled_input.label).__name__}"
                )
            self.register_buffer("_label_tensor", labeled_input.label.reshape(-1))
        else:
            # No label provided
            self.register_buffer("_label_tensor", torch.tensor([], dtype=torch.int64))
        
        self._validate_schema()
    
    @property
    def labeled_input(self) -> "LabeledInputTensor":
        """Get the complete labeled input (tensor + tensor label)."""
        from act.front_end.spec_creator_base import LabeledInputTensor
        # Return tensor label directly
        label = self._label_tensor if self._label_tensor.numel() > 0 else None
        return LabeledInputTensor(tensor=self._input_tensor, label=label)
    
    @property
    def input_tensor(self) -> torch.Tensor:
        """Get input tensor (convenience accessor)."""
        return self._input_tensor
    
    @property
    def label(self) -> torch.Tensor:
        """Get label tensor (always (N,) shape)."""
        return self._label_tensor

    def _validate_schema(self):
        """Validate parameters against INPUT layer schema"""
        schema = REGISTRY[LayerKind.INPUT.value]
        
        # Collect params (optional: labeled_input as unified object)
        params = {
            "labeled_input": self.labeled_input,
        }
        
        # Collect meta (everything else - dtype now REQUIRED)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype)  # REQUIRED - must always be present
        }
        
        # Add non-default desc
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Add Tier 1-3 metadata (only if not None)
        if self.layout is not None:
            meta["layout"] = self.layout
        if self.dataset_name is not None:
            meta["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            meta["num_classes"] = self.num_classes
        if self.value_range is not None:
            meta["value_range"] = self.value_range
        if self.scale_hint is not None:
            meta["scale_hint"] = self.scale_hint
        if self.distribution is not None:
            meta["distribution"] = self.distribution
        if self.sample_id is not None:
            meta["sample_id"] = self.sample_id
        if self.domain is not None:
            meta["domain"] = self.domain
        if self.channels is not None:
            meta["channels"] = self.channels
        
        # Check required/optional params and meta
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputLayer missing required param: {key}")
        for key in params:
            if key not in schema["params_required"] + schema["params_optional"]:
                raise ValueError(f"InputLayer has unknown param: {key}")
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) and return (layers, out_vars)
        
        For batched inputs (shape[0] > 1), allocates batch_size * per_sample_vars.
        """
        batch_size = self.shape[0]
        per_sample = prod(self.shape[1:])
        N = batch_size * per_sample  # Total vars for all samples in batch
        out_vars = list(range(len(in_vars), len(in_vars) + N))
        
        # Collect params (optional: labeled_input for ACT serialization)
        params = {
            "labeled_input": self.labeled_input,
        }
        
        # Collect meta (dtype is REQUIRED, always present)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype),  # REQUIRED
            "batch_size": batch_size,  # Track batch size for batched verification
        }
        
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Add all optional metadata fields (only if not None)
        if self.layout is not None:
            meta["layout"] = self.layout
        if self.dataset_name is not None:
            meta["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            meta["num_classes"] = self.num_classes
        if self.value_range is not None:
            meta["value_range"] = self.value_range
        if self.scale_hint is not None:
            meta["scale_hint"] = self.scale_hint
        if self.distribution is not None:
            meta["distribution"] = self.distribution
        if self.sample_id is not None:
            meta["sample_id"] = self.sample_id
        if self.domain is not None:
            meta["domain"] = self.domain
        if self.channels is not None:
            meta["channels"] = self.channels
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars
        )
        return [layer], out_vars

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Return a summary of all metadata for debugging/logging"""
        return {
            "shape": self.shape,
            "desc": self.desc,
            "dtype": str(self.dtype),  # Always present (required)
            "layout": self.layout,
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "value_range": self.value_range,
            "scale_hint": self.scale_hint,
            "distribution": self.distribution,
            "label": self.label.tolist(),  # Convert tensor to list for readability
            "sample_id": self.sample_id,
            "domain": self.domain,
            "channels": self.channels,
            "input_tensor_shape": tuple(self.input_tensor.shape),
        }

    def __repr__(self) -> str:
        """Enhanced string representation with key metadata"""
        meta_str = f"shape={self.shape}"
        if self.dataset_name:
            meta_str += f", dataset={self.dataset_name}"
        # Display label in readable format
        label_display = self.label.tolist() if self.label.numel() > 1 else self.label.item()
        meta_str += f", label={label_display}"
        if self.layout:
            meta_str += f", layout={self.layout}"
        return f"InputLayer({meta_str})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputSpecLayer(nn.Module):
    """
    Batched input constraint checker (BOX, L∞, LIN_POLY).
    
    Batch Processing:
        Input: x with shape (N, C, H, W)
        Constraints: lb/ub/center with shape (N, C, H, W) - per-sample bounds
        Checks: Each sample x[i] verified independently against its bounds
        Output: (x, satisfied, explanation)
            - satisfied: (N,) bool tensor, one per sample
            - explanation: "✅ INPUT BOX: {n_ok}/{N} satisfied"
    
    Args:
        spec: InputSpec with batched constraint tensors
    
    Forward:
        x (N, ...) → (x, satisfied (N,), explanation str)
    """
    def __init__(self, spec: Optional[InputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or InputSpec(**kwargs)
        self.kind = self.spec.kind
        
        # Register eps as buffer for device mobility
        if self.spec.eps is not None:
            self.register_buffer("eps", self.spec.eps)
        else:
            self.eps = None

        # Register tensor fields as buffers so .to(device) works
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against INPUT_SPEC layer schema"""
        schema = REGISTRY[LayerKind.INPUT_SPEC.value]
        params = {"kind": self.kind}  # kind is now in params
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        if self.eps is not None:
            params["eps"] = self.eps
        
        # Check schema compliance
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputSpecLayer missing required param: {key}")
        for key in params:
            if key not in schema["params_required"] + schema["params_optional"]:
                raise ValueError(f"InputSpecLayer has unknown param: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - INPUT_SPEC doesn't create new vars"""
        params = {"kind": self.kind}  # kind is now in params
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        if self.eps is not None:
            params["eps"] = self.eps
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT_SPEC.value,
            params=params,
            meta={},  
            in_vars=in_vars,
            out_vars=in_vars  # INPUT_SPEC doesn't change variables
        )
        return [layer], in_vars

    def forward(self, x: torch.Tensor):
        """
        Forward pass with unified constraint checking for single or batched input.
        
        Returns:
            Tuple of (tensor, satisfied, explanation)
            - For batch=1 andbatch>1: satisfied is (batch,) bool tensor
        """
        if self.spec is None:
            return (x, True, "✅ INPUT: No constraints")
        
        batch_size = x.shape[0]
        
        if self.kind == InKind.BOX:
            if self.lb is None or self.ub is None:
                return (x, True, "⚠️ INPUT BOX: Missing lb/ub")
            
            lb, ub = self.lb, self.ub
            # model_synthesis.py ensures lb.shape[0] == batch_size

            lb_ok = (x >= lb).reshape(batch_size, -1).all(dim=1)  # (batch,)
            ub_ok = (x <= ub).reshape(batch_size, -1).all(dim=1)  # (batch,)
            satisfied = lb_ok & ub_ok  # (batch,) bool tensor
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT BOX: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LINF_BALL:
            if self.center is None or self.eps is None:
                return (x, True, "⚠️ INPUT L∞: Missing center/eps")
            
            center = self.center
            
            # Per-sample L∞ distance
            linf = (x - center).abs().reshape(batch_size, -1).max(dim=1)[0]  # (batch,)
            
            # Compare against eps threshold (tensor, supports batched comparison)
            satisfied = linf <= self.eps
            
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT L∞: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LIN_POLY:
            if self.A is None or self.b is None:
                return (x, True, "⚠️ INPUT LIN_POLY: Missing A/b")
            
            # LIN_POLY typically single sample; batched requires per-sample A,b
            x_flat = x.reshape(batch_size, -1)  # (batch, n_vars)
            # For now, apply same A,b to each sample
            residuals = x_flat @ self.A.T - self.b  # (batch, n_constraints)
            max_viol = residuals.max(dim=1)[0]  # (batch,)
            satisfied = max_viol <= 0
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT LIN_POLY: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (x, satisfied, explanation)
        
        else:
            return (x, True, f"⚠️ INPUT: Unknown kind {self.kind}")


class OutputSpecLayer(nn.Module):
    """
    Batched output property checker (TOP1_ROBUST, MARGIN_ROBUST, etc.).
    
    Batch Processing:
        Input: y with shape (N, n_classes) or (N, features)
        Properties: y_true (N,), margin (N,) - per-sample targets
        Checks: Each sample y[i] verified independently against its property
        Output: (y, satisfied, explanation)
            - satisfied: (N,) bool tensor, one per sample
            - explanation: "✅ OUTPUT TOP1: {n_ok}/{N} robust"
    
    Args:
        spec: OutputSpec with batched property tensors
    
    Forward:
        y (N, ...) → (y, satisfied (N,), explanation str)
    """
    def __init__(self, spec: Optional[OutputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or OutputSpec(**kwargs)
        self.kind = self.spec.kind
        self.d = self.spec.d
        self.meta = dict(self.spec.meta)
        
        # register as buffer for device mobility
        if self.spec.y_true is not None:
            self.register_buffer("y_true", self.spec.y_true)
        else:
            self.y_true = None
        
        if self.spec.margin is not None:
            self.register_buffer("margin", self.spec.margin)
        else:
            self.margin = None

        for name in ("c", "lb", "ub"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against ASSERT layer schema"""
        schema = REGISTRY[LayerKind.ASSERT.value]
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"OutputSpecLayer missing required meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - ASSERT doesn't create new vars"""
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.ASSERT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # ASSERT doesn't change variables
        )
        return [layer], in_vars

    def forward(self, y: torch.Tensor):
        """
        Forward pass with unified constraint checking for single or batched output.
        
        Returns:
            Tuple of (tensor, satisfied, explanation)
            For batch=1 and batch>1: satisfied is (batch,) bool tensor
        """
        if self.spec is None:
            return (y, True, "✅ OUTPUT: No constraints")
        
        batch_size = y.shape[0]
        
        if self.kind == OutKind.TOP1_ROBUST:
            if self.y_true is None:
                return (y, True, "⚠️ OUTPUT TOP1: Missing y_true")
            
            preds = y.argmax(dim=1)  # (batch,)
            satisfied = preds == self.y_true  # (batch,) bool tensor
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT TOP1: {n_ok}/{batch_size} robust"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.MARGIN_ROBUST:
            if self.y_true is None:
                return (y, True, "⚠️ OUTPUT MARGIN: Missing y_true")
            
            # y is (batch, n_classes)
            n_classes = y.shape[1]
            true_scores = y[torch.arange(batch_size, device=y.device), self.y_true]  # (batch,)
            
            # Mask out true class to get max of others
            mask = torch.ones_like(y, dtype=torch.bool)
            mask[torch.arange(batch_size, device=y.device), self.y_true] = False
            other_scores = y.masked_fill(~mask, float('-inf'))
            max_other = other_scores.max(dim=1)[0]  # (batch,)
            
            actual_margin = true_scores - max_other  # (batch,)
            margin_threshold = self.margin  # Always tensor now (even for n=1)
            satisfied = actual_margin >= margin_threshold  # (batch,) bool
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT MARGIN: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.LINEAR_LE:
            if self.c is None or self.d is None:
                return (y, True, "⚠️ OUTPUT LINEAR_LE: Missing c/d")
            
            # Works for both single and batched (c applied to each sample)
            y_2d = y.reshape(batch_size, -1)  # (batch, n_vars)
            c_typed = self.c.to(dtype=y_2d.dtype, device=y_2d.device)
            lhs = (y_2d @ c_typed)  # (batch,)
            satisfied = lhs <= self.d  # (batch,) bool
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT LINEAR_LE: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.RANGE:
            if self.lb is None or self.ub is None:
                return (y, True, "⚠️ OUTPUT RANGE: Missing lb/ub")
            
            lb, ub = self.lb, self.ub
            # ensures lb.shape[0] == batch_size
            # No broadcast needed - shapes always match
            
            lb_ok = (y >= lb).reshape(batch_size, -1).all(dim=1)  # (batch,)
            ub_ok = (y <= ub).reshape(batch_size, -1).all(dim=1)  # (batch,)
            satisfied = lb_ok & ub_ok  # (batch,) bool
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT RANGE: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        else:
            return (y, True, f"⚠️ OUTPUT: Unknown kind {self.kind}")