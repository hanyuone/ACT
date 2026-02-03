#!/usr/bin/env python3
#===- act/pipeline/act2torch.py - ACT to Torch Converter ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# ACT → PyTorch Converter (Schema-Driven)
# =======================================
# Single source of truth: layer_schema.py REGISTRY
#
# Design:
#   - Zero mapping tables: torch_module, params, meta all from REGISTRY
#   - Param names match PyTorch: weight, bias, stride, padding, etc.
#
# Layer Routing:
#   INPUT      → (skipped)
#   INPUT_SPEC → InputSpecLayer
#   <MODEL>    → REGISTRY[kind]['torch_module'] (if defined)
#   ASSERT     → OutputSpecLayer
#
#===---------------------------------------------------------------------===#

from typing import Optional, Dict, Any, Tuple
import importlib
import torch
import torch.nn as nn
import logging

from act.back_end.core import Net, Layer
from act.back_end.layer_schema import REGISTRY
from act.util.device_manager import get_default_dtype, get_default_device

logger = logging.getLogger(__name__)


class ACTToTorch:
    """
    Convert ACT Net to PyTorch nn.Module using dynamic restoration.
    
    Usage:
        converter = ACTToTorch(act_net)
        model = converter.run()  # Returns VerifiableModel
    """
    
    def __init__(self, act_net: Net):
        """
        Initialize converter with ACT Net.
        
        Args:
            act_net: ACT Net object (contains architecture + weights)
        
        Raises:
            TypeError: If act_net is not a Net instance
        """
        if not isinstance(act_net, Net):
            raise TypeError(f"ACTToTorch expects a Net object, got {type(act_net)}")
        self.act_net = act_net
    
    def run(self) -> nn.Module:
        """
        Convert ACT Net to PyTorch nn.Module.
        
        Iterates through ACT layers, creates corresponding PyTorch layers,
        transfers weights, and assembles into VerifiableModel model.
        
        Returns:
            VerifiableModel model with embedded constraint checking
        
        Raises:
            ValueError: If no valid PyTorch layers can be created
        """
        torch_layers = []
        has_input_spec = False
        has_output_spec = False
        
        # Get target dtype/device once for all tensor conversions
        target_dtype = get_default_dtype()
        target_device = get_default_device()
        
        # Track layers to skip (e.g., BIAS paired with SCALE for BatchNorm)
        skip_layer_ids = set()
        
        for i, act_layer in enumerate(self.act_net.layers):
            # Skip layers marked for skipping
            if act_layer.id in skip_layer_ids:
                continue
            
            kind = act_layer.kind
            meta = act_layer.meta
            
            # Handle wrapper layers specially
            if kind == 'INPUT':
                continue  # Skip INPUT layer (no-op)
            
            if kind == 'INPUT_SPEC':
                # Create InputSpecLayer for constraint checking
                from act.front_end.verifiable_model import InputSpecLayer
                from act.front_end.specs import InputSpec, InKind
                
                # Build InputSpec from ACT layer (kind/eps in params)
                params = act_layer.params
                kind_str = params['kind']
                spec_kind = getattr(InKind, kind_str)  # Convert string to enum
                spec_dict = {'kind': spec_kind}
                if 'eps' in params:
                    spec_dict['eps'] = params['eps']
                
                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ['lb', 'ub', 'center', 'A', 'b']:
                    if param_key in params:
                        tensor = params[param_key]
                        spec_dict[param_key] = tensor.to(dtype=target_dtype, device=target_device)
                
                spec = InputSpec(**spec_dict)
                # InputSpecLayer now always returns tuples
                torch_layers.append(InputSpecLayer(spec))
                has_input_spec = True
                continue
            
            elif kind == 'ASSERT':
                # Create OutputSpecLayer for constraint checking
                from act.front_end.verifiable_model import OutputSpecLayer
                from act.front_end.specs import OutputSpec, OutKind
                
                # Build OutputSpec from ACT layer
                kind_str = meta['kind']
                spec_kind = getattr(OutKind, kind_str)  # Convert string to enum
                spec_dict = {'kind': spec_kind}
                if 'y_true' in meta:
                    spec_dict['y_true'] = meta['y_true']
                if 'margin' in meta:
                    spec_dict['margin'] = meta['margin']
                if 'd' in meta:
                    spec_dict['d'] = meta['d']
                
                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ['c', 'lb', 'ub']:
                    if param_key in act_layer.params:
                        tensor = act_layer.params[param_key]
                        spec_dict[param_key] = tensor.to(dtype=target_dtype, device=target_device)
                
                spec = OutputSpec(**spec_dict)
                # OutputSpecLayer now always returns tuples
                torch_layers.append(OutputSpecLayer(spec))
                has_output_spec = True
                continue
            
            # SCALE with BatchNorm decomposition → Restore BatchNorm
            if kind == 'SCALE' and meta.get('is_batchnorm_decomposition'):
                # Find paired BIAS layer
                bias_layer = self._find_paired_bias(i)
                if bias_layer is not None:
                    skip_layer_ids.add(bias_layer.id)
                
                bn_module = self._restore_batchnorm(act_layer)
                if bn_module is not None:
                    torch_layers.append(bn_module)
                    continue
            
            # Skip BIAS paired with SCALE (already handled)
            if kind == 'BIAS' and meta.get('paired_with_scale'):
                continue
            
            # Schema-driven restoration for all other layers
            torch_layer = self._build_from_schema(act_layer)
            if torch_layer is None:
                # Skip if restore kind is 'skip' or 'graph'
                continue
            
            torch_layers.append(torch_layer)
        
        if not torch_layers:
            raise ValueError("No valid PyTorch layers found in ACT Net")
        
        # Return VerifiableModel for automatic constraint checking
        from act.front_end.verifiable_model import VerifiableModel
        model = VerifiableModel(*torch_layers)
        model.eval()  # Set to evaluation mode by default
        
        logger.info(f"Created VerifiableModel with {len(torch_layers)} layers "
                   f"(INPUT_SPEC={has_input_spec}, OUTPUT_SPEC={has_output_spec})")
        
        return model
    
    def _find_paired_bias(self, scale_idx: int) -> Optional[Layer]:
        """Find BIAS layer paired with SCALE at given index."""
        layers = self.act_net.layers
        for j in range(scale_idx + 1, len(layers)):
            layer = layers[j]
            if layer.kind == 'BIAS' and layer.meta.get('paired_with_scale'):
                return layer
            # Stop if we hit a non-BIAS layer
            if layer.kind != 'BIAS':
                break
        return None
    
    def _restore_batchnorm(self, scale_layer: Layer) -> Optional[nn.Module]:
        """Restore BatchNorm from SCALE layer with batchnorm_* metadata."""
        meta = scale_layer.meta

        bn_module_path = meta.get('batchnorm_module')
        if not bn_module_path:
            return None

        # Parse module path
        mod_name, cls_name = bn_module_path.rsplit('.', 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)

        # Create BatchNorm instance
        args = meta.get('batchnorm_args', [])
        kwargs = meta.get('batchnorm_kwargs', {})
        bn = cls(*args, **kwargs)

        # Load state from batchnorm_state
        bn_state = meta.get('batchnorm_state', {})
        if bn_state:
            state_dict = {}
            for key in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                if key in bn_state:
                    state_dict[key] = bn_state[key]
            if state_dict:
                bn.load_state_dict(state_dict, strict=False)

        return bn

    def _build_from_schema(self, act_layer: Layer) -> Optional[nn.Module]:
        """
        Build PyTorch module from layer_schema.py REGISTRY (single source of truth).

        Schema provides everything needed:
          - torch_module: PyTorch module path (e.g., "torch.nn.Linear")
          - meta_required: Constructor positional args
          - meta_optional: Constructor kwargs (names match PyTorch)
          - params: State dict keys (names match PyTorch: "weight", "bias")
        
        Layers without torch_module are skipped (INPUT, ASSERT, graph ops, etc.)
        """
        kind = act_layer.kind
        meta = act_layer.meta
        
        # Get schema spec - single source of truth
        if kind not in REGISTRY:
            raise ValueError(f"Layer kind '{kind}' not found in REGISTRY")
        spec = REGISTRY[kind]

        # Check for torch_module in schema - layers without it are skipped
        torch_module = spec.get('torch_module')
        if not torch_module:
            # Graph layers (ADD, MUL, CONCAT) require DAG structure - warn user
            if 'requires_graph_restoration' in spec.get('meta_optional', []):
                logger.warning(f"Skipping {kind} layer (id={act_layer.id}): "
                              f"requires DAG structure, not supported in Sequential model")
            return None

        # Parse module path: "torch.nn.Linear" -> ("torch.nn", "Linear")
        mod_name, cls_name = torch_module.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)

        # Build positional args from meta_required
        args = []
        for key in spec.get('meta_required', []):
            if key not in meta:
                raise ValueError(f"Layer '{kind}' (id={act_layer.id}) missing required meta '{key}'")
            args.append(meta[key])

        # Build kwargs from meta_optional (names match PyTorch directly)
        # Only include kwargs that PyTorch module accepts
        kwargs = {}
        # Check if bias exists in params to set bias kwarg for Linear/Conv
        if "bias" in act_layer.params:
            kwargs["bias"] = True
        elif kind in ("DENSE", "CONV1D", "CONV2D", "CONV3D"):
            kwargs["bias"] = False
        # Pass through common kwargs from meta
        for key in ("stride", "padding", "dilation", "groups", "start_dim"):
            if key in meta:
                kwargs[key] = meta[key]

        # Create module instance
        m = cls(*args, **kwargs)

        # Load state_dict directly (param names already match PyTorch)
        if act_layer.params:
            state_dict = {}
            for key, value in act_layer.params.items():
                if not isinstance(value, torch.Tensor):
                    continue
                state_dict[key] = value

            if state_dict:
                m.load_state_dict(state_dict, strict=False)

        return m
