#!/usr/bin/env python3
# ===- act/pipeline/act2torch.py - ACT to Torch Converter ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# ACT → PyTorch Converter (Schema-Driven)
# =======================================
# REGISTRY (layer_schema.py) validates params; _ACT_TO_TORCH (below) maps
# LayerKind → torch.nn.Module for restoration.
#
# Layer Routing:
#   INPUT      → (skipped)
#   INPUT_SPEC → InputSpecLayer
#   <MODEL>    → _ACT_TO_TORCH[kind] (if defined)
#   ASSERT     → OutputSpecLayer
#
# ===---------------------------------------------------------------------===#

from typing import Optional, Dict, Any, Tuple
import importlib
import torch
import torch.nn as nn
import logging

from act.back_end.core import Net, Layer
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.util.device_manager import get_default_dtype, get_default_device

logger = logging.getLogger(__name__)

# ACT LayerKind → PyTorch nn.Module path.
# Layers not listed are skipped during restoration (wrapper, graph ops, functional-only).
_ACT_TO_TORCH = {
    LayerKind.DENSE.value: nn.Linear,
    LayerKind.CONV1D.value: nn.Conv1d,
    LayerKind.CONV2D.value: nn.Conv2d,
    LayerKind.CONV3D.value: nn.Conv3d,
    LayerKind.CONVTRANSPOSE2D.value: nn.ConvTranspose2d,
    LayerKind.MAXPOOL1D.value: nn.MaxPool1d,
    LayerKind.MAXPOOL2D.value: nn.MaxPool2d,
    LayerKind.MAXPOOL3D.value: nn.MaxPool3d,
    LayerKind.AVGPOOL1D.value: nn.AvgPool1d,
    LayerKind.AVGPOOL2D.value: nn.AvgPool2d,
    LayerKind.AVGPOOL3D.value: nn.AvgPool3d,
    LayerKind.ADAPTIVEAVGPOOL2D.value: nn.AdaptiveAvgPool2d,
    LayerKind.RELU.value: nn.ReLU,
    LayerKind.LRELU.value: nn.LeakyReLU,
    LayerKind.PRELU.value: nn.PReLU,
    LayerKind.SIGMOID.value: nn.Sigmoid,
    LayerKind.TANH.value: nn.Tanh,
    LayerKind.SOFTPLUS.value: nn.Softplus,
    LayerKind.SILU.value: nn.SiLU,
    LayerKind.GELU.value: nn.GELU,
    LayerKind.RELU6.value: nn.ReLU6,
    LayerKind.HARDTANH.value: nn.Hardtanh,
    LayerKind.HARDSIGMOID.value: nn.Hardsigmoid,
    LayerKind.HARDSWISH.value: nn.Hardswish,
    LayerKind.MISH.value: nn.Mish,
    LayerKind.SOFTSIGN.value: nn.Softsign,
    LayerKind.FLATTEN.value: nn.Flatten,
    LayerKind.EMBEDDING.value: nn.Embedding,
    LayerKind.RNN.value: nn.RNN,
    LayerKind.GRU.value: nn.GRU,
    LayerKind.LSTM.value: nn.LSTM,
    LayerKind.SOFTMAX.value: nn.Softmax,
    LayerKind.MHA.value: nn.MultiheadAttention,
}


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
            NotImplementedError: If the net has multi-predecessor layers.
            ValueError: If no valid PyTorch layers can be created
        """
        self._assert_chain_structure()

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
            params = act_layer.params

            # Handle wrapper layers specially
            if kind == LayerKind.INPUT.value:
                continue  # Skip INPUT layer (no-op)

            if kind == LayerKind.INPUT_SPEC.value:
                # Create InputSpecLayer for constraint checking
                from act.front_end.verifiable_model import InputSpecLayer
                from act.front_end.specs import InputSpec, InKind

                # Build InputSpec from ACT layer (kind/eps in params)
                kind_str = params["kind"]
                spec_kind = getattr(InKind, kind_str)  # Convert string to enum
                spec_dict = {"kind": spec_kind}
                if "eps" in params:
                    spec_dict["eps"] = params["eps"]

                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ["lb", "ub", "center", "A", "b"]:
                    if param_key in params:
                        tensor = params[param_key]
                        spec_dict[param_key] = tensor.to(
                            dtype=target_dtype, device=target_device
                        )

                spec = InputSpec(**spec_dict)
                # InputSpecLayer now always returns tuples
                torch_layers.append(InputSpecLayer(spec))
                has_input_spec = True
                continue

            elif kind == LayerKind.ASSERT.value:
                # Create OutputSpecLayer for constraint checking
                from act.front_end.verifiable_model import OutputSpecLayer
                from act.front_end.specs import OutputSpec, OutKind

                # Build OutputSpec from ACT layer
                kind_str = params["kind"]
                spec_kind = getattr(OutKind, kind_str)  # Convert string to enum
                spec_dict = {"kind": spec_kind}
                if "y_true" in params:
                    spec_dict["y_true"] = params["y_true"]
                if "margin" in params:
                    spec_dict["margin"] = params["margin"]
                if "d" in params:
                    spec_dict["d"] = params["d"]

                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ["c", "lb", "ub"]:
                    if param_key in params:
                        tensor = params[param_key]
                        spec_dict[param_key] = tensor.to(
                            dtype=target_dtype, device=target_device
                        )

                spec = OutputSpec(**spec_dict)
                # OutputSpecLayer now always returns tuples
                torch_layers.append(OutputSpecLayer(spec))
                has_output_spec = True
                continue

            # SCALE with BatchNorm decomposition → Restore BatchNorm
            if kind == LayerKind.SCALE.value and params.get(
                "is_batchnorm_decomposition"
            ):
                # Find paired BIAS layer
                bias_layer = self._find_paired_bias(i)
                if bias_layer is not None:
                    skip_layer_ids.add(bias_layer.id)

                bn_module = self._restore_batchnorm(act_layer)
                if bn_module is not None:
                    torch_layers.append(bn_module)
                    continue

            # Skip BIAS paired with SCALE (already handled)
            if kind == LayerKind.BIAS.value and params.get("paired_with_scale"):
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

        logger.info(
            f"Created VerifiableModel with {len(torch_layers)} layers "
            f"(INPUT_SPEC={has_input_spec}, OUTPUT_SPEC={has_output_spec})"
        )

        return model

    def _find_paired_bias(self, scale_idx: int) -> Optional[Layer]:
        """Find BIAS layer paired with SCALE at given index."""
        layers = self.act_net.layers
        for j in range(scale_idx + 1, len(layers)):
            layer = layers[j]
            if layer.kind == LayerKind.BIAS.value and layer.params.get(
                "paired_with_scale"
            ):
                return layer
            # Stop if we hit a non-BIAS layer
            if layer.kind != "BIAS":
                break
        return None

    def _restore_batchnorm(self, scale_layer: Layer) -> Optional[nn.Module]:
        """Restore BatchNorm from SCALE layer with batchnorm_* params."""
        params = scale_layer.params

        bn_module_path = params.get("batchnorm_module")
        if not bn_module_path:
            return None

        # Parse module path
        mod_name, cls_name = bn_module_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)

        # Create BatchNorm instance
        args = params.get("batchnorm_args", [])
        kwargs = params.get("batchnorm_kwargs", {})
        bn = cls(*args, **kwargs)

        # Load state from batchnorm_state
        bn_state = params.get("batchnorm_state", {})
        if bn_state:
            state_dict = {}
            for key in [
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ]:
                if key in bn_state:
                    state_dict[key] = bn_state[key]
            if state_dict:
                bn.load_state_dict(state_dict, strict=False)

        return bn

    def _build_from_schema(self, act_layer: Layer) -> Optional[nn.Module]:
        """Build PyTorch module from REGISTRY params + _ACT_TO_TORCH mapping."""
        kind = act_layer.kind
        params = act_layer.params

        if kind not in REGISTRY:
            raise ValueError(f"Layer kind '{kind}' not found in REGISTRY")
        spec = REGISTRY[kind]

        cls = _ACT_TO_TORCH.get(kind)
        if cls is None:
            if "requires_graph_restoration" in spec.get("params_optional", []):
                logger.warning(
                    f"Skipping {kind} layer (id={act_layer.id}): "
                    f"requires DAG structure, not supported in Sequential model"
                )
            return None

        # Build positional args from params_required (excluding tensors)
        # Tensors are auto-detected via isinstance() - they go to state_dict, not constructor
        args = []
        for key in spec.get("params_required", []):
            if key not in params:
                raise ValueError(
                    f"Layer '{kind}' (id={act_layer.id}) missing required param '{key}'"
                )
            # Skip tensor params - they go to state_dict
            if isinstance(params[key], torch.Tensor):
                continue
            args.append(params[key])

        # Build kwargs from params_optional (names match PyTorch directly)
        # Only include kwargs that PyTorch module accepts
        kwargs = {}
        # Check if bias exists in params to set bias kwarg for Linear/Conv
        if "bias" in act_layer.params:
            kwargs["bias"] = True
        elif kind in (
            LayerKind.DENSE.value,
            LayerKind.CONV1D.value,
            LayerKind.CONV2D.value,
            LayerKind.CONV3D.value,
        ):
            kwargs["bias"] = False
        # Pass through common kwargs from params
        for key in (
            "stride",
            "padding",
            "dilation",
            "groups",
            "start_dim",
            "negative_slope",
        ):
            if key in params:
                kwargs[key] = params[key]

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

    def _assert_chain_structure(self) -> None:
        """Fail-loud on DAG nets; silent drop would break soundness."""
        preds = getattr(self.act_net, "preds", {}) or {}
        dag_layers = [
            (lid, self.act_net.by_id[lid].kind)
            for lid, parents in preds.items()
            if parents and len(parents) > 1
        ]
        if dag_layers:
            preview = dag_layers[:5]
            suffix = "..." if len(dag_layers) > 5 else ""
            raise NotImplementedError(
                f"ACTToTorch currently supports chain networks only. "
                f"Found {len(dag_layers)} multi-predecessor layer(s): "
                f"{preview}{suffix}. DAG support is deferred — this is "
                f"intentional fail-loud to prevent silent unsoundness."
            )
