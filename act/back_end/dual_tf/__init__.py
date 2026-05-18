#===- act/back_end/dual_tf/__init__.py - Dual Transfer Functions --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Dual transfer functions module for Lagrangian dual bound (wong & kolter-style) computation.
#   Implements backward pass for certified bound computation.
#   - Precision driven: computes tight bounds on the dual objective by backward Lagrangian method 
#   - Adaptive Optimization: computes the bound via dual variables which can be on-demand optimized with gradient-based methods  
#   - Spurious counterexample: greedy spurious counterexample generation via linear boundary 
#
#===---------------------------------------------------------------------===#

# Disable pyright import-cycle error for this module (circular imports are intentional)
# pyright: reportImportCycles=false
"""Dual Transfer Function package.

## Primary Entry Point: `DualSolver.evaluate_spec`

Unified dispatcher over all supported OutputSpec kinds (LINEAR_LE, UNSAFE_LINEAR,
TOP1_ROBUST, MARGIN_ROBUST, RANGE). Encodes any spec as a batched `SpecBatch`
(B*M linear forms) and runs a single backward pass via `compute_bound`.

```python
from act.back_end.solver import DualSolver
from act.util.stats import SpecBatchResult
from act.front_end.specs import OutputSpec, OutKind

solver = DualSolver(tf=DualTF())
result: SpecBatchResult = solver.evaluate_spec(
    net, bounds_dict,
    OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=y_true),
    num_classes=10,
)
# result.margins: [B, K]  — per-class lower bounds on y_true - y_j
# result.certified: [B] bool
# result.min_slack: [B]  — legacy-compatible worst-case margin
```

## Result Layers (SpecBatchResult vs VerifyResult)

Two result types operate at different abstraction levels:

- **`SpecBatchResult`** (low-level, batched): direct numerical output from
  dual evaluation. Carries `[B, M]` margin tensor, per-cell slack, active-cell
  mask, and `[B]` certification bool. Used for robust training losses and
  intermediate computations.

- **`VerifyResult`** (`act.util.stats`, per-sample): high-level verification
  verdict with `status: VerifyStatus` enum (CERTIFIED / UNKNOWN / FALSIFIED /
  TIMEOUT / ...), optional counterexample, and metadata.

Convert via `SpecBatchResult.to_verify_results() -> List[VerifyResult]`. The
bridge maps `certified=True -> CERTIFIED`, `certified=False -> UNKNOWN`
(never FALSIFIED: dual bounds are sound but not complete; a negative slack
may reflect relaxation gap rather than a true violation).

## Gradient Flow (Robust Training)

All dual backward handlers and `compute_bound` / `evaluate_spec` /
`compute_robust_bound` honor the caller's gradient context via the
`enable_grad: bool = False` parameter (default off for verification).

```python
# Robust training loop
result = solver.evaluate_spec(net, bounds_dict, spec, enable_grad=True)
loss = -result.margins.mean()
loss.backward()  # gradients propagate through dense/conv weights
```

When `enable_grad=False` (default), execution is wrapped in
`torch.set_grad_enabled(False)`, preserving inference-path performance.

## Legacy: `compute_robust_bound`

First-class shortcut for classification robustness, retained for both
verification callers and robust training:

```python
min_slack, certified = solver.compute_robust_bound(
    net, bounds_dict, y_true, num_classes
)  # legacy tuple signature

# Or, for training:
result = solver.compute_robust_bound(
    net, bounds_dict, y_true, num_classes,
    margin=0.5, return_full=True, enable_grad=True,
)  # returns SpecBatchResult
```

Internally delegates to `evaluate_spec` with `TOP1_ROBUST` or
`MARGIN_ROBUST` spec kind.

## Batch Convention

All tensors at module boundaries follow the `[B, *layer_shape]` convention:

- `Bounds.lb`, `Bounds.ub`: `[B, *layer_shape]` (e.g. `[B, 128]`, `[B, C, H, W]`)
- `nu` (dual variable): `[B, *layer_shape]`
- `c` (objective coefficient): `[B, num_classes]`
- `contrib` (per-handler): `[B]`
- `compute_bound` return: `Tensor[B]` or `(Tensor[B], Tensor[B, *in_shape])`
- `compute_robust_bound` return: `(Tensor[B], Tensor[B] bool)` — NOT Python bool

### Input contract (deliberately asymmetric)

- `DualSolver.compute_bound` and `DualSolver.compute_robust_bound` are STRICT:
  they REQUIRE batched input `c: [B, num_classes]` and raise `ValueError` on
  1-D. For a single instance, use `.unsqueeze(0)` before calling and
  `.squeeze(0)` / `.item()` on results.

- `compute_forward_bounds` is LENIENT: it auto-promotes 1-D input
  (`input_lb.dim() < 2`) to `[1, *]` via `.unsqueeze(0)`, processes through the
  fully-batched internal path, and returns `bounds_dict` with every entry
  shaped `[B, *layer_shape]`. This preserves compatibility with experimental
  scripts and the notebook that pass 1-D input, without requiring call-site
  changes.

Internally, `compute_forward_bounds` has NO per-instance Python loop —
every `_fwd_*` handler accepts and produces batched `LinearBound`
(`A_{lb,ub}: [B, out_dim, in_dim]`, `b_{lb,ub}: [B, out_dim]`). See
`tf_forward.py` module docstring for handler details.
"""

# Core DualTF class + ADD/CONCAT dispatch (lives in dual_tf.py beside the class)
from .dual_tf import DualTF, backward_add, backward_concat, forward_add, forward_concat

# MLP batched kernels + dispatch (backward) and forward registry handlers
from .tf_mlp import (
    dual_relu_backward, dual_dense_backward, get_relu_masks,
    dual_bias_backward, dual_scale_backward, dual_bn_backward, dual_identity_backward,
    backward_dense, backward_relu, backward_bias, backward_scale,
    backward_bn, backward_identity,
    forward_dense, forward_relu, forward_bias, forward_scale,
    forward_bn, forward_lrelu, forward_identity, forward_reshape,
)

# Forward bounds
from .tf_forward import compute_forward_bounds, Frame

# CNN batched kernel + dispatch (backward) and forward registry handlers;
# backward_maxpool2d / backward_avgpool2d are registry-signature stubs.
from .tf_cnn import (
    dual_conv2d_backward, dual_maxpool2d_backward, dual_avgpool2d_backward,
    backward_conv2d, backward_maxpool2d, backward_avgpool2d,
    forward_conv2d, forward_maxpool2d, forward_avgpool2d,
)

# Smooth activation batched kernels + dispatch (backward) and forward handlers
from .tf_smooth import (
    dual_smooth_backward, dual_sigmoid_backward, dual_tanh_backward,
    compute_smooth_relaxation, sigmoid, dsigmoid, tanh, dtanh,
    backward_sigmoid, backward_tanh,
    forward_sigmoid, forward_tanh,
)

# RNN / Transformer registry-signature stubs (placeholders — not yet implemented)
from .tf_rnn import forward_lstm, backward_lstm, forward_gru, backward_gru
from .tf_transformer import (
    forward_attention, backward_attention,
    forward_layernorm, backward_layernorm,
    forward_gelu, backward_gelu,
)

__all__ = [
    'DualTF', 'backward_add', 'backward_concat', 'forward_add', 'forward_concat',
    'dual_relu_backward', 'dual_dense_backward', 'get_relu_masks',
    'dual_bias_backward', 'dual_scale_backward', 'dual_bn_backward', 'dual_identity_backward',
    'backward_dense', 'backward_relu', 'backward_bias', 'backward_scale',
    'backward_bn', 'backward_identity',
    'forward_dense', 'forward_relu', 'forward_bias', 'forward_scale',
    'forward_bn', 'forward_lrelu', 'forward_identity', 'forward_reshape',
    'compute_forward_bounds',
    'Frame',
    'dual_conv2d_backward', 'dual_maxpool2d_backward', 'dual_avgpool2d_backward',
    'backward_conv2d', 'backward_maxpool2d', 'backward_avgpool2d',
    'forward_conv2d', 'forward_maxpool2d', 'forward_avgpool2d',
    'dual_smooth_backward', 'dual_sigmoid_backward', 'dual_tanh_backward',
    'compute_smooth_relaxation', 'sigmoid', 'dsigmoid', 'tanh', 'dtanh',
    'backward_sigmoid', 'backward_tanh',
    'forward_sigmoid', 'forward_tanh',
    'forward_lstm', 'backward_lstm', 'forward_gru', 'backward_gru',
    'forward_attention', 'backward_attention',
    'forward_layernorm', 'backward_layernorm',
    'forward_gelu', 'backward_gelu',
]
