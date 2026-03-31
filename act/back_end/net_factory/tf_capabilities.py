#===- act/back_end/net_factory/tf_capabilities.py - TF Registry Query --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Collects supported layer types from each Transfer Function implementation
#   (IntervalTF, HybridzTF, DualTF) and provides utilities to compute layer
#   intersections/unions for TF-aware network generation.
#
#===---------------------------------------------------------------------===#

from typing import Dict, FrozenSet, List, Optional
import functools
import logging

logger = logging.getLogger(__name__)

# Layers that may not appear in every TF registry but should be considered
# supported (identity at inference, or structural-only).
_IMPLICIT_LAYERS = frozenset({
    "OUTPUT_SPEC",
    "DROPOUT",
    "IDENTITY",
})

# ============================================================================
# Core Functions
# ============================================================================

@functools.lru_cache(maxsize=1)
def get_tf_capabilities() -> Dict[str, FrozenSet[str]]:
    """
    Collect supported layers from each TF's registry dynamically.

    Reads directly from:
    - IntervalTF._LAYER_REGISTRY
    - HybridzTF._LAYER_REGISTRY
    - DualTF._BACKWARD_REGISTRY

    Returns:
        Dict mapping TF name to frozenset of supported layer kinds (uppercase).

    Note:
        Results are cached. Call ``get_tf_capabilities.cache_clear()`` to refresh.
    """
    result = {}

    # ---- Interval TF ----
    try:
        from act.back_end.interval_tf import IntervalTF
        registry = getattr(IntervalTF, '_LAYER_REGISTRY', {})
        layers = set(k.upper() for k in registry.keys())
        layers.update(_IMPLICIT_LAYERS)
        result["interval"] = frozenset(layers)
        logger.debug("IntervalTF: loaded %d layers from registry", len(layers))
    except (ImportError, AttributeError) as e:
        logger.error("Failed to load IntervalTF registry: %s", e)
        raise RuntimeError(f"Cannot load IntervalTF._LAYER_REGISTRY: {e}") from e

    # ---- HybridZ TF ----
    try:
        from act.back_end.hybridz_tf import HybridzTF
        registry = getattr(HybridzTF, '_LAYER_REGISTRY', {})
        layers = set(k.upper() for k in registry.keys())
        layers.update(_IMPLICIT_LAYERS)
        result["hybridz"] = frozenset(layers)
        logger.debug("HybridzTF: loaded %d layers from registry", len(layers))
    except (ImportError, AttributeError) as e:
        logger.error("Failed to load HybridzTF registry: %s", e)
        raise RuntimeError(f"Cannot load HybridzTF._LAYER_REGISTRY: {e}") from e

    # ---- Dual TF ----
    try:
        from act.back_end.dual_tf import DualTF
        registry = getattr(DualTF, '_BACKWARD_REGISTRY', {})
        layers = set(k.upper() for k in registry.keys())
        layers.update(_IMPLICIT_LAYERS)
        result["dual"] = frozenset(layers)
        logger.debug("DualTF: loaded %d layers from registry", len(layers))
    except (ImportError, AttributeError) as e:
        logger.error("Failed to load DualTF registry: %s", e)
        raise RuntimeError(f"Cannot load DualTF._BACKWARD_REGISTRY: {e}") from e

    return result


def get_allowed_layers(
    tf_targets: Optional[List[str]] = None,
    mode: str = "intersection",
) -> FrozenSet[str]:
    """
    Compute allowed layer set based on target TFs and combination mode.

    Args:
        tf_targets: Target TF list.
            - None  -> all TFs ``["interval", "hybridz", "dual"]``
            - ``["interval"]`` -> only IntervalTF layers
            - ``["interval", "hybridz"]`` -> intersection or union of both
        mode: ``"intersection"`` (layers supported by ALL targets, default)
              or ``"union"`` (layers supported by ANY target).

    Returns:
        FrozenSet of allowed layer kinds (uppercase).

    Raises:
        ValueError: Unknown TF name, unknown mode, or empty result.
    """
    if tf_targets is None:
        tf_targets = ["interval", "hybridz", "dual"]

    tf_targets = [t.lower().strip() for t in tf_targets]
    capabilities = get_tf_capabilities()

    unknown = set(tf_targets) - set(capabilities.keys())
    if unknown:
        raise ValueError(
            f"Unknown TF targets: {unknown}. Available: {list(capabilities.keys())}"
        )

    target_sets = [capabilities[tf] for tf in tf_targets]

    if len(target_sets) == 1:
        result = target_sets[0]
    elif mode == "intersection":
        result = target_sets[0]
        for s in target_sets[1:]:
            result = result & s
    elif mode == "union":
        result = frozenset().union(*target_sets)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected 'intersection' or 'union'.")

    if not result:
        raise ValueError(
            f"Empty layer set for tf_targets={tf_targets}, mode={mode}. "
            f"Check TF registries or try mode='union'."
        )

    logger.info(
        "Allowed layers: %d for tf_targets=%s, mode=%s",
        len(result), tf_targets, mode,
    )
    return result


# ============================================================================
# Utility / Convenience Functions
# ============================================================================

def get_tf_specific_layers(tf_name: str) -> FrozenSet[str]:
    """Get layers unique to *tf_name* (not supported by any other TF)."""
    caps = get_tf_capabilities()
    tf_name = tf_name.lower()
    if tf_name not in caps:
        raise ValueError(f"Unknown TF: {tf_name}")
    other = set()
    for name, layers in caps.items():
        if name != tf_name:
            other.update(layers)
    return caps[tf_name] - other


def get_common_layers() -> FrozenSet[str]:
    """Layers supported by *all* TFs (intersection)."""
    return get_allowed_layers(None, "intersection")


def get_all_known_layers() -> FrozenSet[str]:
    """Layers supported by *any* TF (union)."""
    return get_allowed_layers(None, "union")


def for_interval() -> FrozenSet[str]:
    return get_allowed_layers(["interval"])

def for_hybridz() -> FrozenSet[str]:
    return get_allowed_layers(["hybridz"])

def for_dual() -> FrozenSet[str]:
    return get_allowed_layers(["dual"])

def for_all_tfs() -> FrozenSet[str]:
    return get_allowed_layers(["interval", "hybridz", "dual"], "intersection")


# ============================================================================
# Diagnostics
# ============================================================================

def print_capabilities_report() -> None:
    """Print a human-readable TF layer support report to stdout."""
    caps = get_tf_capabilities()

    CATEGORIES = {
        "Basic":      ["INPUT", "INPUT_SPEC", "OUTPUT_SPEC", "ASSERT"],
        "Dense":      ["DENSE", "BIAS", "SCALE", "BN"],
        "Activation": ["RELU", "LRELU", "TANH", "SIGMOID", "RELU6", "HARDTANH",
                       "HARDSIGMOID", "HARDSWISH", "SILU", "SOFTPLUS", "MISH",
                       "SOFTSIGN", "ABS", "CLIP", "SQUARE", "POWER"],
        "Conv":       ["CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D"],
        "Pool":       ["MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D",
                       "AVGPOOL1D", "AVGPOOL2D", "AVGPOOL3D"],
        "MultiInput": ["ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN",
                       "MATMUL", "CONCAT"],
        "Reshape":    ["FLATTEN", "RESHAPE", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE",
                       "DROPOUT", "IDENTITY"],
        "Other":      ["PAD", "UPSAMPLE", "SLICE", "GATHER", "TILE", "EXPAND",
                       "INDEX_SELECT", "LSTM", "GRU", "RNN", "EMBEDDING"],
    }

    print("=" * 70)
    print("TF Layer Capabilities Report")
    print("=" * 70)

    for tf_name in sorted(caps):
        layers = caps[tf_name]
        print(f"\n{tf_name.upper()} ({len(layers)} layers):")
        for cat, cat_layers in CATEGORIES.items():
            supported = [l for l in cat_layers if l in layers]
            if supported:
                print(f"    {cat}: {', '.join(supported)}")
        all_categorized = {l for cat_layers in CATEGORIES.values() for l in cat_layers}
        uncategorized = layers - all_categorized
        if uncategorized:
            print(f"    Other: {', '.join(sorted(uncategorized))}")

    intersection = get_allowed_layers(list(caps), "intersection")
    union_set = get_allowed_layers(list(caps), "union")

    print(f"\n{'=' * 70}")
    print(f"\nINTERSECTION ({len(intersection)}): {sorted(intersection)}")
    print(f"\nUNION ({len(union_set)}): {sorted(union_set)}")

    print(f"\n{'=' * 70}")
    print("TF-SPECIFIC LAYERS:")
    for tf_name in sorted(caps):
        specific = get_tf_specific_layers(tf_name)
        tag = f"{tf_name.upper()} only"
        print(f"    {tag}: {sorted(specific) if specific else '(none)'}")

    print(f"\n{'=' * 70}")
    print("SUPPORT MATRIX:")
    print(f"    {'Layer':<22} {'interval':>10} {'hybridz':>10} {'dual':>10}")
    print(f"    {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for layer in sorted(union_set):
        row = f"    {layer:<22}"
        for tf in ("interval", "hybridz", "dual"):
            row += f" {'Y':>10}" if layer in caps.get(tf, set()) else f" {'-':>10}"
        print(row)
