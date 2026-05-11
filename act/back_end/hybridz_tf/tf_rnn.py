#===- act/back_end/hybridz_tf/tf_rnn.py - HybridZ RNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ RNN Transfer Functions. Delegates to interval_tf with unified
#   hz_tf_* signature for future HZ expansion.
#
#===---------------------------------------------------------------------===#

import torch
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer


# ============================================================================
# Public TF entry points (signature matches HybridzTF._LAYER_REGISTRY)
# ============================================================================

def tf_lstm(L: Layer, Bin: Bounds, tf) -> Fact:
    """Interval-style LSTM bound + constraint generation."""
    cfg = _read_rnn_config(L, op_name="tf_lstm")
    seq_lb, seq_ub = _seq_view(Bin, cfg)
    B, T, H = cfg["batch"], cfg["seq_len"], cfg["hidden_size"]

    h_lb = Bin.lb.new_zeros(B, H)
    h_ub = Bin.lb.new_zeros(B, H)
    c_lb = Bin.lb.new_zeros(B, H)
    c_ub = Bin.lb.new_zeros(B, H)

    out_lb_steps, out_ub_steps = [], []
    for t in range(T):
        x_lb_t, x_ub_t = seq_lb[:, t, :], seq_ub[:, t, :]
        h_lb, h_ub, c_lb, c_ub = _lstm_step(
            x_lb_t, x_ub_t, h_lb, h_ub, c_lb, c_ub, cfg["weights"], H,
        )
        out_lb_steps.append(h_lb)
        out_ub_steps.append(h_ub)

    out_lb, out_ub = _stack_and_flatten(out_lb_steps, out_ub_steps, cfg)
    return _make_fact(L, cfg, out_lb, out_ub, op_name="lstm")


def tf_gru(L: Layer, Bin: Bounds, tf) -> Fact:
    """Interval-style GRU bound + constraint generation."""
    cfg = _read_rnn_config(L, op_name="tf_gru")
    seq_lb, seq_ub = _seq_view(Bin, cfg)
    B, T, H = cfg["batch"], cfg["seq_len"], cfg["hidden_size"]

    h_lb = Bin.lb.new_zeros(B, H)
    h_ub = Bin.lb.new_zeros(B, H)

    out_lb_steps, out_ub_steps = [], []
    for t in range(T):
        x_lb_t, x_ub_t = seq_lb[:, t, :], seq_ub[:, t, :]
        h_lb, h_ub = _gru_step(
            x_lb_t, x_ub_t, h_lb, h_ub, cfg["weights"], H,
        )
        out_lb_steps.append(h_lb)
        out_ub_steps.append(h_ub)

    out_lb, out_ub = _stack_and_flatten(out_lb_steps, out_ub_steps, cfg)
    return _make_fact(L, cfg, out_lb, out_ub, op_name="gru")


def tf_rnn(L: Layer, Bin: Bounds, tf) -> Fact:
    """Interval-style vanilla RNN bound + constraint generation."""
    cfg = _read_rnn_config(L, op_name="tf_rnn")
    nonlin = L.params.get("nonlinearity", "tanh")
    if nonlin not in ("tanh", "relu"):
        raise ValueError(f"hybridz tf_rnn: unsupported nonlinearity {nonlin!r}")

    seq_lb, seq_ub = _seq_view(Bin, cfg)
    B, T, H = cfg["batch"], cfg["seq_len"], cfg["hidden_size"]

    h_lb = Bin.lb.new_zeros(B, H)
    h_ub = Bin.lb.new_zeros(B, H)

    out_lb_steps, out_ub_steps = [], []
    W = cfg["weights"]
    for t in range(T):
        z_lb, z_ub = _affine_block(
            seq_lb[:, t, :], seq_ub[:, t, :], h_lb, h_ub,
            W["W_ih"], W["W_hh"], W["b_ih"], W["b_hh"],
        )
        if nonlin == "tanh":
            h_lb, h_ub = torch.tanh(z_lb), torch.tanh(z_ub)
        else:  # relu — monotone, sound on endpoints
            h_lb, h_ub = z_lb.clamp(min=0), z_ub.clamp(min=0)
        out_lb_steps.append(h_lb)
        out_ub_steps.append(h_ub)

    out_lb, out_ub = _stack_and_flatten(out_lb_steps, out_ub_steps, cfg)
    cfg["nonlinearity"] = nonlin
    return _make_fact(L, cfg, out_lb, out_ub, op_name="rnn")


def tf_embedding(L: Layer, Bin: Bounds, tf) -> Fact:
    """Embedding lookup. Indices are discrete and not under verification, so
    the bound is the per-dim worst case across the embedding table.
    """
    weight: torch.Tensor = L.params["weight"]
    embedding_dim = int(L.params.get("embedding_dim", weight.shape[-1]))
    output_shape = L.params.get("output_shape")

    weight_min = weight.min(dim=0).values
    weight_max = weight.max(dim=0).values

    n_out = len(L.out_vars)
    if n_out % embedding_dim != 0:
        raise ValueError(
            f"hybridz tf_embedding: out_vars count {n_out} not divisible by "
            f"embedding_dim {embedding_dim} (output_shape={output_shape})."
        )
    repeats = n_out // embedding_dim
    out_lb = weight_min.repeat(repeats)
    out_ub = weight_max.repeat(repeats)

    B_out = Bounds(out_lb, out_ub)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"embedding:{L.id}",
        "embedding_dim": embedding_dim,
        "input_shape": L.params.get("input_shape"),
        "output_shape": output_shape,
    }))
    C.add_box(L.id, L.out_vars, B_out)
    return Fact(B_out, C)


# ============================================================================
# Cell-level interval primitives (gate math)
# ============================================================================

def _lstm_step(x_lb, x_ub, h_lb, h_ub, c_lb, c_ub, weights, H):
    pre_lb, pre_ub = _affine_block(
        x_lb, x_ub, h_lb, h_ub,
        weights["W_ih"], weights["W_hh"], weights["b_ih"], weights["b_hh"],
    )
    # Gate splits — sigmoid/tanh are monotone, sound on endpoints.
    i_lb, i_ub = torch.sigmoid(pre_lb[:, 0:H]),     torch.sigmoid(pre_ub[:, 0:H])
    f_lb, f_ub = torch.sigmoid(pre_lb[:, H:2*H]),   torch.sigmoid(pre_ub[:, H:2*H])
    g_lb, g_ub = torch.tanh   (pre_lb[:, 2*H:3*H]), torch.tanh   (pre_ub[:, 2*H:3*H])
    o_lb, o_ub = torch.sigmoid(pre_lb[:, 3*H:4*H]), torch.sigmoid(pre_ub[:, 3*H:4*H])

    fc_lb, fc_ub = _interval_mul(f_lb, f_ub, c_lb, c_ub)
    ig_lb, ig_ub = _interval_mul(i_lb, i_ub, g_lb, g_ub)
    new_c_lb, new_c_ub = fc_lb + ig_lb, fc_ub + ig_ub

    tanh_c_lb, tanh_c_ub = torch.tanh(new_c_lb), torch.tanh(new_c_ub)
    new_h_lb, new_h_ub = _interval_mul(o_lb, o_ub, tanh_c_lb, tanh_c_ub)
    return new_h_lb, new_h_ub, new_c_lb, new_c_ub


def _gru_step(x_lb, x_ub, h_lb, h_ub, weights, H):
    W_ih = weights["W_ih"]   # (3H, in)
    W_hh = weights["W_hh"]   # (3H, H)
    b_ih = weights["b_ih"]   # (3H,) or None
    b_hh = weights["b_hh"]

    # r/z gate pre-activation uses the first 2H rows of W_ih and W_hh.
    rz_pre_lb, rz_pre_ub = _affine_block(
        x_lb, x_ub, h_lb, h_ub,
        W_ih[:2*H], W_hh[:2*H],
        None if b_ih is None else b_ih[:2*H],
        None if b_hh is None else b_hh[:2*H],
    )
    r_lb, r_ub = torch.sigmoid(rz_pre_lb[:, 0:H]),   torch.sigmoid(rz_pre_ub[:, 0:H])
    z_lb, z_ub = torch.sigmoid(rz_pre_lb[:, H:2*H]), torch.sigmoid(rz_pre_ub[:, H:2*H])

    # Candidate n: tanh( W_in @ x + b_in + r ⊙ (W_hn @ h + b_hn) )
    in_lb, in_ub = _affine_only(
        x_lb, x_ub, W_ih[2*H:3*H],
        None if b_ih is None else b_ih[2*H:3*H],
    )
    hn_lb, hn_ub = _affine_only(
        h_lb, h_ub, W_hh[2*H:3*H],
        None if b_hh is None else b_hh[2*H:3*H],
    )
    rhn_lb, rhn_ub = _interval_mul(r_lb, r_ub, hn_lb, hn_ub)
    n_pre_lb, n_pre_ub = in_lb + rhn_lb, in_ub + rhn_ub
    n_lb, n_ub = torch.tanh(n_pre_lb), torch.tanh(n_pre_ub)

    # h_t = (1 - z) ⊙ n + z ⊙ h_{t-1}
    one_minus_z_lb, one_minus_z_ub = 1.0 - z_ub, 1.0 - z_lb
    t1_lb, t1_ub = _interval_mul(one_minus_z_lb, one_minus_z_ub, n_lb, n_ub)
    t2_lb, t2_ub = _interval_mul(z_lb, z_ub, h_lb, h_ub)
    return t1_lb + t2_lb, t1_ub + t2_ub


# ============================================================================
# Interval primitives: sign-split affine + 4-corner product
# ============================================================================

def _affine_only(x_lb, x_ub, W, bias):
    """Interval affine y = W @ x + bias preserving the leading batch dim.
    x shape: (..., in); W shape: (out, in); bias shape: (out,) or None.
    """
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    out_lb = x_lb @ W_pos.T + x_ub @ W_neg.T
    out_ub = x_ub @ W_pos.T + x_lb @ W_neg.T
    if bias is not None:
        out_lb = out_lb + bias
        out_ub = out_ub + bias
    return out_lb, out_ub


def _affine_block(x_lb, x_ub, h_lb, h_ub, W_ih, W_hh, b_ih, b_hh):
    """z = W_ih @ x + W_hh @ h + b_ih + b_hh  (interval form)."""
    ih_lb, ih_ub = _affine_only(x_lb, x_ub, W_ih, b_ih)
    hh_lb, hh_ub = _affine_only(h_lb, h_ub, W_hh, b_hh)
    return ih_lb + hh_lb, ih_ub + hh_ub


def _interval_mul(a_lb, a_ub, b_lb, b_ub):
    """[a_lb,a_ub] ⊙ [b_lb,b_ub] = [min,max] over the four corner products."""
    p1 = a_lb * b_lb
    p2 = a_lb * b_ub
    p3 = a_ub * b_lb
    p4 = a_ub * b_ub
    stacked = torch.stack([p1, p2, p3, p4], dim=0)
    return stacked.min(dim=0).values, stacked.max(dim=0).values


# ============================================================================
# Param parsing / shape plumbing / Fact assembly
# ============================================================================

def _read_rnn_config(L: Layer, *, op_name: str) -> dict:
    """Pull the recurrent layer's config + layer-0 weights out of params,
    rejecting num_layers > 1 (the unrolled cell sweep assumes a single layer)
    and bidirectional=True (no reverse-direction sweep yet).
    """
    p = L.params
    num_layers = int(p.get("num_layers", 1))
    if num_layers != 1:
        raise ValueError(
            f"hybridz {op_name}: only num_layers=1 supported, got {num_layers}."
        )
    if bool(p.get("bidirectional", False)):
        raise NotImplementedError(
            f"hybridz {op_name}: bidirectional=True not supported (reverse "
            f"direction over weight_*_l0_reverse not implemented)."
        )
    input_shape = p["input_shape"]
    output_shape = p["output_shape"]
    batch_first = bool(p.get("batch_first", False))
    if batch_first:
        batch, seq_len, _ = (int(s) for s in input_shape)
    else:
        seq_len, batch, _ = (int(s) for s in input_shape)
    return {
        "input_size":   int(p["input_size"]),
        "hidden_size":  int(p["hidden_size"]),
        "num_layers":   num_layers,
        "bidirectional": bool(p.get("bidirectional", False)),
        "batch_first":  batch_first,
        "input_shape":  input_shape,
        "output_shape": output_shape,
        "batch":        batch,
        "seq_len":      seq_len,
        "weights": {
            "W_ih": p["weight_ih_l0"],
            "W_hh": p["weight_hh_l0"],
            "b_ih": p.get("bias_ih_l0"),
            "b_hh": p.get("bias_hh_l0"),
        },
    }


def _seq_view(Bin: Bounds, cfg: dict):
    """Reshape Bin from flat (numel,) to (B, T, F) regardless of layout."""
    seq_lb = Bin.lb.reshape(cfg["input_shape"])
    seq_ub = Bin.ub.reshape(cfg["input_shape"])
    if not cfg["batch_first"]:
        seq_lb = seq_lb.permute(1, 0, 2).contiguous()
        seq_ub = seq_ub.permute(1, 0, 2).contiguous()
    return seq_lb, seq_ub


def _stack_and_flatten(lb_steps, ub_steps, cfg):
    """Stack per-step (B, H) bounds into the layer's declared output_shape,
    then flatten to a 1D Bounds vector aligned with out_vars.
    """
    out_lb = torch.stack(lb_steps, dim=1)  # (B, T, H)
    out_ub = torch.stack(ub_steps, dim=1)
    if not cfg["batch_first"]:
        out_lb = out_lb.permute(1, 0, 2).contiguous()
        out_ub = out_ub.permute(1, 0, 2).contiguous()
    return out_lb.reshape(-1), out_ub.reshape(-1)


def _make_fact(L: Layer, cfg: dict, out_lb, out_ub, op_name: str) -> Fact:
    if out_lb.numel() != len(L.out_vars):
        raise RuntimeError(
            f"hybridz tf_{op_name}: produced {out_lb.numel()} outputs, expected "
            f"len(out_vars)={len(L.out_vars)} (output_shape={cfg['output_shape']})."
        )
    B_out = Bounds(out_lb, out_ub)
    meta = {
        "tag": f"{op_name}:{L.id}",
        "input_size": cfg["input_size"],
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "bidirectional": cfg["bidirectional"],
        "batch_first": cfg["batch_first"],
        "input_shape": cfg["input_shape"],
        "output_shape": cfg["output_shape"],
    }
    if "nonlinearity" in cfg:
        meta["nonlinearity"] = cfg["nonlinearity"]
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), meta))
    C.add_box(L.id, L.out_vars, B_out)
    return Fact(B_out, C)
