#===- act/back_end/interval_tf/tf_transformer.py - Transformer Interval TF ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transformer Interval Transfer Functions. Provides interval-based transfer
#   functions for transformer components including attention mechanisms.
#
#===---------------------------------------------------------------------===#

import torch
from typing import List
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import pwl_meta, scale_interval
from act.back_end.interval_tf.tf_mlp import tf_concat

# tf_embedding is provided by act.back_end.interval_tf.tf_rnn (signature
# (L, Bin) -> Fact). The previous transformer-local definition had a wrong
# signature and would shadow the rnn one via `from tf_transformer import *`
# in interval_tf.py — both EMBEDDING and EMBEDDING_TF would have raised
# TypeError at runtime. Single source of truth lives in tf_rnn.

def tf_posenc(L: Layer, Bin: Bounds) -> Fact:
    P=L.params["pos_vec"]; B=Bounds(Bin.lb+P, Bin.ub+P); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"posenc:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_layernorm(L: Layer, Bin: Bounds) -> Fact:
    if Bin.lb.dim() < 2:
        raise ValueError(f"LAYERNORM expects batched bounds [B, *], got shape {tuple(Bin.lb.shape)}")
    norm_dims = tuple(range(1, Bin.lb.dim()))
    mu_lb = torch.mean(Bin.lb, dim=norm_dims, keepdim=True)
    mu_ub = torch.mean(Bin.ub, dim=norm_dims, keepdim=True)
    cx_lb, cx_ub = Bin.lb - mu_ub, Bin.ub - mu_lb
    radius = 0.5 * (Bin.ub - Bin.lb)
    v_lo = torch.zeros_like(mu_lb)
    v_hi = torch.mean((2 * radius) ** 2, dim=norm_dims, keepdim=True)
    eps=float(L.params.get("eps",1e-5))
    eps_t = Bin.lb.new_tensor(eps)
    inv_lb = torch.rsqrt(v_hi + eps_t)
    inv_ub = torch.rsqrt(torch.clamp_min(v_lo, 0.0) + eps_t)
    sh_lb, sh_ub = scale_interval(cx_lb, cx_ub, inv_lb, inv_ub)
    gamma,beta=L.params["gamma"],L.params["beta"]
    lb=torch.where(gamma>=0, gamma*sh_lb+beta, gamma*sh_ub+beta)
    ub=torch.where(gamma>=0, gamma*sh_ub+beta, gamma*sh_lb+beta)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"layernorm:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_gelu(L: Layer, Bin: Bounds) -> Fact:
    GELU_MIN_X = -0.7517916
    GELU_MIN_Y = -0.17004
    f = lambda x: 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*(x**3))))
    f_lb, f_ub = f(Bin.lb), f(Bin.ub)
    contains_min = (Bin.lb <= GELU_MIN_X) & (Bin.ub >= GELU_MIN_X)
    lb = torch.where(contains_min, torch.full_like(f_lb, GELU_MIN_Y), torch.minimum(f_lb, f_ub))
    ub = torch.maximum(f_lb, f_ub)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"gelu:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,3)}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_att_scores(L: Layer, Bq: Bounds, Bk: Bounds) -> Fact:
    batch_size = Bq.lb.shape[0]
    if Bk.lb.shape[0] != batch_size:
        raise ValueError(f"ATT_SCORES expects matching batch dims, got {batch_size} and {Bk.lb.shape[0]}")
    s=Bq.lb.new_tensor(1.0/float(L.params["dk"]))
    lo=torch.minimum(torch.minimum(Bq.lb*Bk.lb, Bq.lb*Bk.ub), torch.minimum(Bq.ub*Bk.lb, Bq.ub*Bk.ub))
    hi=torch.maximum(torch.maximum(Bq.lb*Bk.lb, Bq.lb*Bk.ub), torch.maximum(Bq.ub*Bk.lb, Bq.ub*Bk.ub))
    lb=s*lo.sum(dim=-1); ub=s*hi.sum(dim=-1)
    if L.params.get("mask") is not None: lb=lb+L.params["mask"]; ub=ub+L.params["mask"]
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["q_vars"] + L.params["k_vars"]), {"tag":f"att_scores:{L.id}","scale":float(s),"mcc":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_softmax(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.zeros_like(Bin.lb), torch.ones_like(Bin.ub))
    rowsize=int(Bin.lb.shape[-1]); mode=L.params.get("mode","simplex"); tag=f"softmax:{mode}:{L.id}"
    C=ConSet()
    if mode=="simplex": C.replace(Con("INEQ", tuple(L.out_vars), {"tag":tag,"rowsize":rowsize}))
    elif mode=="pwl":  C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"segs":{"K":3}}))
    else:              C.replace(Con("BIN",  tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"K":4,"sos2":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_att_mix(L: Layer, Bw: Bounds, Bv: Bounds) -> Fact:
    batch_size = Bw.lb.shape[0]
    if Bv.lb.shape[0] != batch_size:
        raise ValueError(f"ATT_MIX expects matching batch dims, got {batch_size} and {Bv.lb.shape[0]}")
    lo=torch.minimum(torch.minimum(Bw.lb*Bv.lb, Bw.lb*Bv.ub), torch.minimum(Bw.ub*Bv.lb, Bw.ub*Bv.ub)).sum(dim=-1)
    hi=torch.maximum(torch.maximum(Bw.lb*Bv.lb, Bw.lb*Bv.ub), torch.maximum(Bw.ub*Bv.lb, Bw.ub*Bv.ub)).sum(dim=-1)
    B=Bounds(lo,hi); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["w_vars"] + L.params["v_vars"]), {"tag":f"att_mix:{L.id}","mcc":True,"rowsize":L.params["rowsize"]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_mha_split(L: Layer, Bin: Bounds) -> Fact: return Fact(Bin.copy(), ConSet())
def tf_mha_join(L: Layer, Bs: List[Bounds]) -> Fact: return tf_concat(L, Bs)
def tf_mask_add(L: Layer, Bin: Bounds) -> Fact:
    M=L.params["M"]; B=Bounds(Bin.lb+M, Bin.ub+M); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"mask:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)
