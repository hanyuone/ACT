#===- act/back_end/analyze.py - Network Analysis Functions --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Network analysis functions for ACT verification framework.
#   Provides analysis capabilities for neural network structures and properties.
#
#===---------------------------------------------------------------------===#

import torch
from collections import deque
from typing import Dict, Tuple
from act.back_end.core import Bounds, Fact, Net, ConSet
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_functions import dispatch_tf, set_transfer_function_mode

# Initialize default transfer function mode
def initialize_tf_mode(mode: str = "interval"):
    """Initialize transfer function mode. Call this before using analyze()."""
    set_transfer_function_mode(mode)

@torch.no_grad()
def analyze(net: Net, entry_id: int, entry_fact: Fact, eps: float=1e-9) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    """
    Perform abstract interpretation on the network starting from entry_fact.
    Args:
        net: ACT network structure
        entry_id: ID of the entry (INPUT) layer
        entry_fact: Initial Fact containing bounds and constraints for the input
        eps: Convergence epsilon for fixpoint iteration
    
    Returns:
        Tuple of (before, after, globalC) containing propagated facts and global constraints
    """
    # Auto-initialize transfer function mode if not set
    try:
        from act.back_end.transfer_functions import get_transfer_function
        get_transfer_function()  # Check if already initialized
    except RuntimeError:
        initialize_tf_mode("interval")  # Default to interval mode
        
    before: Dict[int, Fact] = {}
    after:  Dict[int, Fact] = {}
    globalC = ConSet()

    # init with +/- inf boxes (vector length per layer's out_vars)
    for L in net.layers:
        n = len(L.out_vars)
        hi = torch.full((n,), float("inf"), device=entry_fact.bounds.lb.device, dtype=entry_fact.bounds.lb.dtype)
        lo = torch.full((n,), -float("inf"), device=entry_fact.bounds.lb.device, dtype=entry_fact.bounds.lb.dtype)
        before[L.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        after[L.id]  = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        L.cache.clear()

    # Seed entry with provided Fact (includes all input constraints)
    before[entry_id] = entry_fact

    # Seed every other zero-indegree source (e.g. CONSTANT layers emitted by
    # torch2act for ONNX initializers). Without this, source-layer bounds stay
    # at +/-inf forever because the worklist starts at entry_id and CONSTANTs
    # have no predecessor that would ever push them on. (Oracle finding #5.)
    seeds = [entry_id]
    for L in net.layers:
        if L.id == entry_id or net.preds.get(L.id):
            continue
        if L.kind == LayerKind.CONSTANT.value:
            val = L.params["value"].flatten().to(
                device=entry_fact.bounds.lb.device,
                dtype=entry_fact.bounds.lb.dtype,
            )
            before[L.id] = Fact(bounds=Bounds(val.clone(), val.clone()), cons=ConSet())
        # Other zero-indegree kinds (none today) would be seeded similarly.
        seeds.append(L.id)

    WL = deque(seeds)
    while WL:
        lid = WL.popleft(); L = net.by_id[lid]

        # merge predecessors into before[lid]
        if net.preds.get(lid):
            preds_list = net.preds[lid]
            # Initialize from first predecessor (not infinite bounds)
            first_bounds = after[preds_list[0]].bounds
            Bjoin = Bounds(lb=first_bounds.lb.clone(), ub=first_bounds.ub.clone())
            Cjoin = ConSet()
            for con in after[preds_list[0]].cons: Cjoin.replace(con)
            # Join with remaining predecessors when shapes match (DAG merge points).
            # Multi-input ops with heterogeneous predecessor shapes (MATMUL, CONCAT,
            # SCATTER_ND, etc.) ignore Bin and pull each predecessor explicitly via
            # get_predecessor_bounds; the join is meaningless for them so we skip
            # rather than crash.
            for pid in preds_list[1:]:
                pb = after[pid].bounds
                if pb.lb.shape == Bjoin.lb.shape:
                    Bjoin = box_join(Bjoin, pb)
                for con in after[pid].cons: Cjoin.replace(con)
            before[lid] = Fact(Bjoin, Cjoin)

        out_fact = dispatch_tf(L, before, after, net)

        if changed_or_maskdiff(L, out_fact.bounds, None, eps):
            after[lid] = out_fact
            update_cache(L, out_fact.bounds, None)
            for con in out_fact.cons: globalC.replace(con)
            for sid in net.succs.get(lid, []): WL.append(sid)

    return before, after, globalC
