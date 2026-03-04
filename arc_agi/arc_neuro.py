# Copyright (c) 2026 Brad Wallace. All rights reserved.
# Subject to Sovereign Integrity Protocol License (SIP License v1.1).
# See SIP_LICENSE.md for full terms.
from __future__ import annotations
import ctypes, os, threading, json
from pathlib import Path
from typing import Optional, List, Tuple
from arc_types import Grid
from arc_bra import _PRIM_COORDS, eigen_charge

# --- Rust Kernel Bindings ---

class SynapticRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("trigger_hash",  ctypes.c_uint64),
        ("trigger_trace", ctypes.c_int64),
        ("trigger_det",   ctypes.c_int64),
        ("output_color",  ctypes.c_uint8),
        ("confidence",    ctypes.c_uint8),
        ("_pad",          ctypes.c_uint8 * 6),
    ]
assert ctypes.sizeof(SynapticRule) == 32

_LIB_PATHS = [
    "/Volumes/Seagate 4tb/unified_field_theory/arc_agi/libarc_heuristics.dylib",
    "/Volumes/Seagate 4tb/unified_field_theory/arc_agi/libarc_heuristics.so",
    "libarc_heuristics.dylib",
    "libarc_heuristics.so",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libarc_heuristics.dylib"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libarc_heuristics.so"),
]

def _load():
    for p in _LIB_PATHS:
        if Path(p).exists():
            try:
                lib = ctypes.CDLL(str(Path(p).absolute()))
                _bind(lib)
                return lib
            except Exception as e:
                print(f"Failed to load {p}: {e}")
    raise FileNotFoundError("libarc_heuristics shared library not found")

def _bind(lib):
    u8p=ctypes.POINTER(ctypes.c_uint8)
    u64p=ctypes.POINTER(ctypes.c_uint64)
    i64p=ctypes.POINTER(ctypes.c_int64)
    RP=ctypes.POINTER(SynapticRule)
    sz=ctypes.c_size_t
    for name,rt,at in [
        ("arc_neuro_verify",           ctypes.c_int32, []),
        ("arc_neuro_extract_fields",   sz, [u8p,sz,sz,u64p,i64p,i64p]),
        ("arc_neuro_context_fields",   sz, [u8p,sz,sz,u64p,i64p,i64p]),
        ("arc_neuro_learn_rules",      sz, [u8p,u8p,sz,sz,ctypes.c_int32,RP,sz]),
        ("arc_neuro_merge_rules",      sz, [RP,sz,RP,sz,RP,sz]),
        ("arc_neuro_apply_rules",      sz, [u8p,sz,sz,RP,sz,u8p]),
        ("arc_neuro_apply_rules_iter", sz, [u8p,sz,sz,RP,sz,u8p,sz]),
        ("arc_neuro_components",       sz, [u8p,sz,sz,u8p]),
        ("arc_neuro_comp_charges",     sz, [u8p,u8p,sz,sz,sz,u64p,i64p,i64p]),
        ("arc_neuro_delta_charge",     ctypes.c_int32, [u8p,u8p,sz,sz,u64p,i64p,i64p]),
        ("arc_neuro_spatial_graph",    ctypes.c_int32, [u8p,sz,sz,sz,u64p,i64p,i64p]),
        ("arc_neuro_pattern_period",   ctypes.c_int32, [u8p,sz,sz,u8p,u8p]),
    ]:
        getattr(lib,name).restype=rt
        getattr(lib,name).argtypes=at

_lib = None
def get_lib():
    global _lib
    if _lib is None: _lib = _load()
    return _lib

def _to_c(grid):
    if not grid: return (ctypes.c_uint8 * 0)(), 0, 0
    h,w = len(grid), len(grid[0])
    flat = (ctypes.c_uint8*(w*h))()
    for r in range(h):
        for c in range(w): flat[r*w+c] = grid[r][c]&0xFF
    return flat, w, h

def _from_c(flat,w,h):
    return [[flat[r*w+c] for c in range(w)] for r in range(h)]

def verify(): return get_lib().arc_neuro_verify()==1

def learn_rules(in_grid,out_grid,wide=False):
    if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]): return []
    lib=get_lib(); ig,w,h=_to_c(in_grid); og,_,_=_to_c(out_grid)
    cap=w*h+1; buf=(SynapticRule*cap)()
    n=lib.arc_neuro_learn_rules(ig,og,w,h,int(wide),buf,cap)
    return list(buf[:n])

def merge_rules(ra,rb):
    if not ra or not rb: return []
    lib=get_lib()
    aa=(SynapticRule*len(ra))(*ra); bb=(SynapticRule*len(rb))(*rb)
    cap=min(len(ra),len(rb))+1; buf=(SynapticRule*cap)()
    n=lib.arc_neuro_merge_rules(aa,len(ra),bb,len(rb),buf,cap)
    return list(buf[:n])

def learn_from_pairs(pairs,wide=False):
    if not pairs: return []
    rules=learn_rules(pairs[0][0],pairs[0][1],wide=wide)
    for inp,out in pairs[1:]:
        if not rules: break
        rules=merge_rules(rules,learn_rules(inp,out,wide=wide))
    return rules

def apply_rules(grid,rules,iterative=False,max_iter=5):
    if not rules: return [row[:] for row in grid]
    lib=get_lib(); ig,w,h=_to_c(grid); arr=(SynapticRule*len(rules))(*rules)
    out=(ctypes.c_uint8*(w*h))()
    if iterative: lib.arc_neuro_apply_rules_iter(ig,w,h,arr,len(rules),out,max_iter)
    else: lib.arc_neuro_apply_rules(ig,w,h,arr,len(rules),out)
    return _from_c(out,w,h)

def components(grid):
    lib=get_lib(); g,w,h=_to_c(grid); ci=(ctypes.c_uint8*(w*h))()
    nc=lib.arc_neuro_components(g,w,h,ci); return list(ci[:w*h]),nc

def component_charges(grid):
    lib=get_lib(); g,w,h=_to_c(grid); ci_flat,nc=components(grid)
    if nc==0: return []
    ci=(ctypes.c_uint8*(w*h))(*ci_flat)
    hh=(ctypes.c_uint64*nc)(); tt=(ctypes.c_int64*nc)(); dd=(ctypes.c_int64*nc)()
    lib.arc_neuro_comp_charges(g,ci,w,h,nc,hh,tt,dd)
    return [(hh[i],tt[i],dd[i]) for i in range(nc)]

def delta_charge(in_grid,out_grid):
    if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]): return (0,0,0)
    lib=get_lib(); ig,w,h=_to_c(in_grid); og,_,_=_to_c(out_grid)
    h_=ctypes.c_uint64(0); t_=ctypes.c_int64(0); d_=ctypes.c_int64(0)
    lib.arc_neuro_delta_charge(ig,og,w,h,ctypes.byref(h_),ctypes.byref(t_),ctypes.byref(d_))
    return (h_.value,t_.value,d_.value)

def extract_fields(grid,wide=False):
    lib=get_lib(); g,w,h=_to_c(grid); n=w*h
    hh=(ctypes.c_uint64*n)(); tt=(ctypes.c_int64*n)(); dd=(ctypes.c_int64*n)()
    fn=lib.arc_neuro_context_fields if wide else lib.arc_neuro_extract_fields
    fn(g,w,h,hh,tt,dd); return [(hh[i],tt[i],dd[i]) for i in range(n)]

def pattern_period(grid):
    lib=get_lib(); g,w,h=_to_c(grid)
    hp=ctypes.c_uint8(0); vp=ctypes.c_uint8(0)
    lib.arc_neuro_pattern_period(g,w,h,ctypes.byref(hp),ctypes.byref(vp))
    return (hp.value,vp.value)

def spatial_graph_charge(grid):
    lib=get_lib(); g,w,h=_to_c(grid); ci_flat,nc=components(grid)
    if nc==0: return (0,0,0)
    ci=(ctypes.c_uint8*(w*h))(*ci_flat)
    h_=ctypes.c_uint64(0); t_=ctypes.c_int64(0); d_=ctypes.c_int64(0)
    lib.arc_neuro_spatial_graph(ci,w,h,nc,ctypes.byref(h_),ctypes.byref(t_),ctypes.byref(d_))
    return (h_.value,t_.value,d_.value)

def learn_from_pairs_union(pairs, wide=False):
    counts = {}
    for inp,out in pairs:
        for rule in learn_rules(inp, out, wide=wide):
            key = (rule.trigger_hash, rule.trigger_trace, rule.trigger_det)
            if key not in counts:
                counts[key] = rule.output_color
            elif counts[key] != rule.output_color:
                counts[key] = None
    result = []
    for (h,tr,dt),color in counts.items():
        if color is None: continue
        r = SynapticRule()
        r.trigger_hash=h; r.trigger_trace=tr; r.trigger_det=dt
        r.output_color=color; r.confidence=1
        result.append(r)
    return result

def solve_by_local_rules_v2(task, wide=False):
    pairs=[(p.input,p.output) for p in task.train if p.output is not None]
    if not pairs: return None
    rules=learn_from_pairs_union(pairs, wide=wide)
    if not rules: return None
    for inp,out in pairs:
        if apply_rules(inp,rules)!=out:
            if apply_rules(inp,rules,iterative=True)!=out: return None
    return apply_rules(task.test[0].input,rules) if task.test else None

def _is_tiling_of(small, large):
    if not small or not large: return False
    sh,sw=len(small),len(small[0])
    lh,lw=len(large),len(large[0])
    if lh<sh or lw<sw: return False
    for r in range(lh):
        for c in range(lw):
            if large[r][c]!=small[r%sh][c%sw]: return False
    return True

def solve_by_period_v2(task):
    pairs=[(p.input,p.output) for p in task.train if p.output is not None]
    if not pairs: return None
    test_inp=task.test[0].input if task.test else None
    if test_inp is None: return None
    for inp,out in pairs:
        if not _is_tiling_of(inp,out): return None
    out0=pairs[0][1]
    oh,ow=len(out0),len(out0[0])
    ih,iw=len(pairs[0][0]),len(pairs[0][0][0])
    th,tw=len(test_inp),len(test_inp[0])
    if th==0 or tw==0 or ih==0 or iw==0: return None
    # Calculate output shape based on scaling
    oh2 = th * oh // ih
    ow2 = tw * ow // iw
    return [[test_inp[r%th][c%tw] for c in range(ow2)] for r in range(oh2)]

def solve_by_delta(task):
    pairs=[(p.input,p.output) for p in task.train if p.output is not None]
    if len(pairs)<2: return None
    h0,w0=len(pairs[0][0]),len(pairs[0][0][0])
    if not all(len(p[0])==h0 and len(p[0][0])==w0 for p in pairs): return None
    charges=[delta_charge(i,o) for i,o in pairs]
    if len(set(charges))!=1: return None
    inp0,out0=pairs[0]
    test_inp=task.test[0].input if task.test else None
    if test_inp is None or len(test_inp)!=h0 or len(test_inp[0])!=w0: return None
    result=[row[:] for row in test_inp]
    for r in range(h0):
        for c in range(w0):
            if inp0[r][c]!=out0[r][c]: result[r][c]=out0[r][c]
    return result

def neuro_solve_v2(task):
    r={"prediction":None,"tier":None,"n_rules":0,"confidence":0}
    for wide,tier in [(False,"local_3x3"),(True,"local_5x5")]:
        try:
            pred=solve_by_local_rules_v2(task,wide=wide)
            if pred is not None:
                pairs=[(p.input,p.output) for p in task.train if p.output]
                rules=learn_from_pairs_union(pairs,wide=wide)
                r.update({"prediction":pred,"tier":tier,"n_rules":len(rules),"confidence":2})
                return r
        except Exception: pass
    for fn,tier in [(solve_by_delta,"delta"),(solve_by_period_v2,"period")]:
        try:
            pred=fn(task)
            if pred is not None:
                r.update({"prediction":pred,"tier":tier,"n_rules":1,"confidence":1})
                return r
        except Exception: pass
    return r

# --- Legacy NeuromorphicBrain (Spike Routing) ---

SYNAPSE_FILE = "/Volumes/Seagate 4tb/unified_field_theory/arc_agi/synapses.json"
_persist_lock = threading.Lock()

class NeuromorphicBrain:
    def __init__(self, spike_threshold: float = 0.6):
        self.base_threshold = spike_threshold
        self.spike_threshold = spike_threshold
        # Each primitive has a resting membrane potential of 0.0
        self.membrane_potentials = {node: 0.0 for node in _PRIM_COORDS.keys()}
        # Synaptic weights for specific task-feature pairings
        self.synapses = self._load_synapses()

    def _load_synapses(self) -> dict:
        if os.path.exists(SYNAPSE_FILE):
            with _persist_lock:
                try:
                    with open(SYNAPSE_FILE, 'r') as f:
                        return json.load(f)
                except:
                    return {}
        return {}

    def _save_synapses(self):
        with _persist_lock:
            try:
                with open(SYNAPSE_FILE, 'w') as f:
                    json.dump(self.synapses, f, indent=2)
            except Exception as e:
                print(f"Error saving synapses: {e}")

    def _apply_stimulus(self, axis_weights: tuple[float, float, float], strength: float = 1.0, task_charge=None):
        sx, sy, sz = axis_weights
        task_key = str(task_charge.hash) if task_charge else None
        memory_boosts = self.synapses.get(task_key, {}) if task_key else {}

        for prim, coords in _PRIM_COORDS.items():
            cx, cy, cz = coords
            dist_sq = (cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2
            if dist_sq == 0:
                potential = strength * 1.5
            else:
                potential = strength / dist_sq
            
            boost = memory_boosts.get(prim, 0.0)
            self.membrane_potentials[prim] += potential + boost

    def learn_success(self, task_grid: Grid, effective_primitives: list[str]):
        data = json.dumps(task_grid).encode('utf-8')
        charge = eigen_charge(data)
        task_key = str(charge.hash)
        
        if task_key not in self.synapses:
            self.synapses[task_key] = {}
            
        for prim in effective_primitives:
            current = self.synapses[task_key].get(prim, 0.0)
            self.synapses[task_key][prim] = current + 0.5
            
        self._save_synapses()

    def neuromodulate(self, candidate_count: int):
        if candidate_count > 10:
            self.spike_threshold += 0.05
        elif candidate_count == 0:
            self.spike_threshold = max(0.1, self.spike_threshold - 0.05)
        else:
            delta = (self.base_threshold - self.spike_threshold) * 0.1
            self.spike_threshold += delta

    def route_ast(self, input_grid: Grid, output_grid: Grid, com_metrics: dict, 
                  train_pairs: list = None, memory_boost: list[str] = None,
                  pedagogical_boost: list[str] = None) -> list[str]:
        # Feature Extraction
        data = json.dumps(input_grid).encode('utf-8')
        charge = eigen_charge(data)
        
        features = set()
        if com_metrics:
            dx, dy, dz = com_metrics.get("target_coords", (0, 0, 0))
            if dx > 15: features.add("geometric")
            if dy > 15: features.add("ontological")
            if dz > 15: features.add("spectral")
        
        if train_pairs:
            try:
                from arc_hard_heuristics import HardHeuristicsEngine as HHE
                if HHE.detect_recursive_tiling(train_pairs):
                    features.add("recursive"); features.add("tiling")
                import numpy as np
                if len(HHE.apply_nca_logic(np.array(input_grid))) > 0:
                    features.add("nca"); features.add("cellular")
            except: pass

        # Tier Escalation
        tier1_conjuncts = [{"nca", "cellular"}, {"local", "iterative"}]
        tier2_conjuncts = [{"recursive", "tiling"}, {"relational"}]
        tier3_conjuncts = [{"geometric"}, {"ontological"}]

        activated_experts = []
        for group in tier1_conjuncts:
            if group.issubset(features):
                activated_experts.extend(["kernel_local_3x3", "kernel_local_5x5", "nca_flood", "nca_outline"])
                break
        for group in tier2_conjuncts:
            if group.issubset(features):
                activated_experts.extend(["kernel_period", "kernel_delta", "relational_graph"])
                break
        for group in tier3_conjuncts:
            if group.issubset(features):
                activated_experts.extend(["rotate", "flip", "recolor", "outline", "kernel_local_3x3"])
                break

        # Resonate and Spike
        self.membrane_potentials = {node: 0.0 for node in _PRIM_COORDS.keys()}
        if com_metrics:
            self._apply_stimulus(com_metrics.get("target_coords", (0, 0, 0)), strength=5.0, task_charge=charge)

        for exp in activated_experts:
            if exp in self.membrane_potentials:
                self.membrane_potentials[exp] += 1.0

        if memory_boost:
            for exp in memory_boost:
                if exp in self.membrane_potentials: self.membrane_potentials[exp] += 2.0
                else: self.membrane_potentials[exp] = 2.0

        if pedagogical_boost:
            for exp in pedagogical_boost:
                if exp in self.membrane_potentials: self.membrane_potentials[exp] += 4.0
                else: self.membrane_potentials[exp] = 4.0

        spiked_nodes = []
        for prim, potential in self.membrane_potentials.items():
            if potential >= self.spike_threshold:
                spiked_nodes.append((prim, potential))

        spiked_nodes.sort(key=lambda x: x[1], reverse=True)
        self.neuromodulate(len(spiked_nodes))
        
        return [prim for prim, _ in spiked_nodes]
