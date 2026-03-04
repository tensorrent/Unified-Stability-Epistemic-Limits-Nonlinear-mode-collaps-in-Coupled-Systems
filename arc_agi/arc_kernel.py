from __future__ import annotations
import ctypes, os
from pathlib import Path
from typing import Optional, List, Tuple
Grid = list[list[int]]

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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libarc_heuristics.dylib"),
    "/Volumes/Seagate 4tb/unified_field_theory/arc_agi/libarc_heuristics.dylib",
    "/tmp/libarc_heuristics.so",
    "libarc_heuristics.so",
]

def _load():
    for p in _LIB_PATHS:
        if Path(p).exists():
            lib = ctypes.CDLL(p); _bind(lib); return lib
    raise FileNotFoundError("libarc_heuristics.so not found")

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
    h,w = len(grid), len(grid[0]) if grid else 0
    flat = (ctypes.c_uint8*(w*h))()
    for r in range(h):
        for c in range(w): flat[r*w+c] = grid[r][c]&0xFF
    return flat, w, h

def _from_c(flat,w,h):
    return [[flat[r*w+c] for c in range(w)] for r in range(h)]

def verify(): return get_lib().arc_neuro_verify()==1

def learn_rules(in_grid,out_grid,wide=False):
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

def solve_by_local_rules(task,wide=False):
    pairs=[(p.input,p.output) for p in task.train if p.output is not None]
    if not pairs: return None
    rules=learn_from_pairs(pairs,wide=wide)
    if not rules: return None
    for inp,out in pairs:
        if apply_rules(inp,rules)!=out:
            if apply_rules(inp,rules,iterative=True)!=out: return None
    return apply_rules(task.test[0].input,rules) if task.test else None

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

def solve_by_period(task):
    pairs=[(p.input,p.output) for p in task.train if p.output is not None]
    if not pairs: return None
    test_inp=task.test[0].input if task.test else None
    if test_inp is None: return None
    for inp,out in pairs:
        hp,vp=pattern_period(inp)
        if hp==0 and vp==0: return None
        if len(out)<=len(inp) and len(out[0])<=len(inp[0]): return None
    inp0,out0=pairs[0]
    ih,iw=len(inp0),len(inp0[0]); oh,ow=len(out0),len(out0[0])
    th,tw=len(test_inp),len(test_inp[0])
    th2=th*oh//ih if ih>0 else oh; tw2=tw*ow//iw if iw>0 else ow
    return [[test_inp[r%th][c%tw] for c in range(tw2)] for r in range(th2)]

def neuro_solve(task):
    r={"prediction":None,"tier":None,"n_rules":0,"confidence":0}
    for wide,tier in [(False,"local_3x3"),(True,"local_5x5")]:
        try:
            pred=solve_by_local_rules(task,wide=wide)
            if pred is not None:
                pairs=[(p.input,p.output) for p in task.train if p.output]
                rules=learn_from_pairs(pairs,wide=wide)
                r.update({"prediction":pred,"tier":tier,"n_rules":len(rules),"confidence":2})
                return r
        except Exception: pass
    for fn,tier in [(solve_by_delta,"delta"),(solve_by_period,"period")]:
        try:
            pred=fn(task)
            if pred is not None:
                r.update({"prediction":pred,"tier":tier,"n_rules":1,"confidence":1})
                return r
        except Exception: pass
    return r

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
    th,tw=len(test_inp),len(test_inp[0])
    if th==0 or tw==0: return None
    return [[test_inp[r%th][c%tw] for c in range(ow)] for r in range(oh)]

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
