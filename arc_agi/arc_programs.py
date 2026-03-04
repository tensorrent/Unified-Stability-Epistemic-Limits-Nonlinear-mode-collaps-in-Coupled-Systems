"""
arc_programs.py — Hand-Crafted ARC Pattern Library
====================================================
~60 common ARC-AGI transformation patterns coded directly.
Tried before LLM calls — instant, deterministic, perfect.
"""
from __future__ import annotations
from typing import Optional
from arc_types import (
    ARCTask, Grid, grid_shape, background_color,
    grid_unique_colors, count_colors, detect_symmetry,
    extract_objects,
)

# ── Heuristic helpers ──────────────────────────────────────────────────────────

def _shapes_same(t):
    return all(grid_shape(p.input)==grid_shape(p.output) for p in t.train if p.output)
def _shapes_change(t):
    return any(grid_shape(p.input)!=grid_shape(p.output) for p in t.train if p.output)
def _scaled(t, f):
    return all(grid_shape(p.output)==(grid_shape(p.input)[0]*f,grid_shape(p.input)[1]*f)
               for p in t.train if p.output)
def _downscaled(t, f):
    return all(grid_shape(p.output)==(grid_shape(p.input)[0]//f,grid_shape(p.input)[1]//f)
               for p in t.train if p.output)
def _square(t):
    return all(grid_shape(p.input)[0]==grid_shape(p.input)[1] for p in t.train)
def _n_colors(t):
    s=set()
    [s.update(c for row in p.input for c in row) for p in t.train]
    return len(s)
def _h_double(t):
    return all(grid_shape(p.output)[1]==grid_shape(p.input)[1]*2 for p in t.train if p.output)
def _v_double(t):
    return all(grid_shape(p.output)[0]==grid_shape(p.input)[0]*2 for p in t.train if p.output)
def _sym_gain(t):
    for p in t.train:
        if not p.output: continue
        if sum(detect_symmetry(p.output).values())>sum(detect_symmetry(p.input).values()):
            return True
    return False
def _always(t): return True

# ── Pattern registry ───────────────────────────────────────────────────────────

PATTERNS = []

def _p(name, cat, code, test=_always):
    PATTERNS.append({"name":name,"category":cat,"code":code.strip(),"test":test})

# GEOMETRIC
_p("rot90","geometric","def transform(g): return rot90(g)", lambda t:_square(t) or _shapes_change(t))
_p("rot180","geometric","def transform(g): return rot180(g)",_shapes_same)
_p("rot270","geometric","def transform(g): return rot270(g)", lambda t:_square(t) or _shapes_change(t))
_p("reflect_h","geometric","def transform(g): return reflect_h(g)",_shapes_same)
_p("reflect_v","geometric","def transform(g): return reflect_v(g)",_shapes_same)
_p("reflect_diag","geometric","def transform(g): return reflect_diag(g)",_square)
_p("reflect_anti","geometric","def transform(g): return reflect_anti(g)",_square)
_p("upscale_2","geometric","def transform(g): return upscale(g,2)",lambda t:_scaled(t,2))
_p("upscale_3","geometric","def transform(g): return upscale(g,3)",lambda t:_scaled(t,3))
_p("downscale_2","geometric","def transform(g): return downscale(g,2)",lambda t:_downscaled(t,2))
_p("rot90_flip_h","geometric","def transform(g): return reflect_h(rot90(g))",_always)
_p("rot90_flip_v","geometric","def transform(g): return reflect_v(rot90(g))",_always)

# COLOR
_p("invert_2color","color","""
def transform(g):
    cs=sorted(set(c for row in g for c in row))
    return replace_colors(g,{cs[0]:cs[-1],cs[-1]:cs[0]}) if len(cs)==2 else [r[:] for r in g]
""",lambda t:_n_colors(t)==2)

for _a,_b in [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2),(1,4),(4,1),(2,4),(4,2),
               (3,4),(4,3),(1,5),(5,1),(2,5),(5,2),(3,5),(5,3)]:
    _p(f"swap_{_a}_{_b}","color",
       f"def transform(g): return replace_colors(g,{{{_a}:{_b},{_b}:{_a}}})",
       lambda t,a=_a,b=_b:{a,b}<=grid_unique_colors(t.train[0].input))

_p("all_fg_to_1","color","""
def transform(g):
    bg=background_color(g)
    return [[1 if c!=bg else 0 for c in r] for r in g]
""",lambda t:_n_colors(t)<=3)

_p("crop_to_content","color","def transform(g): return crop_to_content(g)",_shapes_change)
_p("normalize_colors","color","def transform(g): return normalize_colors(g)",_always)

# Unidirectional recolor (a→b, don't require b in input)
for _a, _b in [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2),(1,4),(4,1),(1,5),(5,1),
               (2,4),(4,2),(2,5),(5,2),(3,4),(4,3),(3,5),(5,3),
               (1,0),(2,0),(3,0),(4,0),(5,0),
               (0,1),(0,2),(0,3),(0,4),(0,5)]:
    _p(f"recolor_{_a}_to_{_b}","color",
       f"def transform(g): return recolor(g,{_a},{_b})",
       lambda t,a=_a: a in grid_unique_colors(t.train[0].input) if t.train else False)
del _a, _b

# SYMMETRY
_p("complete_h","symmetry","def transform(g): return complete_h_symmetry(g)",_sym_gain)
_p("complete_v","symmetry","def transform(g): return complete_v_symmetry(g)",_sym_gain)
_p("complete_rot180","symmetry","def transform(g): return complete_rot180_symmetry(g)",_sym_gain)
_p("complete_diag","symmetry","def transform(g): return complete_diagonal_symmetry(g)",lambda t:_square(t)and _sym_gain(t))
_p("enforce_h","symmetry","def transform(g): return enforce_h_symmetry(g)",_shapes_same)
_p("enforce_v","symmetry","def transform(g): return enforce_v_symmetry(g)",_shapes_same)
_p("4fold_sym","symmetry","def transform(g): return complete_v_symmetry(complete_h_symmetry(g))",_sym_gain)

# GRAVITY
_p("gravity_down","gravity","def transform(g): return gravity(g,'down')",_shapes_same)
_p("gravity_up","gravity","def transform(g): return gravity(g,'up')",_shapes_same)
_p("gravity_left","gravity","def transform(g): return gravity(g,'left')",_shapes_same)
_p("gravity_right","gravity","def transform(g): return gravity(g,'right')",_shapes_same)
_p("gravity_blocked_down","gravity","def transform(g): return gravity_blocked(g,'down')",_shapes_same)

# OBJECT
_p("keep_largest","object","def transform(g): return largest_object(g)",_shapes_same)
_p("keep_smallest","object","def transform(g): return smallest_object(g)",_shapes_same)
_p("keep_border","object","def transform(g): return objects_touching_border(g)",_shapes_same)
_p("keep_interior","object","def transform(g): return objects_not_touching_border(g)",_shapes_same)
_p("rank_by_size","object","def transform(g): return color_objects_by_rank(g,'size')",_shapes_same)
_p("fill_holes","morphology","def transform(g): return fill_holes(g)",_shapes_same)
_p("outline","morphology","def transform(g): return outline(g)",_shapes_same)
_p("dilate","morphology","def transform(g): return dilate(g)",_shapes_same)
_p("erode","morphology","def transform(g): return erode(g)",_shapes_same)
_p("convex_hull","morphology","def transform(g): return convex_hull_fill(g)",_shapes_same)

# LOGIC
_p("xor_rot180","logic","def transform(g): return grid_xor(g,rot180(g))",lambda t:_shapes_same(t)and _square(t))
_p("and_reflect_h","logic","def transform(g): return grid_and(g,reflect_h(g))",_shapes_same)
_p("or_reflect_v","logic","def transform(g): return grid_or(g,reflect_v(g))",_shapes_same)

# SPLIT/JOIN
_p("hstack_self","split","def transform(g): return hstack(g,g)",_h_double)
_p("vstack_self","split","def transform(g): return vstack(g,g)",_v_double)
_p("hstack_reflected","split","def transform(g): return hstack(g,reflect_h(g))",_h_double)
_p("vstack_reflected","split","def transform(g): return vstack(g,reflect_v(g))",_v_double)
_p("tile_2x2","split","def transform(g): return tile(g,2,2)",lambda t:_scaled(t,2))
_p("hstack_rot180","split","def transform(g): return hstack(g,rot180(g))",_h_double)

# PATTERN
_p("repeat_h3","pattern","def transform(g): return repeat_pattern(g,3,'h')",
   lambda t:all(grid_shape(p.output)[1]==grid_shape(p.input)[1]*3 for p in t.train if p.output))
_p("repeat_v3","pattern","def transform(g): return repeat_pattern(g,3,'v')",
   lambda t:all(grid_shape(p.output)[0]==grid_shape(p.input)[0]*3 for p in t.train if p.output))


# ── Executor ───────────────────────────────────────────────────────────────────

def _exec_globals():
    from arc_types import (
        grid_height,grid_width,grid_shape,grid_copy,empty_grid,
        rot90,rot180,rot270,reflect_h,reflect_v,reflect_diag,reflect_anti,
        crop,pad,tile,recolor,replace_colors,fill,overlay,hstack,vstack,
        upscale,downscale,extract_objects,background_color,crop_to_content,
        count_colors,most_common_color,least_common_color,detect_symmetry,
        grid_unique_colors,grid_color_count,grid_diff,grid_similarity,
        grid_set,grid_get,grid_eq,
    )
    g = {
        "grid_height":grid_height,"grid_width":grid_width,"grid_shape":grid_shape,
        "grid_copy":grid_copy,"empty_grid":empty_grid,"rot90":rot90,"rot180":rot180,
        "rot270":rot270,"reflect_h":reflect_h,"reflect_v":reflect_v,
        "reflect_diag":reflect_diag,"reflect_anti":reflect_anti,"crop":crop,"pad":pad,
        "tile":tile,"recolor":recolor,"replace_colors":replace_colors,"fill":fill,
        "overlay":overlay,"hstack":hstack,"vstack":vstack,"upscale":upscale,
        "downscale":downscale,"extract_objects":extract_objects,
        "background_color":background_color,"crop_to_content":crop_to_content,
        "count_colors":count_colors,"most_common_color":most_common_color,
        "least_common_color":least_common_color,"detect_symmetry":detect_symmetry,
        "grid_unique_colors":grid_unique_colors,"grid_color_count":grid_color_count,
        "grid_diff":grid_diff,"grid_similarity":grid_similarity,
        "grid_set":grid_set,"grid_get":grid_get,"grid_eq":grid_eq,
    }
    try:
        from arc_types import normalize_colors
        g["normalize_colors"]=normalize_colors
    except ImportError: pass
    try:
        from arc_dsl_ext import EXT_DSL_NAMESPACE
        g.update(EXT_DSL_NAMESPACE)
    except ImportError: pass
    return g

def _run(code, grid):
    import threading
    from copy import deepcopy
    res=[None]; err=[""]
    globs=_exec_globals()
    try:
        exec(compile(code,"<pat>","exec"),globs)
        fn=globs.get("transform")
        if not fn: return None,"no fn"
        def _go():
            try: res[0]=fn(deepcopy(grid))
            except Exception as e: err[0]=str(e)
        t=threading.Thread(target=_go,daemon=True)
        t.start(); t.join(5.0)
        if t.is_alive(): return None,"timeout"
        return (res[0],"") if not err[0] else (None,err[0])
    except Exception as e: return None,str(e)

def match_program(task: ARCTask, min_score: float=1.0,
                  verbose: bool=False) -> Optional[dict]:
    best=None; best_sc=-1.0
    for pat in PATTERNS:
        try:
            if not pat["test"](task): continue
        except: pass
        correct=0; total=len(task.train)
        if not total: continue
        ok=True
        for pair in task.train:
            r,e=_run(pat["code"],pair.input)
            if e: ok=False; break
            if pair.output is not None and r==pair.output: correct+=1
        if not ok and correct==0: continue
        sc=correct/total
        if verbose and sc>0: print(f"    [pat] {pat['name']}: {sc:.2f}")
        if sc>best_sc:
            best_sc=sc
            best={"name":pat["name"],"code":pat["code"],"score":sc,"category":pat["category"]}
        if sc>=min_score: return best
    return best if best and best_sc>0 else None

def list_patterns():
    return [{"name":p["name"],"category":p["category"]} for p in PATTERNS]

if __name__=="__main__":
    print(f"Patterns: {len(PATTERNS)}")
    from collections import Counter
    cats=Counter(p["category"] for p in PATTERNS)
    for c,n in sorted(cats.items()): print(f"  {c}: {n}")
