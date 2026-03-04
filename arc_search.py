"""
arc_search.py — DSL Brute-Force Search and Multi-Candidate Voting
=================================================================
Two complementary search strategies:

  1. BRUTE-FORCE DSL SEARCH
     Enumerate compositions of DSL primitives (depth-limited).
     Exhaustively tests single-function and two-function pipelines.
     Fast enough for simple transformations without LLM calls.
     Falls back to LLM synthesis for complex tasks.

  2. MULTI-CANDIDATE VOTING
     Generate N candidate programs (via LLM with varied prompts/temps).
     Run all candidates against training pairs.
     Vote on test predictions: the prediction most candidates agree on wins.
     Ties broken by training score.

Both strategies record findings in the vexel scroll (EV_RESONANCE on hit,
EV_MISS on miss) and store winning programs in the pattern library.

Search taxonomy (primitives tried in brute force):
  Arity-1 (single transforms):
    identity, rot90, rot180, rot270, reflect_h, reflect_v,
    reflect_diag, reflect_anti, crop_to_content, fill_holes,
    gravity_down, gravity_up, gravity_left, gravity_right,
    complete_h_symmetry, complete_v_symmetry, enforce_h_symmetry,
    dilate, erode, outline, largest_object, smallest_object,
    objects_touching_border, objects_not_touching_border

  Parameterized single (try all valid params):
    recolor(a→b) for all color pairs
    upscale(2), upscale(3), downscale(2)
    gravity(direction) for all 4 directions
    reflect_h, reflect_v composed with recolor

  Arity-2 compositions (f then g):
    all pairs of arity-1 where shapes are compatible
"""

from __future__ import annotations
import time
import itertools
from typing import Callable, Optional
from copy import deepcopy

from arc_types import (
    Grid, Pair, ARCTask, ARCPrediction,
    grid_shape, grid_eq, grid_similarity, grid_copy,
    count_colors, background_color, grid_unique_colors,
    rot90, rot180, rot270, reflect_h, reflect_v,
    reflect_diag, reflect_anti, recolor, upscale, downscale,
    crop_to_content, extract_objects,
    empty_grid,
)


# ── Candidate program ──────────────────────────────────────────────────────────

class Candidate:
    """A candidate program with its training evaluation."""
    def __init__(self, code: str, score: float = 0.0, source: str = ""):
        self.code   = code
        self.score  = score
        self.source = source       # "brute_force" | "llm" | "augment"
        self.predictions: list[Optional[Grid]] = []  # one per test

    def __repr__(self):
        return f"Candidate(score={self.score:.2f}, source={self.source!r})"


# ── Brute-force primitive library ─────────────────────────────────────────────

def _build_arity1_primitives(task: ARCTask) -> list[tuple[str, Callable]]:
    """
    Build list of (name, fn) for all arity-1 DSL primitives to try.
    Parameters are instantiated from the task's color space.
    """
    prims = []
    bg    = background_color(task.train[0].input) if task.train else 0
    colors = sorted(grid_unique_colors(task.train[0].input)) if task.train else []

    # Geometric
    prims.append(("identity",     lambda g: grid_copy(g)))
    prims.append(("rot90",        rot90))
    prims.append(("rot180",       rot180))
    prims.append(("rot270",       rot270))
    prims.append(("reflect_h",    reflect_h))
    prims.append(("reflect_v",    reflect_v))
    prims.append(("reflect_diag", reflect_diag))
    prims.append(("reflect_anti", reflect_anti))
    prims.append(("crop_to_content", lambda g: crop_to_content(g, bg)))

    # Scale
    prims.append(("upscale_2", lambda g: upscale(g, 2)))
    prims.append(("upscale_3", lambda g: upscale(g, 3)))
    prims.append(("downscale_2", lambda g: downscale(g, 2)))

    # Gravity (lazy import from arc_dsl_ext)
    try:
        from arc_dsl_ext import gravity, fill_holes, dilate, erode, outline
        from arc_dsl_ext import (largest_object, smallest_object,
                                  objects_touching_border,
                                  objects_not_touching_border,
                                  complete_h_symmetry, complete_v_symmetry,
                                  enforce_h_symmetry, enforce_v_symmetry)

        prims.append(("gravity_down",   lambda g: gravity(g, "down",  bg)))
        prims.append(("gravity_up",     lambda g: gravity(g, "up",    bg)))
        prims.append(("gravity_left",   lambda g: gravity(g, "left",  bg)))
        prims.append(("gravity_right",  lambda g: gravity(g, "right", bg)))
        prims.append(("fill_holes",     lambda g: fill_holes(g, bg)))
        prims.append(("dilate",         lambda g: dilate(g, bg=bg)))
        prims.append(("erode",          lambda g: erode(g, bg)))
        prims.append(("outline",        lambda g: outline(g, bg)))
        prims.append(("largest_object", lambda g: largest_object(g, bg)))
        prims.append(("smallest_object",lambda g: smallest_object(g, bg)))
        prims.append(("border_objects", lambda g: objects_touching_border(g, bg)))
        prims.append(("interior_objects",lambda g: objects_not_touching_border(g, bg)))
        prims.append(("complete_h_sym", lambda g: complete_h_symmetry(g, bg)))
        prims.append(("complete_v_sym", lambda g: complete_v_symmetry(g, bg)))
        prims.append(("enforce_h_sym",  lambda g: enforce_h_symmetry(g)))
        prims.append(("enforce_v_sym",  lambda g: enforce_v_symmetry(g)))
    except ImportError:
        pass

    # Recolor pairs
    non_bg = [c for c in colors if c != bg]
    for ca in non_bg:
        for cb in range(10):
            if ca != cb:
                ca_, cb_ = ca, cb  # capture
                prims.append((f"recolor_{ca_}_{cb_}",
                               lambda g, a=ca_, b=cb_: recolor(g, a, b)))

    return prims


def _eval_fn(fn: Callable, task: ARCTask) -> float:
    """Evaluate a callable transform on all training pairs. Returns score."""
    if not task.train:
        return 0.0
    correct = 0
    for pair in task.train:
        try:
            result = fn(deepcopy(pair.input))
            if pair.output and result == pair.output:
                correct += 1
        except Exception:
            pass
    return correct / len(task.train)


def _fn_to_code(name: str, fn: Callable, task: ARCTask) -> str:
    """Generate Python source for a named primitive."""
    bg = background_color(task.train[0].input) if task.train else 0

    simple_map = {
        "identity":      "return grid_copy(grid)",
        "rot90":         "return rot90(grid)",
        "rot180":        "return rot180(grid)",
        "rot270":        "return rot270(grid)",
        "reflect_h":     "return reflect_h(grid)",
        "reflect_v":     "return reflect_v(grid)",
        "reflect_diag":  "return reflect_diag(grid)",
        "reflect_anti":  "return reflect_anti(grid)",
        "crop_to_content": f"return crop_to_content(grid, {bg})",
        "upscale_2":     "return upscale(grid, 2)",
        "upscale_3":     "return upscale(grid, 3)",
        "downscale_2":   "return downscale(grid, 2)",
        "gravity_down":  f"return gravity(grid, 'down', {bg})",
        "gravity_up":    f"return gravity(grid, 'up', {bg})",
        "gravity_left":  f"return gravity(grid, 'left', {bg})",
        "gravity_right": f"return gravity(grid, 'right', {bg})",
        "fill_holes":    f"return fill_holes(grid, {bg})",
        "dilate":        f"return dilate(grid, bg={bg})",
        "erode":         f"return erode(grid, {bg})",
        "outline":       f"return outline(grid, {bg})",
        "largest_object":  f"return largest_object(grid, {bg})",
        "smallest_object": f"return smallest_object(grid, {bg})",
        "border_objects":  f"return objects_touching_border(grid, {bg})",
        "interior_objects":f"return objects_not_touching_border(grid, {bg})",
        "complete_h_sym": f"return complete_h_symmetry(grid, {bg})",
        "complete_v_sym": f"return complete_v_symmetry(grid, {bg})",
        "enforce_h_sym": "return enforce_h_symmetry(grid)",
        "enforce_v_sym": "return enforce_v_symmetry(grid)",
    }
    if name in simple_map:
        return f"def transform(grid):\n    {simple_map[name]}"

    if name.startswith("recolor_"):
        parts = name.split("_")
        ca, cb = parts[1], parts[2]
        return f"def transform(grid):\n    return recolor(grid, {ca}, {cb})"

    if name.startswith("compose_"):
        inner = name[8:]
        names = inner.split("__")
        if len(names) == 2:
            n1, n2 = names
            c1 = _fn_to_code(n1, None, task).replace("def transform(grid):\n    return ", "")
            c2 = _fn_to_code(n2, None, task).replace("def transform(grid):\n    return ", "")
            return (f"def transform(grid):\n"
                    f"    step1 = {c1}\n"
                    f"    return {c2.replace('grid', 'step1')}")

    return f"def transform(grid):\n    # {name}\n    return grid_copy(grid)"


def brute_force_search(task: ARCTask,
                        max_depth: int = 2,
                        time_limit: float = 30.0,
                        verbose: bool = False,
                        upg_prim_order: list = None) -> Optional[Candidate]:
    """
    Exhaustively search the DSL primitive space.
    max_depth=1: try all arity-1 primitives.
    max_depth=2: also try all 2-composition pipelines.
    upg_prim_order: if provided (from arc_bra.upg_ordered_primitives),
                    test primitives in UPG geometric order instead of linearly.
    Returns the first perfect-score Candidate, or the best found.
    """
    prims    = _build_arity1_primitives(task)
    t0       = time.time()
    best     = None
    best_sc  = -1.0

    # Respect UPG geometric ordering if provided
    if upg_prim_order:
        prim_map = {name: fn for name, fn in prims}
        ordered  = [(n, prim_map[n]) for n in upg_prim_order if n in prim_map]
        # Append any not in UPG order at end
        ordered += [(n, f) for n, f in prims if n not in {x[0] for x in ordered}]
        prims = ordered

    # Depth 1
    for name, fn in prims:
        if time.time() - t0 > time_limit:
            break
        score = _eval_fn(fn, task)
        if verbose and score > 0:
            print(f"    [bf] {name}: {score:.2f}")
        if score > best_sc:
            best_sc = score
            code    = _fn_to_code(name, fn, task)
            best    = Candidate(code, score, "brute_force")
        if score >= 1.0:
            return best

    if max_depth < 2 or time.time() - t0 > time_limit:
        return best

    # Depth 2: compose pairs
    for (n1, f1), (n2, f2) in itertools.product(prims, prims):
        if time.time() - t0 > time_limit:
            break
        if n1 == "identity" and n2 == "identity":
            continue
        try:
            composed = lambda g, _f1=f1, _f2=f2: _f2(deepcopy(_f1(deepcopy(g))))
            score    = _eval_fn(composed, task)
        except Exception:
            continue
        if score > best_sc:
            best_sc  = score
            comp_name = f"compose_{n1}__{n2}"
            code     = _fn_to_code(comp_name, composed, task)
            best     = Candidate(code, score, "brute_force")
            if verbose and score > 0:
                print(f"    [bf] {comp_name}: {score:.2f}")
        if score >= 1.0:
            return best

    return best


# ── Multi-candidate LLM voting ─────────────────────────────────────────────────

def generate_candidates(task: ARCTask,
                         rule: str,
                         n_candidates: int = 5,
                         verbose: bool = False) -> list[Candidate]:
    """
    Generate N candidate programs via LLM with varied prompts.
    Uses temperature variation and different prompt framings.
    """
    from arc_solver import (
        _call_llm, _extract_python, evaluate_program,
        SYNTH_SYSTEM, build_synthesis_prompt,
    )
    from arc_abstraction import encode_task_compact, ABSTRACT_SYNTHESIS_SYSTEM

    # TENT gate: n_candidates=0 when BRA gate is closed (brute_force pipeline)
    # This prevents any LLM calls on structurally simple tasks.
    if n_candidates == 0:
        return []
    candidates = []
    prompt_variants = [
        # Variant 1: standard
        (build_synthesis_prompt(task, rule), SYNTH_SYSTEM),
        # Variant 2: object-centric compact
        (f"Rule: {rule}\n\n{encode_task_compact(task)}\n\nWrite transform().",
         ABSTRACT_SYNTHESIS_SYSTEM),
        # Variant 3: "think step by step" framing
        (f"Rule: {rule}\n\n"
         f"Think step by step about what each cell or object must do.\n"
         f"Then write transform().\n\n{encode_task_compact(task)}",
         SYNTH_SYSTEM),
        # Variant 4: concise framing
        (f"Rule: {rule}\n\n"
         f"Write the shortest correct transform() that implements this rule.\n\n"
         f"{encode_task_compact(task)}",
         SYNTH_SYSTEM),
        # Variant 5: explicit edge-case handling
        (f"Rule: {rule}\n\n"
         f"Implement transform(). Handle all edge cases.\n"
         f"Check: does your program handle the case where there are "
         f"zero, one, and multiple objects?\n\n{encode_task_compact(task)}",
         SYNTH_SYSTEM),
    ]

    for i in range(n_candidates):
        prompt, system = prompt_variants[i % len(prompt_variants)]
        if verbose:
            print(f"    [cand] generating candidate {i+1}/{n_candidates}...")
        try:
            resp = _call_llm(
                [{"role": "user", "content": prompt}],
                system=system, max_tokens=1200
            )
            code  = _extract_python(resp)
            ev    = evaluate_program(code, task)
            c     = Candidate(code, ev["score"], "llm")
            candidates.append(c)
            if verbose:
                print(f"    [cand] candidate {i+1}: score={ev['score']:.2f}")
            if ev["score"] >= 1.0:
                break
        except Exception as e:
            if verbose:
                print(f"    [cand] candidate {i+1} failed: {e}")

    return sorted(candidates, key=lambda c: -c.score)


# ── Voting ─────────────────────────────────────────────────────────────────────

def vote_predictions(candidates: list[Candidate],
                     n_test: int) -> list[Optional[Grid]]:
    """
    Run all candidates against test inputs and vote.
    Candidates weighted by training score.
    Returns voted predictions (one per test input).
    """
    from arc_solver import execute_program

    # Collect predictions per test index
    pred_buckets: list[dict] = [{} for _ in range(n_test)]

    for cand in candidates:
        for ti in range(n_test):
            # Need test input — extracted from task via caller
            if ti < len(cand.predictions) and cand.predictions[ti] is not None:
                pred    = cand.predictions[ti]
                key     = str(pred)
                if key not in pred_buckets[ti]:
                    pred_buckets[ti][key] = {"grid": pred, "weight": 0.0, "count": 0}
                pred_buckets[ti][key]["weight"] += cand.score
                pred_buckets[ti][key]["count"]  += 1

    voted = []
    for ti in range(n_test):
        bucket = pred_buckets[ti]
        if not bucket:
            voted.append(None)
        else:
            best_key = max(bucket, key=lambda k: (bucket[k]["weight"], bucket[k]["count"]))
            voted.append(bucket[best_key]["grid"])

    return voted


def fill_candidate_predictions(candidates: list[Candidate],
                                task: ARCTask) -> list[Candidate]:
    """Run each candidate's program against all test inputs and store results."""
    from arc_solver import execute_program

    for cand in candidates:
        cand.predictions = []
        for pair in task.test:
            pred, err = execute_program(cand.code, pair.input)
            cand.predictions.append(pred if not err else None)
    return candidates


# ── Full search pipeline ───────────────────────────────────────────────────────

def search_and_vote(task: ARCTask,
                    rule: str,
                    n_llm_candidates: int = 3,
                    brute_force_first: bool = True,
                    brute_time_limit: float = 20.0,
                    verbose: bool = False) -> dict:
    """
    Combined search + vote pipeline.

    1. Brute-force DSL search (fast, catches simple transforms)
    2. LLM multi-candidate generation
    3. Vote on test predictions

    Returns:
      {
        best_program: str,
        best_score: float,
        voted_predictions: list[Optional[Grid]],
        candidates: list[Candidate],
        brute_force_found: bool,
      }
    """
    all_candidates = []
    bf_found = False

    # Stage A: brute force
    if brute_force_first:
        if verbose:
            print("    [search] Running brute-force DSL search...")
        bf_cand = brute_force_search(
            task, max_depth=2, time_limit=brute_time_limit, verbose=verbose)
        if bf_cand:
            all_candidates.append(bf_cand)
            if verbose:
                print(f"    [search] Brute-force best: {bf_cand.score:.2f}")
            if bf_cand.score >= 1.0:
                bf_found = True

    # Stage B: LLM candidates (skip if brute force already perfect)
    if not bf_found and n_llm_candidates > 0:
        if verbose:
            print(f"    [search] Generating {n_llm_candidates} LLM candidates...")
        llm_cands = generate_candidates(task, rule, n_llm_candidates, verbose)
        all_candidates.extend(llm_cands)

    if not all_candidates:
        return {
            "best_program": "", "best_score": 0.0,
            "voted_predictions": [None]*len(task.test),
            "candidates": [], "brute_force_found": False,
        }

    # Fill test predictions for all candidates
    all_candidates = fill_candidate_predictions(all_candidates, task)

    # Vote
    voted = vote_predictions(all_candidates, len(task.test))

    # Best program = highest training score
    best = max(all_candidates, key=lambda c: c.score)

    return {
        "best_program":        best.code,
        "best_score":          best.score,
        "voted_predictions":   voted,
        "candidates":          all_candidates,
        "brute_force_found":   bf_found,
        "n_candidates":        len(all_candidates),
    }


# ── Program ensemble ──────────────────────────────────────────────────────────

def ensemble_programs(programs: list[str],
                      task: ARCTask,
                      weights: list[float] = None) -> list[Optional[Grid]]:
    """
    Run an ensemble of programs against all test inputs.
    Weights default to equal. Returns voted predictions.
    """
    from arc_solver import execute_program

    weights = weights or [1.0] * len(programs)
    n_test  = len(task.test)
    buckets: list[dict] = [{} for _ in range(n_test)]

    for prog, w in zip(programs, weights):
        for ti, pair in enumerate(task.test):
            pred, err = execute_program(prog, pair.input)
            if not err and pred is not None:
                key = str(pred)
                if key not in buckets[ti]:
                    buckets[ti][key] = {"grid": pred, "weight": 0.0}
                buckets[ti][key]["weight"] += w

    voted = []
    for ti in range(n_test):
        b = buckets[ti]
        if not b:
            voted.append(None)
        else:
            best = max(b, key=lambda k: b[k]["weight"])
            voted.append(b[best]["grid"])

    return voted


# ── Pattern-library warm-start search ────────────────────────────────────────

def warm_search(task: ARCTask, rule: str,
                library=None, verbose: bool = False) -> Optional[Candidate]:
    """
    Before running brute force or LLM, check the pattern library for a
    previously solved program that matches this task's structure.

    Search order:
      1. BRA eigenstate fingerprint (primary) — task_charge() integer resonance,
         min_resonance=2 (exact structural match). Finds the same transformation
         applied to a different task_id. Deterministic, no text matching.
      2. task_id string lookup (fallback) — catches replayed identical tasks.
      3. Category keyword filter — last resort for rule-text hints.
    """
    if library is None:
        return None

    from arc_solver import evaluate_program

    # 1. PRIMARY: BRA task fingerprint — finds same transformation pattern
    #    regardless of task_id. This is the sovereign path.
    bra_hits = library.lookup(task, limit=5)  # ARCTask → BRA charge scan
    for rec in bra_hits:
        ev = evaluate_program(rec.program, task)
        if ev["score"] >= 1.0:
            if verbose:
                print(f"    [warm] BRA eigenstate hit: {rec.task_id} resonance=2 score=1.0")
            return Candidate(rec.program, ev["score"], "bra_eigenstate")
        if ev["score"] >= 0.5:
            if verbose:
                print(f"    [warm] BRA near hit: {rec.task_id} score={ev['score']:.2f}")
            return Candidate(rec.program, ev["score"], "bra_near")

    # 2. FALLBACK: task_id string match — catches exact replay of known tasks
    exact = library._db.best_for_task(task.task_id)
    if exact and exact.train_score >= 1.0:
        ev = evaluate_program(exact.program, task)
        if ev["score"] >= 0.5:
            if verbose:
                print(f"    [warm] task_id hit: {task.task_id} score={ev['score']:.2f}")
            return Candidate(exact.program, ev["score"], "library_exact")

    # 3. LAST RESORT: category keyword filter from rule text
    if rule:
        matches = library.lookup(rule, limit=3)
        for m in matches:
            if m.train_score >= 0.9:
                ev = evaluate_program(m.program, task)
                if ev["score"] >= 0.5:
                    if verbose:
                        print(f"    [warm] category hit: {m.task_id} score={ev['score']:.2f}")
                    return Candidate(m.program, ev["score"], "library_category")

    return None
