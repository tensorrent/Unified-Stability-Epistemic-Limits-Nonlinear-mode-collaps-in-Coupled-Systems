"""
arc_augment.py — ARC Task Augmentation
=======================================
Generates augmented versions of an ARC task for:
  1. Test-time consistency checking — solve all augmented views,
     inverse-transform the predictions, vote on the best answer.
  2. Candidate generation — if a program scores 0 on the original
     orientation, try all 8 D4 transforms; a program that works on
     the rotated version can be un-rotated.
  3. Confidence estimation — if the majority of augmented views agree,
     confidence is high; disagreement → low confidence.

D4 group (8 symmetries of the square):
  identity, rot90, rot180, rot270,
  reflect_h, reflect_v, reflect_diag, reflect_anti

Color permutation:
  Systematically swaps non-background color pairs to generate
  color-invariant augmentations (useful for tasks where color assignment
  is arbitrary).

All augmentations are invertible — we always know how to undo them.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Callable

from arc_types import (
    Grid, Pair, ARCTask, ARCPrediction,
    grid_copy, grid_shape, grid_height, grid_width,
    rot90, rot180, rot270, reflect_h, reflect_v,
    reflect_diag, reflect_anti, replace_colors,
    background_color, count_colors, grid_eq,
)


# ── D4 transforms ─────────────────────────────────────────────────────────────

@dataclass
class Transform:
    name:    str
    forward: Callable[[Grid], Grid]
    inverse: Callable[[Grid], Grid]

    def apply(self, g: Grid) -> Grid:
        return self.forward(g)

    def undo(self, g: Grid) -> Grid:
        return self.inverse(g)


# The 8 elements of D4
D4_TRANSFORMS = [
    Transform("identity",    lambda g: grid_copy(g),  lambda g: grid_copy(g)),
    Transform("rot90",       rot90,                    rot270),
    Transform("rot180",      rot180,                   rot180),
    Transform("rot270",      rot270,                   rot90),
    Transform("reflect_h",   reflect_h,                reflect_h),
    Transform("reflect_v",   reflect_v,                reflect_v),
    Transform("reflect_d",   reflect_diag,             reflect_diag),
    Transform("reflect_a",   reflect_anti,             reflect_anti),
]

D4_MAP = {t.name: t for t in D4_TRANSFORMS}


def apply_d4_to_pair(pair: Pair, transform: Transform) -> Pair:
    """Apply a D4 transform to both input and output of a pair."""
    inp = transform.apply(pair.input)
    out = transform.apply(pair.output) if pair.output else None
    return Pair(input=inp, output=out)


def apply_d4_to_task(task: ARCTask, transform: Transform) -> ARCTask:
    """Apply a D4 transform to all pairs in a task."""
    return ARCTask(
        task_id = f"{task.task_id}_{transform.name}",
        train   = [apply_d4_to_pair(p, transform) for p in task.train],
        test    = [apply_d4_to_pair(p, transform) for p in task.test],
        source  = task.source,
    )


def unapply_d4(grid: Grid, transform: Transform) -> Grid:
    """Undo a D4 transform from a predicted grid."""
    return transform.undo(grid)


# ── Color permutation augmentation ────────────────────────────────────────────

def _nonbg_colors(task: ARCTask) -> list[int]:
    """Return sorted list of non-background colors seen in training pairs."""
    bg = background_color(task.train[0].input) if task.train else 0
    colors = set()
    for pair in task.train:
        for g in [pair.input, pair.output]:
            if g:
                colors.update(
                    c for row in g for c in row if c != bg
                )
    return sorted(colors)


def _make_color_mapping(colors: list[int],
                         perm: tuple) -> dict[int, int]:
    """Build color mapping dict from permutation of colors."""
    return {c: perm[i] for i, c in enumerate(colors)}


def _apply_color_map(task: ARCTask, mapping: dict) -> ARCTask:
    """Apply a color remapping to all grids in a task."""
    def remap(g: Grid) -> Grid:
        return replace_colors(g, mapping)

    new_train = []
    for p in task.train:
        new_train.append(Pair(
            input  = remap(p.input),
            output = remap(p.output) if p.output else None,
        ))
    new_test = []
    for p in task.test:
        new_test.append(Pair(
            input  = remap(p.input),
            output = remap(p.output) if p.output else None,
        ))
    return ARCTask(
        task_id = task.task_id + "_cperm",
        train   = new_train,
        test    = new_test,
        source  = task.source,
    )


def color_permutation_augments(task: ARCTask,
                                max_perms: int = 6) -> list[ARCTask]:
    """
    Generate tasks with systematically permuted non-background colors.
    Returns up to max_perms augmented tasks (excluding identity).
    """
    colors = _nonbg_colors(task)
    if len(colors) < 2:
        return []

    # Limit: only swap pairs, not full permutations (too many)
    aug_tasks = []
    swaps = list(itertools.combinations(range(len(colors)), 2))
    for i, j in swaps[:max_perms]:
        perm    = list(colors)
        perm[i], perm[j] = perm[j], perm[i]
        mapping = _make_color_mapping(colors, perm)
        inv_map = {v: k for k, v in mapping.items()}
        aug_tasks.append((_apply_color_map(task, mapping), inv_map))

    return aug_tasks   # list of (augmented_task, inverse_mapping)


# ── Full augmentation suite ────────────────────────────────────────────────────

@dataclass
class AugmentedView:
    task:         ARCTask
    d4:           Transform
    color_map:    dict      # applied color mapping (color → color)
    inv_color:    dict      # inverse color mapping
    aug_name:     str

    def unapply(self, grid: Grid) -> Grid:
        """Inverse-transform a predicted grid back to original space."""
        # First undo color permutation
        g = replace_colors(grid, self.inv_color) if self.inv_color else grid
        # Then undo D4
        return self.d4.undo(g)


def generate_augmented_views(task: ARCTask,
                              d4_subset: list[str] = None,
                              include_color_perms: bool = True,
                              max_color_perms: int = 4) -> list[AugmentedView]:
    """
    Generate all augmented views of a task.

    d4_subset: list of D4 names to include (default: all 8).
    include_color_perms: whether to include color swaps.
    max_color_perms: max color swap augmentations.

    Returns list of AugmentedView.
    """
    d4_names  = d4_subset or [t.name for t in D4_TRANSFORMS]
    views     = []

    # D4 augments (geometric)
    for name in d4_names:
        t        = D4_MAP[name]
        aug_task = apply_d4_to_task(task, t)
        views.append(AugmentedView(
            task      = aug_task,
            d4        = t,
            color_map = {},
            inv_color = {},
            aug_name  = name,
        ))

    # Color permutation augments
    if include_color_perms:
        color_augs = color_permutation_augments(task, max_color_perms)
        for aug_task, inv_map in color_augs:
            # Identity D4 + color swap
            fwd_map = {v: k for k, v in inv_map.items()}
            views.append(AugmentedView(
                task      = aug_task,
                d4        = D4_MAP["identity"],
                color_map = fwd_map,
                inv_color = inv_map,
                aug_name  = f"color_swap",
            ))

    return views


# ── Augmented-view solver integration ────────────────────────────────────────

def solve_with_augmentation(
        task: ARCTask,
        solver_fn: Callable,          # fn(ARCTask) -> list[Grid | None]
        d4_subset: list[str] = None,
        include_color_perms: bool = False,
        max_views: int = 4,
        verbose: bool = False,
) -> list[Grid]:
    """
    Solve a task using multiple augmented views and majority vote.

    solver_fn: takes an ARCTask, returns list[Optional[Grid]] (one per test).
    Returns: list[Optional[Grid]] — voted best predictions.

    Strategy:
      1. Generate up to max_views augmented versions of the task.
      2. Solve each version.
      3. Inverse-transform each prediction back to original space.
      4. Vote by exact-match frequency; ties broken by first occurrence.
    """
    views = generate_augmented_views(
        task,
        d4_subset         = d4_subset or ["identity", "rot90", "rot180", "rot270"],
        include_color_perms = include_color_perms,
        max_color_perms   = 2,
    )[:max_views]

    n_test = len(task.test)
    all_preds: list[list] = [[] for _ in range(n_test)]

    for view in views:
        if verbose:
            print(f"    [aug] view={view.aug_name}")
        try:
            preds = solver_fn(view.task)
            for i, pred in enumerate(preds):
                if pred is not None:
                    # Inverse-transform back to original space
                    restored = view.unapply(pred)
                    all_preds[i].append(restored)
        except Exception as e:
            if verbose:
                print(f"    [aug] view={view.aug_name} failed: {e}")

    # Vote per test input
    voted = []
    for i in range(n_test):
        candidates = all_preds[i]
        if not candidates:
            voted.append(None)
            continue
        # Stringify for comparison
        counts: dict[str, list] = {}
        for c in candidates:
            key = str(c)
            if key not in counts:
                counts[key] = c
        # Most frequent
        freq: dict[str, int] = {}
        for c in candidates:
            key = str(c)
            freq[key] = freq.get(key, 0) + 1
        best_key = max(freq, key=freq.get)
        voted.append(counts[best_key])

    return voted


# ── Test-time augmented evaluation ────────────────────────────────────────────

def augmented_eval(task: ARCTask, program: str) -> dict:
    """
    Evaluate a program under all 8 D4 augmentations.
    A program that fails on the original orientation may succeed
    on a transformed version — useful for detecting near-correct programs.

    Returns dict with per-augmentation scores and best orientation.
    """
    from arc_solver import execute_program

    results = {}
    for t in D4_TRANSFORMS:
        aug_task = apply_d4_to_task(task, t)
        correct  = 0
        total    = len(aug_task.train)
        for pair in aug_task.train:
            pred, err = execute_program(program, pair.input)
            if not err and pred is not None and pair.output is not None:
                if pred == pair.output:
                    correct += 1
        results[t.name] = {
            "score":   correct / total if total else 0.0,
            "correct": correct,
            "total":   total,
        }

    best_name  = max(results, key=lambda k: results[k]["score"])
    best_score = results[best_name]["score"]

    return {
        "per_aug": results,
        "best":    best_name,
        "best_score": best_score,
        "orientation_sensitive": any(
            results[k]["score"] != results["identity"]["score"]
            for k in results
        ),
    }


def wrap_program_for_augment(program: str, aug_name: str) -> str:
    """
    Wrap an existing program so it first applies a D4 transform,
    runs the original logic, then inverse-transforms the result.
    Useful when the best scoring augmentation is not identity.
    """
    t = D4_MAP.get(aug_name)
    if not t or aug_name == "identity":
        return program

    fwd = aug_name
    inv_map = {
        "rot90":      "rot270",
        "rot270":     "rot90",
        "rot180":     "rot180",
        "reflect_h":  "reflect_h",
        "reflect_v":  "reflect_v",
        "reflect_d":  "reflect_diag",
        "reflect_a":  "reflect_anti",
    }
    inv = inv_map.get(aug_name, "identity")

    fwd_call = {
        "rot90":     "rot90",
        "rot180":    "rot180",
        "rot270":    "rot270",
        "reflect_h": "reflect_h",
        "reflect_v": "reflect_v",
        "reflect_d": "reflect_diag",
        "reflect_a": "reflect_anti",
    }.get(aug_name, "lambda g: g")

    inv_call = {
        "rot90":     "rot270",
        "rot270":    "rot90",
        "rot180":    "rot180",
        "reflect_h": "reflect_h",
        "reflect_v": "reflect_v",
        "reflect_d": "reflect_diag",
        "reflect_a": "reflect_anti",
    }.get(aug_name, "lambda g: g")

    wrapped = f"""
# Auto-wrapped for {aug_name} augmentation
def _inner_transform(grid):
{chr(10).join('    ' + line for line in program.strip().splitlines())}

def transform(grid):
    rotated = {fwd_call}(grid)
    result  = _inner_transform(rotated)
    return {inv_call}(result)
"""
    return wrapped.strip()


# ── Candidate generation via augmentation ────────────────────────────────────

def generate_augmented_programs(program: str, task: ARCTask,
                                 min_score: float = 0.5) -> list[dict]:
    """
    Try all D4 orientations; return wrapped programs that achieve ≥ min_score.
    Used as additional candidates when the original program under-performs.
    """
    from arc_solver import evaluate_program

    candidates = []
    aug_result = augmented_eval(task, program)

    for name, res in aug_result["per_aug"].items():
        if res["score"] >= min_score and name != "identity":
            wrapped = wrap_program_for_augment(program, name)
            candidates.append({
                "program":  wrapped,
                "score":    res["score"],
                "aug_name": name,
            })

    return sorted(candidates, key=lambda x: -x["score"])
