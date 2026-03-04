"""
arc_types.py — ARC-AGI Core Types and DSL Primitives
=====================================================
Provides the foundational data structures and Core Knowledge DSL for ARC-AGI.

ARC task format (JSON):
  {
    "train": [{"input": [[...]], "output": [[...]]}, ...],
    "test":  [{"input": [[...]]}, ...]
  }

Grid: 2D list of integers 0-9 (colors).
Each task has 2-5 training pairs showing input→output transformations.
The solver must infer the rule and apply it to each test input.

ARC Color map (Chollet canonical):
  0 = black    1 = blue     2 = red      3 = green    4 = yellow
  5 = gray     6 = magenta  7 = orange   8 = azure    9 = maroon

DSL primitives implemented here are fully deterministic — no ML needed.
They form the search space for program synthesis.
"""

from __future__ import annotations
import copy
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── ARC color palette (RGB tuples) ────────────────────────────────────────────

ARC_COLORS = {
    0: (0,   0,   0),    # black
    1: (0,   116, 217),  # blue
    2: (255, 65,  54),   # red
    3: (46,  204, 64),   # green
    4: (255, 220, 0),    # yellow
    5: (170, 170, 170),  # gray
    6: (240, 18,  190),  # magenta
    7: (255, 133, 27),   # orange
    8: (127, 219, 255),  # azure
    9: (135, 12,  37),   # maroon
}

ARC_COLOR_NAMES = {
    0: "black", 1: "blue",    2: "red",     3: "green",  4: "yellow",
    5: "gray",  6: "magenta", 7: "orange",  8: "azure",  9: "maroon",
}

# Compact single-char representation for text encoding
ARC_COLOR_CHARS = {
    0: ".", 1: "b", 2: "r", 3: "g", 4: "y",
    5: "W", 6: "m", 7: "o", 8: "a", 9: "n",
}
ARC_CHAR_COLORS = {v: k for k, v in ARC_COLOR_CHARS.items()}

# ── Grid type ──────────────────────────────────────────────────────────────────

Grid = list[list[int]]   # 2D list of ints 0-9


def grid_height(g: Grid) -> int:
    return len(g)

def grid_width(g: Grid) -> int:
    return len(g[0]) if g else 0

def grid_shape(g: Grid) -> tuple[int, int]:
    return (grid_height(g), grid_width(g))

def grid_copy(g: Grid) -> Grid:
    return [row[:] for row in g]

def grid_eq(a: Grid, b: Grid) -> bool:
    return a == b

def grid_to_text(g: Grid) -> str:
    """Compact text encoding. Each cell → color char, rows separated by newline."""
    return "\n".join("".join(ARC_COLOR_CHARS[c] for c in row) for row in g)

def text_to_grid(t: str) -> Grid:
    """Inverse of grid_to_text."""
    return [[ARC_CHAR_COLORS[c] for c in row] for row in t.strip().split("\n")]

def grid_to_json_str(g: Grid) -> str:
    return json.dumps(g)

def grid_from_json(j) -> Grid:
    if isinstance(j, str):
        j = json.loads(j)
    return [[int(c) for c in row] for row in j]

def grid_unique_colors(g: Grid) -> set[int]:
    return {c for row in g for c in row}

def grid_color_count(g: Grid, color: int) -> int:
    return sum(row.count(color) for row in g)

def grid_get(g: Grid, r: int, c: int, default: int = -1) -> int:
    if 0 <= r < grid_height(g) and 0 <= c < grid_width(g):
        return g[r][c]
    return default

def grid_set(g: Grid, r: int, c: int, v: int) -> Grid:
    g2 = grid_copy(g)
    g2[r][c] = v
    return g2

def grid_diff(a: Grid, b: Grid) -> list[tuple[int,int,int,int]]:
    """Returns list of (r, c, val_a, val_b) for differing cells."""
    diffs = []
    for r in range(max(grid_height(a), grid_height(b))):
        for c in range(max(grid_width(a), grid_width(b))):
            va = grid_get(a, r, c, -1)
            vb = grid_get(b, r, c, -1)
            if va != vb:
                diffs.append((r, c, va, vb))
    return diffs

def grid_similarity(a: Grid, b: Grid) -> float:  # LEGACY — use bra_resonance_score()
    """Pixel-level similarity [0.0, 1.0]."""
    if grid_shape(a) != grid_shape(b):
        return 0.0
    h, w = grid_shape(a)
    if h * w == 0:
        return 1.0
    matches = sum(1 for r in range(h) for c in range(w) if a[r][c] == b[r][c])
    return matches / (h * w)

def empty_grid(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill] * w for _ in range(h)]


# ── ARC task dataclasses ───────────────────────────────────────────────────────

@dataclass
class Pair:
    input:  Grid
    output: Optional[Grid] = None  # None for test pairs

    @classmethod
    def from_dict(cls, d: dict) -> "Pair":
        return cls(
            input  = grid_from_json(d["input"]),
            output = grid_from_json(d["output"]) if "output" in d else None,
        )

@dataclass
class ARCTask:
    task_id:  str
    train:    list[Pair]
    test:     list[Pair]
    source:   str = ""   # path or URL
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, task_id: str = "unknown", source: str = "") -> "ARCTask":
        return cls(
            task_id = task_id,
            train   = [Pair.from_dict(p) for p in d["train"]],
            test    = [Pair.from_dict(p) for p in d["test"]],
            source  = source,
            metadata = d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path_or_str: str, task_id: str = None) -> "ARCTask":
        p = Path(path_or_str)
        if p.exists():
            data = json.loads(p.read_text())
            tid  = task_id or p.stem
            return cls.from_dict(data, tid, str(p))
        else:
            # Treat as raw JSON string
            data = json.loads(path_or_str)
            return cls.from_dict(data, task_id or "inline", "inline")

    def summary(self) -> dict:
        return {
            "task_id":     self.task_id,
            "train_pairs": len(self.train),
            "test_pairs":  len(self.test),
            "input_shape": grid_shape(self.train[0].input) if self.train else None,
            "colors_seen": sorted(set(
                c for p in self.train
                for g in [p.input, p.output] if g
                for row in g for c in row
            )),
        }

@dataclass
class ARCPrediction:
    task_id:    str
    test_index: int
    predicted:  Optional[Grid]
    confidence: float = 0.0     # 0.0–1.0
    program:    str   = ""      # Python source that produced it
    reasoning:  str   = ""      # chain-of-thought
    train_score: float = 0.0    # fraction of training pairs correctly transformed
    correct:    Optional[bool] = None  # None if ground truth unknown
    vexel_root: str   = "0x0000000000000000"

    @property
    def ok(self) -> bool:
        return self.predicted is not None

# ── Dataset loader ─────────────────────────────────────────────────────────────

ARC_PUBLIC_BASE = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
ARC_SPLIT_DIRS  = {
    "training":   f"{ARC_PUBLIC_BASE}/training",
    "evaluation": f"{ARC_PUBLIC_BASE}/evaluation",
}

def load_task(task_id: str, split: str = "training",
              cache_dir: Path = None) -> ARCTask:
    """
    Load an ARC task by ID. Downloads from GitHub if not cached locally.
    task_id: e.g. "007bbfb7" (no .json extension)
    split: "training" or "evaluation"
    """
    cache_dir = cache_dir or Path.home() / ".arc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = cache_dir / split / f"{task_id}.json"
    if cached.exists():
        return ARCTask.from_json(str(cached), task_id)

    url  = f"{ARC_SPLIT_DIRS[split]}/{task_id}.json"
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "sovereign-arc/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode()
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(data)
        return ARCTask.from_dict(json.loads(data), task_id, url)
    except Exception as e:
        raise RuntimeError(f"Could not load ARC task {task_id!r}: {e}")

def load_task_from_file(path: str) -> ARCTask:
    return ARCTask.from_json(path)

def list_cached_tasks(split: str = "training",
                      cache_dir: Path = None) -> list[str]:
    cache_dir = cache_dir or Path.home() / ".arc_cache"
    d = cache_dir / split
    if not d.exists():
        return []
    return [p.stem for p in sorted(d.glob("*.json"))]


# ── Core Knowledge DSL Primitives ──────────────────────────────────────────────
# These are the basic operations that appear repeatedly across ARC tasks.
# The solver uses these as building blocks for program synthesis.

def rot90(g: Grid) -> Grid:
    """Rotate 90° clockwise."""
    h, w = grid_shape(g)
    return [[g[h - 1 - c][r] for c in range(h)] for r in range(w)]

def rot180(g: Grid) -> Grid:
    return rot90(rot90(g))

def rot270(g: Grid) -> Grid:
    return rot90(rot90(rot90(g)))

def reflect_h(g: Grid) -> Grid:
    """Reflect horizontally (flip left-right)."""
    return [row[::-1] for row in g]

def reflect_v(g: Grid) -> Grid:
    """Reflect vertically (flip top-bottom)."""
    return g[::-1]

def reflect_diag(g: Grid) -> Grid:
    """Reflect along main diagonal (transpose)."""
    h, w = grid_shape(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]

def reflect_anti(g: Grid) -> Grid:
    """Reflect along anti-diagonal."""
    return reflect_diag(rot180(g))

def crop(g: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Crop to rows [r0:r1], cols [c0:c1]."""
    return [row[c0:c1] for row in g[r0:r1]]

def pad(g: Grid, top: int=0, bottom: int=0, left: int=0,
        right: int=0, fill: int=0) -> Grid:
    h, w = grid_shape(g)
    result = empty_grid(h + top + bottom, w + left + right, fill)
    for r in range(h):
        for c in range(w):
            result[r + top][c + left] = g[r][c]
    return result

def tile(g: Grid, n_rows: int, n_cols: int) -> Grid:
    """Tile grid n_rows×n_cols times."""
    return [
        [g[r % grid_height(g)][c % grid_width(g)]
         for c in range(grid_width(g) * n_cols)]
        for r in range(grid_height(g) * n_rows)
    ]

def recolor(g: Grid, from_color: int, to_color: int) -> Grid:
    return [[to_color if c == from_color else c for c in row] for row in g]

def replace_colors(g: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(c, c) for c in row] for row in g]

def fill(g: Grid, r: int, c: int, new_color: int) -> Grid:
    """Flood-fill from (r, c) with new_color (4-connected)."""
    g2 = grid_copy(g)
    old = g2[r][c]
    if old == new_color:
        return g2
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if not (0 <= cr < grid_height(g2) and 0 <= cc < grid_width(g2)):
            continue
        if g2[cr][cc] != old:
            continue
        g2[cr][cc] = new_color
        stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
    return g2

def upscale(g: Grid, factor: int) -> Grid:
    """Scale each cell up by factor (pixelate)."""
    return [
        [g[r][c] for c in range(grid_width(g)) for _ in range(factor)]
        for r in range(grid_height(g)) for _ in range(factor)
    ]

def downscale(g: Grid, factor: int) -> Grid:
    """Take top-left cell of each factor×factor block."""
    h, w = grid_shape(g)
    return [
        [g[r][c] for c in range(0, w, factor)]
        for r in range(0, h, factor)
    ]

def overlay(base: Grid, mask: Grid, bg: int = 0) -> Grid:
    """Overlay mask onto base: non-bg cells in mask overwrite base."""
    h = max(grid_height(base), grid_height(mask))
    w = max(grid_width(base),  grid_width(mask))
    result = empty_grid(h, w, bg)
    for r in range(grid_height(base)):
        for c in range(grid_width(base)):
            result[r][c] = base[r][c]
    for r in range(grid_height(mask)):
        for c in range(grid_width(mask)):
            if mask[r][c] != bg:
                result[r][c] = mask[r][c]
    return result

def hstack(a: Grid, b: Grid) -> Grid:
    """Horizontally stack two grids (must have same height)."""
    return [ra + rb for ra, rb in zip(a, b)]

def vstack(a: Grid, b: Grid) -> Grid:
    """Vertically stack two grids (must have same width)."""
    return a + b

def extract_objects(g: Grid, bg: int = 0) -> list[dict]:
    """
    Find all connected components (objects) in the grid.
    Returns list of dicts: {color, cells: [(r,c)...], bbox: (r0,c0,r1,c1), grid}
    """
    h, w = grid_shape(g)
    visited = [[False]*w for _ in range(h)]
    objects = []

    for r in range(h):
        for c in range(w):
            color = g[r][c]
            if color == bg or visited[r][c]:
                continue
            # BFS
            cells = []
            queue = [(r, c)]
            visited[r][c] = True
            while queue:
                cr, cc = queue.pop(0)
                cells.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (0<=nr<h and 0<=nc<w and
                            not visited[nr][nc] and g[nr][nc] == color):
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            r1 = max(r for r, _ in cells) + 1
            c1 = max(c for _, c in cells) + 1
            # Extract subgrid
            sub = empty_grid(r1-r0, c1-c0, bg)
            for cr, cc in cells:
                sub[cr-r0][cc-c0] = color

            objects.append({
                "color":  color,
                "cells":  cells,
                "size":   len(cells),
                "bbox":   (r0, c0, r1, c1),
                "grid":   sub,
                "height": r1-r0,
                "width":  c1-c0,
            })

    return sorted(objects, key=lambda o: (-o["size"], o["bbox"][0], o["bbox"][1]))

def detect_symmetry(g: Grid) -> dict[str, bool]:
    """Detect common symmetries in a grid."""
    return {
        "h_sym":    g == reflect_h(g),
        "v_sym":    g == reflect_v(g),
        "rot90_sym": g == rot90(g),
        "rot180_sym": g == rot180(g),
        "diag_sym": grid_shape(g)[0] == grid_shape(g)[1] and g == reflect_diag(g),
    }

def count_colors(g: Grid) -> dict[int, int]:
    counts = {}
    for row in g:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items()))

def most_common_color(g: Grid, exclude: set = None) -> int:
    exclude = exclude or set()
    counts = {k: v for k, v in count_colors(g).items() if k not in exclude}
    return max(counts, key=counts.get) if counts else 0

def least_common_color(g: Grid, exclude: set = None) -> int:
    exclude = exclude or set()
    counts = {k: v for k, v in count_colors(g).items() if k not in exclude}
    return min(counts, key=counts.get) if counts else 0

def normalize_colors(g: Grid) -> Grid:
    """
    Remap non-background colors to lowest values starting from 1.
    Preserves 0. Useful for color-invariant matching and normalization.
    Example: [[2,0,5],[5,2,0]] → [[1,0,2],[2,1,0]]
    """
    present = sorted(set(c for row in g for c in row if c != 0))
    if not present:
        return grid_copy(g)
    mapping = {c: i+1 for i, c in enumerate(present)}
    mapping[0] = 0
    return replace_colors(g, mapping)


def background_color(g: Grid) -> int:
    """Most common color is usually background."""
    return most_common_color(g)

def crop_to_content(g: Grid, bg: int = None) -> Grid:
    """Crop grid to remove uniform background border."""
    if bg is None:
        bg = background_color(g)
    h, w = grid_shape(g)
    rows = [r for r in range(h) if any(g[r][c] != bg for c in range(w))]
    cols = [c for c in range(w) if any(g[r][c] != bg for r in range(h))]
    if not rows or not cols:
        return g
    return crop(g, rows[0], cols[0], rows[-1]+1, cols[-1]+1)

# ── ARC task description (text) ────────────────────────────────────────────────

def describe_grid(g: Grid, name: str = "Grid") -> str:
    h, w = grid_shape(g)
    colors = count_colors(g)
    color_desc = ", ".join(
        f"{ARC_COLOR_NAMES.get(c,'c'+str(c))}×{n}"
        for c, n in sorted(colors.items())
    )
    symm = detect_symmetry(g)
    symm_desc = ", ".join(k for k, v in symm.items() if v) or "none"
    text = grid_to_text(g)
    return (
        f"{name}: {h}×{w}  colors=[{color_desc}]  symmetry=[{symm_desc}]\n"
        f"{text}"
    )

def describe_pair(pair: Pair, index: int) -> str:
    lines = [f"=== Training Pair {index+1} ==="]
    lines.append(describe_grid(pair.input, "INPUT"))
    if pair.output:
        lines.append(describe_grid(pair.output, "OUTPUT"))
        diffs = grid_diff(pair.input, pair.output)
        lines.append(f"Changes: {len(diffs)} cells")
        ih, iw = grid_shape(pair.input)
        oh, ow = grid_shape(pair.output)
        if (ih, iw) != (oh, ow):
            lines.append(f"Shape change: {ih}×{iw} → {oh}×{ow}")
    return "\n".join(lines)

def describe_task(task: ARCTask) -> str:
    lines = [f"=== ARC Task {task.task_id} ===",
             f"Training pairs: {len(task.train)}  Test inputs: {len(task.test)}"]
    for i, p in enumerate(task.train):
        lines.append(describe_pair(p, i))
    for i, p in enumerate(task.test):
        lines.append(f"=== Test Input {i+1} ===")
        lines.append(describe_grid(p.input, "INPUT"))
    return "\n".join(lines)

# ── Scoring ────────────────────────────────────────────────────────────────────

def score_prediction(pred: Grid, truth: Grid) -> int:
    """
    Sovereign scoring: integer BRA resonance (0/1/2).
    2 = exact match (all EigenCharge components identical)
    1 = near resonance (charge proximity within BRA thresholds)
    0 = no resonance
    Replaces float pixel similarity — deterministic, no rounding.
    """
    if pred == truth:
        return 2          # exact
    if grid_shape(pred) != grid_shape(truth):
        return 0          # shape mismatch = structural miss
    try:
        from arc_bra import grid_eigen_charge, bra_resonance_score
        return bra_resonance_score(grid_eigen_charge(pred),
                                   grid_eigen_charge(truth))
    except ImportError:
        # Fallback: exact-only if BRA unavailable
        return 2 if pred == truth else 0

def score_task(predictions: list[Optional[Grid]],
               task: ARCTask) -> dict:
    """
    Score a list of predictions against task test outputs.
    ARC scoring: a task is "solved" only if ALL test outputs are exactly correct.
    """
    results = []
    for i, (pred, pair) in enumerate(zip(predictions, task.test)):
        if pair.output is None:
            results.append({"test_index": i, "correct": None, "similarity": None})
        else:
            exact      = pred == pair.output if pred else False
            resonance  = score_prediction(pred, pair.output) if pred else 0
            results.append({"test_index": i, "correct": exact,
                            "resonance": resonance, "similarity": resonance / 2.0})
    all_correct = all(r["correct"] for r in results if r["correct"] is not None)
    return {
        "task_id":      task.task_id,
        "solved":       all_correct,
        "results":      results,
        "bra_resonance": (
            sum(r["resonance"] for r in results if r.get("resonance") is not None) /
            max(1, sum(1 for r in results if r["similarity"] is not None))
        ),
    }
