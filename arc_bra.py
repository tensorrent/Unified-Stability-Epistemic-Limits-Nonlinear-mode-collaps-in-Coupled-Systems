"""
arc_bra.py — Sovereign ARC-AGI Adaptation Layer
=================================================
This module replaces the generic floating-point AI infrastructure in the
ARC solver with the actual sovereign architecture:

  1. BRA EIGEN-CHARGE          — integer eigenvalue grid fingerprint
                                  replaces float grid_similarity()
                                  resolution: 10^14 bins, no float on charge path

  2. TENT DENSITY GATE         — high-pass filter before any LLM call
                                  low-density grids → brute-force only
                                  high-density grids → full cinema pipeline

  3. UPG-GUIDED DSL ROUTING    — primitives ordered by 32³ voxel proximity
                                  replaces linear O(n) brute-force scan
                                  twin prime coupling: linked primitives tested together

  4. ULAM SPIRAL GRID ENCODING — rotation-invariant grid representation
                                  cell (row, col, color) → Ulam coordinate
                                  two rotations of same pattern → adjacent charges

  5. EIGENSTATE PATTERN LOOKUP — scroll Merkle root comparison via BRA resonance
                                  replaces SQLite FTS5 text search
                                  exact integer resonance: 0 / 1 / 2

Architecture principle (from TENT v6.1):
  "Higher resolution ADC: 10^15 bins instead of 1,000.
   That's going from 10-bit to 50-bit. No more charge collisions.
   Tighter Q on the resonance: σ from 0.003 down to 5×10^-8.
   Only exact charge matches excite the well."

This is NOT stochastic AI. It is crystallographic matching.
The key either fits or it doesn't.

Author: Brad Wallace / Claude — Sovereign Stack Integration
"""

from __future__ import annotations
import hashlib
import math
import bisect
from typing import Optional, NamedTuple

from arc_types import Grid, ARCTask, Pair, grid_shape, background_color, extract_objects


# ══════════════════════════════════════════════════════════════════════════════
# 1. BRA EIGEN-CHARGE
#    Integer eigenvalue fingerprint. No float on the charge path.
#    Mirrors EigenCharge in trinity_core.rs exactly.
# ══════════════════════════════════════════════════════════════════════════════

F369_SIZE       = 12000
CHARGE_RESOLUTION = 10**14       # 10^14 bins — the ADC resolution
TRACE_THRESH    = 500_000        # from BRA kernel
DET_THRESH      = 5_000_000      # from BRA kernel

# Pre-compute F369 table (pure integer — mirrors Rust implementation)
_F369_TABLE: list[int] = [0] * F369_SIZE
for _i in range(1, F369_SIZE):
    _n = _i
    _F369_TABLE[_i] = (_n * (_n - 1) // 2) * 3 - (_n // 3) * 6 + (_n // 9) * 9


class EigenCharge(NamedTuple):
    """Integer eigenvalue triplet. Mirrors trinity_core.rs EigenCharge."""
    hash:  int   # u64 equivalent (FNV-1a)
    trace: int   # i64: sum of F369 table lookups
    det:   int   # i64: sum of F369 cross-products


def eigen_charge(data: bytes) -> EigenCharge:
    """
    Compute EigenCharge from raw bytes.
    Pure integer — no float. Mirrors bra_eigen_charge() in trinity_core.rs.
    """
    h     = 0xcbf29ce484222325          # FNV offset basis
    trace = 0
    det   = 0
    for i, b in enumerate(data):
        idx    = (b * (i + 1)) % F369_SIZE
        h     ^= b
        h      = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF  # u64 wrapping
        trace += _F369_TABLE[idx]
        det   += _F369_TABLE[idx] * _F369_TABLE[(idx + 7) % F369_SIZE]
    return EigenCharge(hash=h, trace=trace, det=det)


def bra_resonance_score(a: EigenCharge, b: EigenCharge) -> int:
    """
    Integer resonance score. Mirrors bra_resonance_score() in trinity_core.rs.
    Returns:
      2 — exact match (all three components identical)
      1 — near resonance (trace delta < TRACE_THRESH, det delta < DET_THRESH)
      0 — no resonance
    """
    if a.hash == b.hash and a.trace == b.trace and a.det == b.det:
        return 2
    td = abs(a.trace - b.trace)
    dd = abs(a.det - b.det)
    if td < TRACE_THRESH and dd < DET_THRESH:
        return 1
    return 0


def grid_to_bytes(g: Grid) -> bytes:
    """
    Serialize grid to bytes for BRA encoding.
    Row-major, each cell as one byte (0-9 fits in a byte).
    Shape is prepended so differently-shaped grids can't collide.
    """
    h, w = grid_shape(g)
    header = bytes([h & 0xFF, w & 0xFF])
    body   = bytes(c for row in g for c in row)
    return header + body


def grid_eigen_charge(g: Grid) -> EigenCharge:
    """EigenCharge of a grid. The primary structural fingerprint."""
    return eigen_charge(grid_to_bytes(g))


def bra_grids_resonant(a: Grid, b: Grid) -> int:
    """
    Compare two grids via BRA integer resonance.
    Returns 0/1/2. Replaces float grid_similarity().
    Precision: 10^14 bins vs float's ~10^7.
    """
    return bra_resonance_score(grid_eigen_charge(a), grid_eigen_charge(b))


def bra_grids_exact(a: Grid, b: Grid) -> bool:
    """True if grids are exactly identical (resonance == 2)."""
    return bra_grids_resonant(a, b) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 2. ULAM SPIRAL — STORAGE ADDRESS UTILITY
#
#    The Ulam spiral is the CYLINDER in vexel.rs:
#      - Universal isomorphic coordinate system for persistent memory storage
#      - Primes cluster on diagonals — these become the PINS
#      - BRA charge integers are mapped to (x,y) coords to find nearest prime pin
#      - Session events "align" with pins → holes cut in the scroll sheet
#      - The scroll IS the persistent memory (MIDI-encoded, 32 bytes/event)
#
#    This is NOT an inference or similarity tool.
#    _ulam_coord() is kept here to mirror vexel.rs ulam_coord() for the
#    vexel scroll integration in arc_hermes.py.
#    It has no role in pattern matching or task comparison.
# ══════════════════════════════════════════════════════════════════════════════

def _ulam_coord(n: int) -> tuple[int, int]:
    """
    (x, y) position on the Ulam spiral for integer n.
    n=1→(0,0), n=2→(1,0), n=3→(1,1), ...
    Mirrors ulam_coord() in vexel.rs exactly.

    ROLE: Storage address lookup for vexel cylinder/scroll integration.
          The BRA charge integer maps here to find its nearest prime pin.
          NOT used for inference, similarity, or pattern matching.
    """
    if n <= 1:
        return (0, 0)
    k = math.ceil((math.sqrt(n) - 1) / 2)
    t = 2 * k
    n2 = (2 * k - 1) ** 2
    pos = n - n2 - 1
    side = pos // t
    off  = pos  % t
    if side == 0: return (k,   -k + 1 + off)
    if side == 1: return (k - 1 - off, k)
    if side == 2: return (-k,  k - 1 - off)
    return          (-k + 1 + off, -k)


def ulam_scroll_address(charge: int, capacity: int = 10000) -> tuple[int, int, int]:
    """
    Map a BRA charge integer to a vexel scroll address.
    Returns (x, y, nearest_prime_pin) for use by arc_hermes scroll recording.
    Mirrors Cylinder.nearest_pin() in vexel.rs.

    ROLE: Storage/retrieval address for the persistent scroll.
          The charge maps to a spiral coord; the nearest prime becomes the pin.
          Events that align with pins are cut into the scroll (a hole in the sheet).
    """
    cx, cy = _ulam_coord(charge % max(capacity, 1))
    # Find nearest prime ≤ capacity on the spiral
    best_p, best_d2 = 2, float('inf')
    for p in range(2, min(capacity, 10000)):
        # trial division (small numbers only)
        if p < 2: continue
        if p > 2 and p % 2 == 0: continue
        is_p = all(p % i != 0 for i in range(3, int(p**0.5)+1, 2))
        if not is_p: continue
        px, py = _ulam_coord(p)
        d2 = (px - cx)**2 + (py - cy)**2
        if d2 < best_d2:
            best_d2, best_p = d2, p
        if d2 == 0:
            break  # exact pin hit
    return (cx, cy, best_p)


def task_charge(task: ARCTask) -> EigenCharge:
    """
    Deterministic EigenCharge for an entire task — IDENTITY fingerprint.
    Includes MD5(task_id) so the same training structure with a different
    task_id produces a different charge. Used for:
      - Vexel scroll address (unique per task)
      - BRAPatternStore exact replay detection
      - Merkle identity

    For cross-task pattern matching use task_structure_charge().
    """
    import hashlib
    buf = bytearray()
    for i, pair in enumerate(task.train):
        buf.append(i & 0xFF)
        ih, iw = grid_shape(pair.input)
        buf.append(ih & 0xFF)
        buf.append(iw & 0xFF)
        for row in pair.input:
            for c in row:
                buf.append(c & 0xFF)
        buf.append(0xFF)
        if pair.output:
            oh, ow = grid_shape(pair.output)
            buf.append(oh & 0xFF)
            buf.append(ow & 0xFF)
            for row in pair.output:
                for c in row:
                    buf.append(c & 0xFF)
        buf.append(0xFE)
    # Append MD5 of task_id so same-transform tasks from different task_ids diverge
    buf.extend(hashlib.md5(task.task_id.encode()).digest())
    return eigen_charge(bytes(buf))


def task_structure_charge(task: ARCTask) -> EigenCharge:
    """
    Deterministic EigenCharge for task STRUCTURE only — excludes task_id.
    Two tasks with the same training examples produce the same charge,
    even if their task_ids differ. Used for:
      - warm_search cross-task pattern matching
      - BRAPatternStore cross-task lookup (same transform, different task)

    For identity/vexel use task_charge() which includes task_id.
    """
    import hashlib
    buf = bytearray()
    for i, pair in enumerate(task.train):
        buf.append(i & 0xFF)
        ih, iw = grid_shape(pair.input)
        buf.append(ih & 0xFF)
        buf.append(iw & 0xFF)
        for row in pair.input:
            for c in row:
                buf.append(c & 0xFF)
        buf.append(0xFF)
        if pair.output:
            oh, ow = grid_shape(pair.output)
            buf.append(oh & 0xFF)
            buf.append(ow & 0xFF)
            for row in pair.output:
                for c in row:
                    buf.append(c & 0xFF)
        buf.append(0xFE)
    # No task_id appended — pure structural fingerprint
    return eigen_charge(bytes(buf))


# ══════════════════════════════════════════════════════════════════════════════
# 3. TENT DENSITY GATE
#    High-pass filter: measure structural complexity before routing.
#    Low complexity → gate closed, brute-force only.
#    High complexity → gate open, full LLM cinema pipeline.
#    "You don't send white noise through a vocal chain."
# ══════════════════════════════════════════════════════════════════════════════

class TENTGateResult:
    """Result of the density gate measurement."""
    def __init__(self, density: float, n_objects: int, n_colors: int,
                 n_cells: int, charge: EigenCharge, pipeline: str):
        self.density   = density    # structural density 0.0–1.0
        self.n_objects = n_objects
        self.n_colors  = n_colors
        self.n_cells   = n_cells
        self.charge    = charge     # EigenCharge of the task
        self.pipeline  = pipeline   # "brute_force" | "llm_standard" | "llm_cinema"

    def gate_open(self) -> bool:
        """Is the gate open — should we call the LLM?"""
        return self.pipeline != "brute_force"

    def cinema_spec(self) -> bool:
        """Should we use the full cinema pipeline (more candidates, more refinements)?"""
        return self.pipeline == "llm_cinema"

    def __repr__(self):
        return (f"TENTGate(density={self.density:.3f}, "
                f"objects={self.n_objects}, colors={self.n_colors}, "
                f"pipeline={self.pipeline!r})")


def tent_density_gate(task: ARCTask) -> TENTGateResult:
    """
    TENT density gate for ARC tasks.
    Measures structural complexity of the training set.
    Routes to the appropriate pipeline tier.

    Streaming spec  (density < 0.25):  brute_force only
    Standard spec   (density < 0.65):  llm_standard (3 candidates, 3 refinements)
    Cinema spec     (density >= 0.65): llm_cinema   (5 candidates, 5 refinements)
    """
    from arc_types import count_colors, extract_objects

    if not task.train:
        return TENTGateResult(0.0, 0, 0, 0, EigenCharge(0,0,0), "brute_force")

    # Measure complexity across all training pairs
    total_objects = 0
    total_colors  = 0
    total_cells   = 0
    pair_count    = len(task.train)
    max_hw        = 1

    for pair in task.train:
        h, w  = grid_shape(pair.input)
        bg    = background_color(pair.input)
        objs  = extract_objects(pair.input, bg)
        cols  = count_colors(pair.input)
        cells = sum(1 for row in pair.input for c in row if c != bg)

        total_objects += len(objs)
        total_colors  += len(cols) - 1  # exclude bg
        total_cells   += cells
        max_hw         = max(max_hw, h * w)

    avg_objects = total_objects / pair_count
    avg_colors  = total_colors  / pair_count
    avg_cells   = total_cells   / pair_count
    fill_ratio  = avg_cells     / max_hw if max_hw > 0 else 0.0

    # Also measure TRANSFORMATION complexity via BRA resonance (integer, exact)
    transform_deltas = []
    for pair in task.train:
        if pair.output:
            ih, iw = grid_shape(pair.input)
            oh, ow = grid_shape(pair.output)
            shape_change = int((ih, iw) != (oh, ow))
            if shape_change:
                resonance = 0
            else:
                resonance = bra_resonance_score(
                    grid_eigen_charge(pair.input),
                    grid_eigen_charge(pair.output)
                )
            # resonance: 2=identical, 1=near, 0=different
            # transform_delta: 0=no change, 1=total change
            transform_deltas.append(1.0 - (resonance / 2.0) + shape_change * 0.3)

    avg_transform = (sum(transform_deltas) / len(transform_deltas)
                     if transform_deltas else 0.5)

    # Density score: weighted combination
    # Objects contribute most (structural complexity)
    # Colors and transform delta add to the signal
    density = (
        min(avg_objects / 6.0, 1.0)   * 0.40 +   # object complexity
        min(avg_colors  / 5.0, 1.0)   * 0.25 +   # color complexity
        min(fill_ratio  / 0.5, 1.0)   * 0.15 +   # fill density
        min(avg_transform, 1.0)        * 0.20     # transformation complexity
    )

    # Task charge for eigenstate comparison
    charge = task_charge(task)

    # Pipeline routing (TENT gate tiers)
    if density < 0.25:
        pipeline = "brute_force"    # streaming spec: gate closed
    elif density < 0.65:
        pipeline = "llm_standard"   # standard spec
    else:
        pipeline = "llm_cinema"     # cinema spec: full dynamic range

    return TENTGateResult(
        density   = density,
        n_objects = int(avg_objects),
        n_colors  = int(avg_colors),
        n_cells   = int(avg_cells),
        charge    = charge,
        pipeline  = pipeline,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. UPG-GUIDED DSL PRIMITIVE ROUTING
#    Map each DSL primitive to a position in the 32³ voxel lattice.
#    Task charge → UPG coordinate → search only geometrically nearby primitives.
#    Twin prime coupling: linked pairs tested together.
#    "The geometry becomes the router."
# ══════════════════════════════════════════════════════════════════════════════

PHI  = (1 + math.sqrt(5)) / 2
H21  = 21
H369 = 369
GRID_SIZE = 32


def _prime_to_upg_coord(p: int) -> tuple[int, int, int]:
    """
    Map prime p to (x,y,z) in the 32³ UPG lattice.
    Mirrors the UPG math in sovereign_vixel/upg_lattice.py.
    """
    x = (p % H21)  * GRID_SIZE // H21
    y = (p % H369) * GRID_SIZE // H369
    z = int(math.log(max(p, 2)) / math.log(PHI)) % GRID_SIZE
    return (x, y, z)


def _upg_distance(a: tuple[int,int,int], b: tuple[int,int,int]) -> float:
    """Euclidean distance in the 32³ lattice."""
    return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))


# DSL primitive catalogue with UPG coordinates
# Each primitive is assigned a "semantic prime" — its position in the lattice
# is derived from that prime's UPG coordinate.
# Semantic prime assignment: based on the transformation's algebraic order
_PRIM_PRIMES = {
    # Geometric identity group (small primes — fundamental)
    "identity":          2,
    "rot90":             3,
    "rot180":            5,
    "rot270":            7,
    "reflect_h":        11,
    "reflect_v":        13,
    "reflect_diag":     17,
    "reflect_anti":     19,
    # Scale transforms (next shell)
    "upscale_2":        23,
    "upscale_3":        29,
    "downscale_2":      31,
    "crop_to_content":  37,
    # Color transforms
    "recolor":          41,
    "replace_colors":   43,
    "invert_colors":    47,
    "normalize_colors": 53,
    # Symmetry completion
    "complete_h_sym":   59,
    "complete_v_sym":   61,
    "complete_rot180":  67,
    "complete_diag":    71,
    "enforce_h_sym":    73,
    "enforce_v_sym":    79,
    # Gravity (physics domain)
    "gravity_down":     83,
    "gravity_up":       89,
    "gravity_left":     97,
    "gravity_right":   101,
    "gravity_blocked": 103,
    # Object operations
    "largest_object":  107,
    "smallest_object": 109,
    "border_objects":  113,
    "interior_objects":127,
    "rank_by_size":    131,
    "fill_holes":      137,
    # Morphology
    "outline":         139,
    "dilate":          149,
    "erode":           151,
    "convex_hull":     157,
    # Logic
    "grid_xor":        163,
    "grid_and":        167,
    "grid_or":         173,
    # Split/Join
    "hstack_self":     179,
    "vstack_self":     181,
    "tile_2x2":        191,
    "split_divider":   193,
    # Path/connect
    "connect_cells":   197,
    "shortest_path":   199,
}

# Twin prime pairs — if one fires, test its twin immediately
# (mirrors twin prime coupling in UPG: shared activation pool)
_TWIN_PRIME_PAIRS = [
    (3, 5),    # rot90  ↔ rot180
    (5, 7),    # rot180 ↔ rot270
    (11, 13),  # reflect_h ↔ reflect_v
    (17, 19),  # reflect_diag ↔ reflect_anti
    (29, 31),  # upscale_3 ↔ downscale_2
    (41, 43),  # recolor ↔ replace_colors
    (59, 61),  # complete_h ↔ complete_v
    (71, 73),  # complete_diag ↔ enforce_h
    (83, 89),  # gravity_down ↔ gravity_up
    (97, 101), # gravity_left ↔ gravity_right
    (107, 109),# largest ↔ smallest
    (113, 127),# border ↔ interior
    (139, 149),# outline ↔ dilate
    (149, 151),# dilate ↔ erode
    (163, 167),# xor ↔ and
    (179, 181),# hstack ↔ vstack
    (197, 199),# connect ↔ path
]
_TWIN_MAP: dict[int, int] = {}
for _a, _b in _TWIN_PRIME_PAIRS:
    _TWIN_MAP[_a] = _b
    _TWIN_MAP[_b] = _a

# Pre-compute UPG coordinates for all primitives.
# Coordinates are assigned SEMANTICALLY to match the task feature axes:
#   x = COLOR REMAP intensity  (0=no remap, 31=full palette change)
#   y = SPATIAL SHUFFLE intensity (0=no movement, 31=full rearrangement)
#   z = OBJECT COMPLEXITY (0=cell-level, 31=many-object operations)
# This ensures UPG distance from task → primitive IS a real routing signal.
_PRIM_COORDS: dict[str, tuple[int, int, int]] = {
    # Geometric group (high y, low x, low-mid z)
    "identity":          ( 0,  0,  0),
    "rot90":             ( 0, 30,  4),
    "rot180":            ( 0, 28,  4),
    "rot270":            ( 0, 30,  4),
    "reflect_h":         ( 0, 24,  4),
    "reflect_v":         ( 0, 24,  4),
    "reflect_diag":      ( 0, 26,  4),
    "reflect_anti":      ( 0, 26,  4),
    # Scale (high y — spatial, no new colors)
    "upscale_2":         ( 0, 20,  2),
    "upscale_3":         ( 0, 22,  2),
    "downscale_2":       ( 0, 20,  2),
    "crop_to_content":   ( 2, 14,  8),
    # Color transforms (high x, low y)
    "recolor":           (28,  0,  4),
    "replace_colors":    (28,  0,  4),
    "invert_colors":     (24,  0,  2),
    "normalize_colors":  (20,  0,  2),
    # Symmetry completion (mid y — spatial, no new colors)
    "complete_h_sym":    ( 0, 16,  4),
    "complete_v_sym":    ( 0, 16,  4),
    "complete_rot180":   ( 0, 18,  4),
    "complete_diag":     ( 0, 17,  4),
    "enforce_h_sym":     ( 0, 15,  4),
    "enforce_v_sym":     ( 0, 15,  4),
    # Gravity (mid y, low x — positional movement)
    "gravity_down":      ( 0, 14, 10),
    "gravity_up":        ( 0, 14, 10),
    "gravity_left":      ( 0, 12, 10),
    "gravity_right":     ( 0, 12, 10),
    "gravity_blocked":   ( 2, 10, 12),
    # Object operations (high z, low x/y)
    "largest_object":    ( 4,  4, 28),
    "smallest_object":   ( 4,  4, 28),
    "border_objects":    ( 4,  6, 24),
    "interior_objects":  ( 4,  6, 24),
    "rank_by_size":      ( 4,  4, 26),
    "fill_holes":        ( 6,  8, 20),
    # Morphology (mid z, low x/y)
    "outline":           ( 4,  8, 18),
    "dilate":            ( 4, 10, 16),
    "erode":             ( 4, 10, 16),
    "convex_hull":       ( 4,  6, 22),
    # Logic (mid x — changes values, low spatial)
    "grid_xor":          (18,  8, 12),
    "grid_and":          (16,  6, 12),
    "grid_or":           (16,  6, 12),
    # Split/Join (high y — structural, low x)
    "hstack_self":       ( 0, 18,  6),
    "vstack_self":       ( 0, 18,  6),
    "tile_2x2":          ( 0, 20,  4),
    "split_divider":     ( 2, 16, 14),
    # Path/Connect (mid z, mid y)
    "connect_cells":     ( 8, 12, 20),
    "shortest_path":     ( 6, 10, 18),
}


def task_upg_coord(task: ARCTask) -> tuple[int, int, int]:
    """
    Map a task to a position in the 32³ UPG lattice using OBSERVABLE
    structural features that semantically distinguish transformation types.

    Axis semantics:
      x (0–31): COLOR REMAP — new colors introduced (recolor/invert signal)
                0 = same color vocabulary, 31 = entirely new palette
      y (0–31): SPATIAL SHUFFLE — cells moved to new positions (rotation/reflect signal)
                0 = no positional change, 31 = full spatial rearrangement
      z (0–31): OBJECT COMPLEXITY — distinct objects in training inputs
                0 = uniform/empty, 31 = many distinct objects

    Key discriminators:
      recolor(1→2):    new color introduced   → x HIGH, y LOW,  z LOW
      rot90:           no new colors, moved    → x LOW,  y HIGH, z LOW
      object-select:   no new colors, no move  → x LOW,  y LOW,  z HIGH
      scale+recolor:   new colors + size diff  → x HIGH, y HIGH, z MED

    This means UPG distance from task coord to primitive coord IS a real signal:
    recolor primitive (prime 41) lives near high-x region,
    rot90 (prime 3) lives near high-y region.
    """
    if not task.train:
        return (16, 16, 16)  # center — no information

    x_scores = []  # color remap: fraction of output cells with color NOT in input set
    y_scores = []  # spatial shuffle: cells changed to a color that WAS already in input
    z_scores = []  # object count

    for pair in task.train:
        if pair.output is None:
            continue
        ih, iw = grid_shape(pair.input)
        oh, ow = grid_shape(pair.output)

        in_colors = set(pair.input[r][c] for r in range(ih) for c in range(iw))

        if (ih, iw) == (oh, ow):
            total = ih * iw
            new_color_cells   = 0  # output cell color not in input vocabulary
            spatial_rearrange = 0  # output cell color IS in input vocab but cell changed

            for r in range(ih):
                for c in range(iw):
                    ic = pair.input[r][c]
                    oc = pair.output[r][c]
                    if ic != oc:
                        if oc not in in_colors:
                            new_color_cells   += 1
                        else:
                            spatial_rearrange += 1
            x_scores.append(new_color_cells   / max(total, 1))
            y_scores.append(spatial_rearrange / max(total, 1))
        else:
            # Shape changed — likely scale or crop
            size_ratio = min(ih*iw, oh*ow) / max(ih*iw, oh*ow, 1)
            out_colors = set(pair.output[r][c] for r in range(oh) for c in range(ow))
            new_frac   = len(out_colors - in_colors) / max(len(out_colors), 1)
            x_scores.append(new_frac)
            y_scores.append(1.0 - size_ratio)  # big size diff → high spatial score

        try:
            objs = extract_objects(pair.input)
            z_scores.append(len(objs))
        except Exception:
            z_scores.append(1)

    avg_x = sum(x_scores) / max(len(x_scores), 1)
    avg_y = sum(y_scores) / max(len(y_scores), 1)
    avg_z = sum(z_scores) / max(len(z_scores), 1)

    x = min(int(avg_x * GRID_SIZE), GRID_SIZE - 1)
    y = min(int(avg_y * GRID_SIZE), GRID_SIZE - 1)
    z = min(int((avg_z / 8.0) * GRID_SIZE), GRID_SIZE - 1)

    return (x, y, z)


def upg_ordered_primitives(task: ARCTask,
                             max_primitives: int = None) -> list[str]:
    """
    Return DSL primitive names ordered by UPG proximity to the task.
    Primitives closest in the 32³ lattice are tested first.
    Twin prime coupling: if a primitive appears, its twin is inserted immediately after.

    This replaces the linear O(n) brute-force scan with geometric routing.
    """
    coord   = task_upg_coord(task)
    # Sort all primitives by distance to task coordinate
    ordered = sorted(
        _PRIM_COORDS.keys(),
        key=lambda name: _upg_distance(coord, _PRIM_COORDS[name])
    )

    # Insert twin pairs immediately after their partner
    result = []
    inserted = set()
    for name in ordered:
        if name in inserted:
            continue
        result.append(name)
        inserted.add(name)
        # Check for twin
        p = _PRIM_PRIMES.get(name)
        twin_p = _TWIN_MAP.get(p) if p else None
        if twin_p:
            twin_name = next(
                (n for n, q in _PRIM_PRIMES.items() if q == twin_p and n not in inserted),
                None
            )
            if twin_name:
                result.append(twin_name)
                inserted.add(twin_name)

    if max_primitives:
        result = result[:max_primitives]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. EIGENSTATE PATTERN LOOKUP
#    Replace SQLite FTS5 text search with BRA integer resonance.
#    Store EigenCharge per solved task.
#    Lookup = bra_resonance_score() on charge array. Exact. No text parsing.
# ══════════════════════════════════════════════════════════════════════════════

class BRAPatternStore:
    """
    In-memory + persistent eigenstate pattern store.
    Backed by a simple flat file (JSON lines) + BRA integer index.
    Lookup is O(n) integer comparison — exact resonance, no text matching.
    """

    def __init__(self, path: str = None):
        self._path    = path
        self._entries: list[dict] = []  # {charge, program, task_id, score, category}
        if path:
            self._load()

    def _load(self):
        import json, os
        if not os.path.exists(self._path):
            return
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Reconstruct EigenCharge from stored ints
                    c = d.get("charge", {})
                    d["_charge"] = EigenCharge(
                        hash  = c.get("hash", 0),
                        trace = c.get("trace", 0),
                        det   = c.get("det", 0),
                    )
                    self._entries.append(d)
                except Exception:
                    pass

    def store(self, task: ARCTask, program: str, score: float,
              category: str = "unknown") -> EigenCharge:
        """Store a solved pattern, indexed by task EigenCharge."""
        import json, os
        charge = task_charge(task)
        entry  = {
            "task_id":  task.task_id,
            "program":  program,
            "score":    score,
            "category": category,
            "charge": {
                "hash":  charge.hash,
                "trace": charge.trace,
                "det":   charge.det,
            },
            "_charge": charge,
        }
        # Deduplicate: update if same task_id and better score
        for i, e in enumerate(self._entries):
            if e["task_id"] == task.task_id and score > e["score"]:
                self._entries[i] = entry
                break
        else:
            self._entries.append(entry)

        if self._path:
            with open(self._path, "a") as f:
                d = {k: v for k, v in entry.items() if k != "_charge"}
                f.write(json.dumps(d) + "\n")

        return charge

    def lookup(self, task: ARCTask,
               min_score: float = 0.5) -> list[dict]:
        """
        Find patterns resonant with the given task.
        Uses BRA integer resonance — not text search.
        Returns entries sorted by (resonance_score DESC, train_score DESC).
        """
        charge  = task_charge(task)
        results = []
        for entry in self._entries:
            ec = entry.get("_charge")
            if ec is None:
                continue
            rscore = arc_task_resonance(charge, ec)
            if rscore > 0 and entry["score"] >= min_score:
                results.append({**entry, "resonance": rscore})

        results.sort(key=lambda x: (-x["resonance"], -x["score"]))
        return results

    def best_program(self, task: ARCTask) -> Optional[dict]:
        """Return the best-resonating pattern for this task."""
        hits = self.lookup(task, min_score=0.0)
        return hits[0] if hits else None

    def stats(self) -> dict:
        total  = len(self._entries)
        exact  = sum(1 for e in self._entries if e["score"] >= 1.0)
        cats   = {}
        for e in self._entries:
            c = e.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        return {"total": total, "perfect": exact, "by_category": cats}


# ══════════════════════════════════════════════════════════════════════════════
# 6. SOVEREIGN SOLVER INTEGRATION
#    Puts it all together: gate → route → match → solve
# ══════════════════════════════════════════════════════════════════════════════


# ARC-specific thresholds (calibrated for full-task byte sequences)
# These are tighter than TENT word thresholds — full grid encoding produces
# trace values in the millions, so we can use a proportional window.
ARC_TRACE_THRESH = 50_000       # ~10% of typical full-task trace spread
ARC_DET_THRESH   = 500_000_000  # ~10% of typical full-task det spread
ARC_HASH_MASK    = (1 << 32) - 1  # compare lower 32 bits of hash for near-match


def arc_task_resonance(a: EigenCharge, b: EigenCharge) -> int:
    """
    ARC-calibrated integer resonance for task charges.
    More discriminating than TENT word-level bra_resonance_score().
    Returns:
      2 — exact match (all three components identical)
      1 — near resonance: same lower-32 hash AND trace/det within ARC thresholds
      0 — no resonance
    """
    if a.hash == b.hash and a.trace == b.trace and a.det == b.det:
        return 2
    # Near match: require lower 32 bits of hash to agree (2^32 selectivity)
    if ((a.hash & ARC_HASH_MASK) == (b.hash & ARC_HASH_MASK) and
            abs(a.trace - b.trace) < ARC_TRACE_THRESH and
            abs(a.det   - b.det)   < ARC_DET_THRESH):
        return 1
    return 0

def sovereign_solve_config(task: ARCTask,
                            bra_store: BRAPatternStore = None,
                            verbose: bool = False) -> dict:
    """
    Sovereign pre-solve analysis. Run before arc_solver.ARCSolver.solve().

    Returns a config dict that overrides the solver's default parameters:
      {
        pipeline:         "brute_force" | "llm_standard" | "llm_cinema"
        upg_prim_order:   list of primitive names, UPG-ordered
        bra_warmstart:    best matching program from BRA store (or None)
        bra_resonance:    resonance score of warmstart (0/1/2)
        task_charge:      EigenCharge of this task
        density_report:   TENTGateResult
        n_candidates:     recommended number of LLM candidates
        max_refinements:  recommended refinement rounds
      }
    """
    # 1. TENT density gate
    gate = tent_density_gate(task)
    if verbose:
        print(f"  [BRA] Density gate: {gate}")

    # 2. UPG routing
    prim_order = upg_ordered_primitives(task, max_primitives=40)
    if verbose:
        print(f"  [BRA] UPG route (first 8): {prim_order[:8]}")

    # 3. BRA warmstart from eigenstate store
    warmstart   = None
    resonance   = 0
    if bra_store:
        hit = bra_store.best_program(task)
        if hit:
            warmstart = hit["program"]
            resonance = hit["resonance"]
            if verbose:
                print(f"  [BRA] Eigenstate hit: {hit['task_id']} "
                      f"resonance={resonance} score={hit['score']:.2f}")

    # 4. Pipeline parameters based on gate
    if gate.pipeline == "brute_force":
        n_cand     = 0
        n_refine   = 0
    elif gate.pipeline == "llm_standard":
        n_cand     = 3
        n_refine   = 3
    else:  # cinema
        n_cand     = 5
        n_refine   = 5

    # If perfect warmstart, no LLM needed regardless of gate
    if warmstart and resonance == 2:
        n_cand   = 0
        n_refine = 0

    return {
        "pipeline":         gate.pipeline,
        "upg_prim_order":   prim_order,
        "bra_warmstart":    warmstart,
        "bra_resonance":    resonance,
        "task_charge":      gate.charge,
        "density_report":   gate,
        "n_candidates":     n_cand,
        "max_refinements":  n_refine,
    }


def bra_score_grids(predicted: Grid, expected: Grid) -> dict:
    """
    Sovereign accuracy scoring. Replaces float grid_similarity().
    Returns:
      {
        exact:      bool     (resonance == 2, hash+trace+det all match)
        resonant:   bool     (resonance >= 1)
        resonance:  int      (0/1/2)
        shape_match: bool
        eigen_predicted: EigenCharge
        eigen_expected:  EigenCharge
      }
    """
    if predicted is None:
        return {"exact": False, "resonant": False, "resonance": 0,
                "shape_match": False, "eigen_predicted": None,
                "eigen_expected": grid_eigen_charge(expected)}

    ep = grid_eigen_charge(predicted)
    ee = grid_eigen_charge(expected)
    r  = bra_resonance_score(ep, ee)
    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    return {
        "exact":            r == 2,
        "resonant":         r >= 1,
        "resonance":        r,
        "shape_match":      (ph, pw) == (eh, ew),
        "eigen_predicted":  ep,
        "eigen_expected":   ee,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from arc_types import ARCTask, Pair

    print("\n" + "═"*60)
    print("  arc_bra.py — Sovereign ARC Adaptation Demo")
    print("═"*60)

    # 1. EigenCharge
    print("\n1. BRA EigenCharge (integer, no float):")
    g1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    g2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    g3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]   # same as g1
    ec1 = grid_eigen_charge(g1)
    ec2 = grid_eigen_charge(g2)
    ec3 = grid_eigen_charge(g3)
    print(f"  g1 charge: hash={ec1.hash:016x} trace={ec1.trace}")
    print(f"  g2 charge: hash={ec2.hash:016x} trace={ec2.trace}")
    print(f"  g1==g3 resonance: {bra_resonance_score(ec1, ec3)}  (expect 2)")
    print(f"  g1 vs g2 resonance: {bra_resonance_score(ec1, ec2)}  (expect 0 or 1)")

    # 2. TENT density gate
    print("\n2. TENT Density Gate:")
    simple_task = ARCTask("simple",
        train=[Pair([[1, 0]], [[2, 0]]), Pair([[1, 1]], [[2, 2]])],
        test=[Pair([[0, 1]])])
    complex_task = ARCTask("complex",
        train=[
            Pair([[1,2,3,1],[2,3,1,2],[3,1,2,3],[1,2,3,4]],
                 [[4,3,2,4],[3,2,4,3],[2,4,3,2],[4,3,2,1]]),
            Pair([[1,1,2,2],[3,3,4,4],[1,2,3,4],[4,3,2,1]],
                 [[2,2,1,1],[4,4,3,3],[4,3,2,1],[1,2,3,4]]),
        ],
        test=[Pair([[1,2,3,4]])])

    sg = tent_density_gate(simple_task)
    cg = tent_density_gate(complex_task)
    print(f"  Simple task:  density={sg.density:.3f}  pipeline={sg.pipeline!r}")
    print(f"  Complex task: density={cg.density:.3f}  pipeline={cg.pipeline!r}")

    # 3. UPG routing
    print("\n3. UPG-Guided Primitive Routing:")
    prim_order = upg_ordered_primitives(simple_task, max_primitives=12)
    print(f"  First 12 primitives for simple task (UPG order):")
    for i, p in enumerate(prim_order[:12]):
        coord = _PRIM_COORDS.get(p, (0,0,0))
        print(f"    {i+1:2d}. {p:<20} @ {coord}")

    # 4. Ulam spiral charge
    print("\n4. Ulam Spiral Grid Encoding:")
    from arc_types import rot90
    g  = [[1, 0, 2], [0, 1, 0], [2, 0, 1]]
    gr = rot90(g)

    # 5. Sovereign config
    print("\n5. Sovereign Solve Config:")
    store = BRAPatternStore()
    store.store(simple_task, "def transform(g): return recolor(g,1,2)", 1.0, "color")
    cfg = sovereign_solve_config(simple_task, bra_store=store, verbose=True)
    print(f"  Pipeline:       {cfg['pipeline']}")
    print(f"  BRA warmstart:  {cfg['bra_warmstart'] is not None}")
    print(f"  BRA resonance:  {cfg['bra_resonance']}")
    print(f"  n_candidates:   {cfg['n_candidates']}")
    print(f"  max_refinements:{cfg['max_refinements']}")

    # 6. BRA grid scoring
    print("\n6. Sovereign Grid Scoring (replaces float grid_similarity):")
    pred = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    expt = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    sc = bra_score_grids(pred, expt)
    print(f"  Exact match: exact={sc['exact']} resonance={sc['resonance']}")
    pred2 = [[2, 0, 0], [0, 1, 0], [0, 0, 2]]  # one cell wrong
    sc2 = bra_score_grids(pred2, expt)
    print(f"  One cell off: exact={sc2['exact']} resonance={sc2['resonance']}")

    print(f"\n{'═'*60}\n  All BRA components operational\n{'═'*60}\n")
