"""
arc_abstraction.py — Object-Centric ARC Representation
=======================================================
Translates raw ARC grids into a rich structured description that
doubles LLM accuracy by explicitly encoding spatial relationships,
object properties, and transformation deltas.

Based on the finding (Xu et al.) that:
  - Linearized text grids cause horizontal/vertical asymmetry bias
  - Explicit object-level descriptions + coordinate DSL ~2× GPT-4 accuracy
  - Standardized object names + positional anchors help generalization

This module builds three complementary representations:
  1. OBJECT LIST    — per-object properties (color, size, bbox, shape sig)
  2. SPATIAL MAP    — relative positions between objects
  3. DELTA REPORT   — what changed between input and output (for training pairs)

All three are serialized to compact text suitable for LLM prompts.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from arc_types import (
    Grid, Pair, ARCTask,
    grid_shape, grid_to_text, grid_height, grid_width, grid_get,
    extract_objects, background_color, count_colors, detect_symmetry,
    ARC_COLOR_NAMES, ARC_COLOR_CHARS, grid_diff,
    most_common_color, crop,
)


# ── Object descriptor ──────────────────────────────────────────────────────────

@dataclass
class ObjectDesc:
    obj_id:   str         # e.g. "A", "B", "C"
    color:    int
    size:     int         # cell count
    bbox:     tuple       # (r0, c0, r1, c1)
    center:   tuple       # (float row, float col)
    shape_sig: str        # normalized bitmask e.g. "1011@2x2"
    is_rect:  bool
    is_line:  bool        # 1-row or 1-col
    height:   int
    width:    int
    touching_border: bool
    grid:     Grid        # sub-grid of object

    @property
    def color_name(self) -> str:
        return ARC_COLOR_NAMES.get(self.color, f"color{self.color}")

    def to_text(self, grid_h: int, grid_w: int) -> str:
        r0, c0, r1, c1 = self.bbox
        # Quadrant
        cr, cc = self.center
        if cr < grid_h / 2:
            vpos = "top"
        elif cr > grid_h / 2:
            vpos = "bottom"
        else:
            vpos = "middle"
        if cc < grid_w / 2:
            hpos = "left"
        elif cc > grid_w / 2:
            hpos = "right"
        else:
            hpos = "center"

        shape_desc = []
        if self.is_rect:
            shape_desc.append("rectangle")
        if self.is_line:
            shape_desc.append("line")
        if not shape_desc:
            shape_desc.append("irregular")
        shape_str = "/".join(shape_desc)

        border_str = " [border]" if self.touching_border else ""
        return (
            f"  {self.obj_id}: {self.color_name} {shape_str} "
            f"{self.height}×{self.width} cells={self.size} "
            f"at ({r0},{c0})-({r1-1},{c1-1}) {vpos}-{hpos}{border_str}"
        )

    def grid_text(self) -> str:
        return grid_to_text(self.grid)


# ── Object extraction with stable IDs ────────────────────────────────────────

_ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _assign_ids(objs: list[dict]) -> list[ObjectDesc]:
    """Convert raw object dicts to ObjectDescs with stable letter IDs."""
    result = []
    for i, obj in enumerate(objs):
        g  = obj["grid"]
        h  = obj["height"]
        w  = obj["width"]
        r0, c0, r1, c1 = obj["bbox"]

        # Shape signature (normalized bitmask)
        bg  = 0
        sig = "".join(
            "1" if g[r][c] != bg else "0"
            for r in range(h) for c in range(w)
        ) + f"@{h}x{w}"

        cells  = obj["cells"]
        n      = len(cells)
        center = (sum(r for r,_ in cells)/n, sum(c for _,c in cells)/n)

        # Rect check
        is_rect = (n == h * w)
        # Line check
        is_line = (h == 1 or w == 1)

        result.append(ObjectDesc(
            obj_id   = _ID_CHARS[i % 26] + (str(i // 26) if i >= 26 else ""),
            color    = obj["color"],
            size     = n,
            bbox     = obj["bbox"],
            center   = center,
            shape_sig = sig,
            is_rect  = is_rect,
            is_line  = is_line,
            height   = h,
            width    = w,
            touching_border = obj.get("touching_border", False),
            grid     = g,
        ))
    return result


def extract_object_descs(g: Grid, bg: int = None) -> tuple[list[ObjectDesc], int]:
    """
    Extract ObjectDescs from a grid.
    Returns (descs, bg_color).
    """
    if bg is None:
        bg = background_color(g)
    raw_objs = extract_objects(g, bg)
    h, w     = grid_shape(g)

    descs = _assign_ids(raw_objs)

    # Tag border-touching
    for desc, raw in zip(descs, raw_objs):
        desc.touching_border = any(
            r == 0 or r == h-1 or c == 0 or c == w-1
            for r, c in raw["cells"]
        )

    return descs, bg


# ── Spatial relationship map ───────────────────────────────────────────────────

def _direction(from_desc: ObjectDesc, to_desc: ObjectDesc) -> str:
    """Cardinal direction from one object to another."""
    dr = to_desc.center[0] - from_desc.center[0]
    dc = to_desc.center[1] - from_desc.center[1]
    if abs(dr) < 0.5 and abs(dc) < 0.5:
        return "same"
    if abs(dr) > abs(dc) * 1.5:
        return "below" if dr > 0 else "above"
    if abs(dc) > abs(dr) * 1.5:
        return "right" if dc > 0 else "left"
    diag = ("below" if dr > 0 else "above") + "-" + ("right" if dc > 0 else "left")
    return diag


def build_spatial_map(descs: list[ObjectDesc]) -> str:
    """
    Build a compact spatial relationship description between all object pairs.
    """
    if len(descs) <= 1:
        return ""
    lines = ["  Spatial relationships:"]
    for i, a in enumerate(descs):
        for b in descs[i+1:]:
            dist_r = abs(a.center[0] - b.center[0])
            dist_c = abs(a.center[1] - b.center[1])
            dist   = math.sqrt(dist_r**2 + dist_c**2)
            dirn   = _direction(a, b)
            lines.append(
                f"    {a.obj_id}→{b.obj_id}: {dirn} dist={dist:.1f}"
            )
    return "\n".join(lines)


# ── Delta report (input→output changes) ───────────────────────────────────────

@dataclass
class DeltaReport:
    shape_changed: bool
    new_shape:     Optional[tuple]
    cells_changed: int
    color_changes: list   # [(from_color, to_color, count)]
    objects_added: list   # ObjectDesc list
    objects_removed: list # ObjectDesc list
    objects_moved: list   # (from_id, to_id, dr, dc) — matched by shape
    similarity:    float

    def to_text(self) -> str:
        lines = []
        if self.shape_changed:
            lines.append(f"  Shape changed → {self.new_shape[0]}×{self.new_shape[1]}")
        lines.append(f"  Cells changed: {self.cells_changed}")
        lines.append(f"  Pixel similarity: {self.similarity:.1%}")
        for fc, tc, n in self.color_changes[:8]:
            fn = ARC_COLOR_NAMES.get(fc, str(fc))
            tn = ARC_COLOR_NAMES.get(tc, str(tc))
            lines.append(f"  Recolor: {fn}→{tn} ({n} cells)")
        if self.objects_added:
            lines.append(f"  Added objects: {len(self.objects_added)}")
        if self.objects_removed:
            lines.append(f"  Removed objects: {len(self.objects_removed)}")
        for fr_id, to_id, dr, dc in self.objects_moved:
            lines.append(f"  Moved {fr_id}→{to_id}: Δ({dr:+d},{dc:+d})")
        return "\n".join(lines)


def compute_delta(inp: Grid, out: Grid) -> DeltaReport:
    """Compute a detailed delta between input and output grids."""
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    shape_changed = (ih, iw) != (oh, ow)
    diffs = grid_diff(inp, out) if not shape_changed else []
    # Sovereign: BRA integer resonance replaces float grid_similarity
    try:
        from arc_bra import grid_eigen_charge, bra_resonance_score
        _bra_r = bra_resonance_score(grid_eigen_charge(inp),
                                     grid_eigen_charge(out)) if not shape_changed else 0
        sim    = _bra_r / 2.0   # normalised for DeltaReport.similarity display only
    except ImportError:
        sim = 0.0

    # Color transition counts
    color_trans: dict[tuple, int] = {}
    for _, _, fc, tc in diffs:
        key = (fc, tc)
        color_trans[key] = color_trans.get(key, 0) + 1
    color_changes = sorted(
        [(fc, tc, n) for (fc, tc), n in color_trans.items()],
        key=lambda x: -x[2]
    )

    # Object-level delta
    in_descs,  _ = extract_object_descs(inp)
    out_descs, _ = extract_object_descs(out)

    # Match by shape signature
    in_sigs  = {d.shape_sig: d for d in in_descs}
    out_sigs = {d.shape_sig: d for d in out_descs}

    added    = [d for d in out_descs if d.shape_sig not in in_sigs]
    removed  = [d for d in in_descs  if d.shape_sig not in out_sigs]
    moved    = []
    for sig in set(in_sigs) & set(out_sigs):
        a = in_sigs[sig]
        b = out_sigs[sig]
        dr = int(b.bbox[0] - a.bbox[0])
        dc = int(b.bbox[1] - a.bbox[1])
        if dr != 0 or dc != 0:
            moved.append((a.obj_id, b.obj_id, dr, dc))

    return DeltaReport(
        shape_changed  = shape_changed,
        new_shape      = (oh, ow) if shape_changed else None,
        cells_changed  = len(diffs),
        color_changes  = color_changes,
        objects_added  = added,
        objects_removed = removed,
        objects_moved  = moved,
        similarity     = sim,
    )


# ── Grid summary ───────────────────────────────────────────────────────────────

def grid_summary(g: Grid, name: str = "Grid") -> str:
    """Compact multi-line summary of a grid for LLM prompts."""
    h, w   = grid_shape(g)
    colors = count_colors(g)
    symm   = detect_symmetry(g)
    symm_s = ", ".join(k for k, v in symm.items() if v) or "none"
    bg     = background_color(g)
    descs, _ = extract_object_descs(g, bg)
    color_s  = ", ".join(
        f"{ARC_COLOR_NAMES.get(c,'c'+str(c))}:{n}"
        for c, n in sorted(colors.items()) if c != bg
    )
    lines = [
        f"{name}: {h}×{w}  bg={ARC_COLOR_NAMES.get(bg,'?')}  "
        f"colors=[{color_s}]  symmetry=[{symm_s}]",
        f"  Objects ({len(descs)}):",
    ]
    for d in descs:
        lines.append(d.to_text(h, w))
    lines.append("  Grid:")
    lines.append(grid_to_text(g))
    return "\n".join(lines)


# ── Full pair abstraction ──────────────────────────────────────────────────────

def abstract_pair(pair: Pair, index: int) -> str:
    """
    Full structured description of one training pair for the LLM prompt.
    Includes: grid summaries, object lists, spatial map, delta report.
    """
    lines = [f"=== Example {index+1} ==="]
    lines.append(grid_summary(pair.input, "INPUT"))

    if pair.output:
        lines.append(grid_summary(pair.output, "OUTPUT"))
        delta = compute_delta(pair.input, pair.output)
        lines.append("TRANSFORMATION:")
        lines.append(delta.to_text())

    return "\n".join(lines)


def abstract_task(task: ARCTask) -> str:
    """
    Build the full object-centric task description for Stage 1 analysis.
    """
    lines = [
        "You are solving an ARC-AGI puzzle.",
        "Color encoding: " + "  ".join(
            f"{ARC_COLOR_CHARS[i]}={ARC_COLOR_NAMES[i]}" for i in range(10)),
        "",
    ]
    for i, pair in enumerate(task.train):
        lines.append(abstract_pair(pair, i))
        lines.append("")

    lines.append("=== TEST INPUT ===")
    for i, pair in enumerate(task.test):
        lines.append(grid_summary(pair.input, f"TEST {i+1}"))
        lines.append("")

    lines += [
        "=== INSTRUCTIONS ===",
        "1. Study the training examples carefully.",
        "2. Identify the EXACT transformation rule (be specific about:",
        "   - Which objects are affected",
        "   - What spatial/color/size conditions trigger the rule",
        "   - What the output shape will be)",
        "3. Apply the rule to the test input.",
        "4. State the rule precisely, then write the transform() program.",
    ]
    return "\n".join(lines)


# ── Compact encoding for synthesis prompt ─────────────────────────────────────

def encode_pair_compact(pair: Pair, index: int) -> str:
    """
    Compact encoding for the synthesis prompt (rule already known).
    Shows grid, object list, and delta — no spatial map.
    """
    lines = [f"Ex{index+1}:"]
    h, w  = grid_shape(pair.input)
    bg    = background_color(pair.input)
    in_descs, _ = extract_object_descs(pair.input, bg)

    # Input grid
    lines.append(f"  IN({h}×{w}): " + grid_to_text(pair.input).replace("\n", " | "))
    lines.append(f"  Objects: " + ", ".join(
        f"{d.obj_id}={d.color_name}({d.size})" for d in in_descs
    ))

    if pair.output:
        oh, ow = grid_shape(pair.output)
        lines.append(f"  OUT({oh}×{ow}): " + grid_to_text(pair.output).replace("\n", " | "))
        delta = compute_delta(pair.input, pair.output)
        if delta.cells_changed:
            for fc, tc, n in delta.color_changes[:3]:
                fn = ARC_COLOR_NAMES.get(fc, str(fc))
                tn = ARC_COLOR_NAMES.get(tc, str(tc))
                lines.append(f"  Δ: {fn}→{tn} ×{n}")
        if delta.shape_changed:
            lines.append(f"  Δshape: → {oh}×{ow}")

    return "\n".join(lines)


def encode_task_compact(task: ARCTask) -> str:
    """Compact multi-pair encoding for synthesis prompts."""
    parts = [encode_pair_compact(p, i) for i, p in enumerate(task.train)]
    return "\n".join(parts)


# ── Abstraction-aware system prompt ───────────────────────────────────────────

ABSTRACT_ANALYSIS_SYSTEM = """\
You are an expert ARC-AGI solver.
I will show you training examples in two forms:
  1. A character grid (each char = one colored cell)
  2. An object list (structured properties of each connected component)
  3. A delta report (what changed from input to output)

Your job: identify the EXACT transformation rule.
Then describe it precisely in 2-4 sentences covering:
  - Which objects are transformed
  - What geometric/color/count/size condition triggers the transformation
  - What the output shape will be (same as input, or different)
  - Any exceptions or edge cases

Think step by step. Be precise. One rule."""

ABSTRACT_SYNTHESIS_SYSTEM = """\
You are an ARC-AGI solver that writes Python programs.
Write a `transform(grid)` function implementing the given rule.

Available helpers (pre-imported, do NOT redefine):
  BASE DSL:
    grid_height, grid_width, grid_shape, grid_copy, empty_grid
    rot90, rot180, rot270, reflect_h, reflect_v, reflect_diag, reflect_anti
    crop, pad, tile, recolor, replace_colors, fill, overlay, hstack, vstack
    upscale, downscale, extract_objects, background_color, crop_to_content
    count_colors, most_common_color, least_common_color, detect_symmetry
    grid_diff, grid_similarity, grid_unique_colors, grid_color_count

  EXTENDED DSL:
    gravity(g, direction)          # "down"|"up"|"left"|"right"
    gravity_blocked(g, direction)
    complete_h_symmetry(g, bg=0)
    complete_v_symmetry(g, bg=0)
    complete_rot180_symmetry(g, bg=0)
    complete_diagonal_symmetry(g, bg=0)
    enforce_h_symmetry(g)
    enforce_v_symmetry(g)
    split_by_divider(g, divider_color, direction)  # returns list[Grid]
    split_into_quadrants(g)        # returns [TL, TR, BL, BR]
    split_grid(g, n_rows, n_cols)  # returns list[list[Grid]]
    join_grids(parts)
    hstack(*grids), vstack(*grids)
    filter_objects_by_color(g, color, bg=0)
    filter_objects_by_size(g, min_size, max_size, bg=0)
    largest_object(g, bg=0)
    smallest_object(g, bg=0)
    sort_objects_by(g, key, reverse=False, bg=0)  # key: size|color|row|col
    color_objects_by_rank(g, key, colors=None, bg=0)
    place_object(base, obj_grid, r, c, bg=0)
    count_objects(g, bg=0)
    objects_touching_border(g, bg=0)
    objects_not_touching_border(g, bg=0)
    dilate(g, color=None, bg=0, connectivity=4)
    erode(g, bg=0, connectivity=4)
    outline(g, bg=0, thickness=1, outline_color=None)
    fill_holes(g, bg=0)
    convex_hull_fill(g, color=None, bg=0)
    grid_xor(a, b, bg=0)
    grid_and(a, b, bg=0)
    grid_or(a, b, color_b_wins=True, bg=0)
    conditional_recolor(g, condition_grid, from_color, to_color, bg=0)
    mask_apply(g, mask, fill_color, bg=0)
    detect_period(g, direction)    # returns int or None
    tile_to_size(g, target_h, target_w)
    mosaic(parts, n_rows, n_cols)
    repeat_pattern(g, n, direction)
    shortest_path(g, start, end, passable=None, bg=0)
    draw_path(g, path, color)
    connect_cells(g, color_a, color_b, line_color, bg=0)
    grid_add(a, b, mod=10)
    grid_subtract(a, b, mod=10)
    threshold(g, value, above_color, below_color)
    normalize_colors(g)

Rules:
  - def transform(grid): ... return result
  - grid is a list[list[int]], values 0-9
  - No numpy, PIL, or external libraries
  - Return ONLY the function, in a ```python``` block"""
