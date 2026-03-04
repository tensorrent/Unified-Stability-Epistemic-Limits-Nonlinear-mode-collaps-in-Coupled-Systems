"""
arc_dsl_ext.py — Extended ARC-AGI DSL Primitives
==================================================
Augments arc_types.py with higher-level operations that appear
frequently in ARC-AGI tasks but require more than one atomic step.

All functions are pure: Grid → Grid (or similar), no side effects.
All are available in the solver's exec namespace.

Categories:
  GRAVITY       — objects fall/float in a direction until blocked
  SYMMETRY      — complete/detect/enforce axis symmetry
  SPLITTING     — divide grids into sub-grids by dividers or count
  OBJECT OPS    — filter, sort, align, count, place objects
  MORPHOLOGY    — erosion, dilation, outline, fill_holes, convex hull
  COLOR LOGIC   — conditional recolor, XOR/AND/OR of two grids
  PATTERN       — detect period, tile-complete, mosaic
  PATH/TRACE    — trace shortest path, connect two points
  ARITHMETIC    — add/subtract grid values, scale, threshold
  SHAPE UTILS   — bounding box, centroid, area, neighbors
"""

from __future__ import annotations
from arc_types import (
    Grid, empty_grid, grid_copy, grid_height, grid_width, grid_shape,
    grid_get, grid_set, fill, extract_objects, background_color,
    reflect_h, reflect_v, reflect_diag, rot90, crop, overlay, hstack,
    vstack, recolor, replace_colors, upscale, downscale, count_colors,
    most_common_color, least_common_color,
)
from arc_tensor import (
    int_einsum, einsum_affine, einsum_color_map
)
from collections import deque
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# GRAVITY
# ═══════════════════════════════════════════════════════════════

def gravity(g: Grid, direction: str = "down", bg: int = 0) -> Grid:
    """
    Make all non-bg cells fall in direction until blocked by another
    non-bg cell or the wall.
    direction: "down" | "up" | "left" | "right"
    """
    h, w = grid_shape(g)

    if direction == "down":
        result = empty_grid(h, w, bg)
        for c in range(w):
            col = [g[r][c] for r in range(h)]
            non_bg = [v for v in col if v != bg]
            for i, v in enumerate(non_bg):
                result[h - len(non_bg) + i][c] = v
        return result

    if direction == "up":
        result = empty_grid(h, w, bg)
        for c in range(w):
            col = [g[r][c] for r in range(h)]
            non_bg = [v for v in col if v != bg]
            for i, v in enumerate(non_bg):
                result[i][c] = v
        return result

    if direction == "right":
        result = empty_grid(h, w, bg)
        for r in range(h):
            non_bg = [v for v in g[r] if v != bg]
            for i, v in enumerate(non_bg):
                result[r][w - len(non_bg) + i] = v
        return result

    if direction == "left":
        result = empty_grid(h, w, bg)
        for r in range(h):
            non_bg = [v for v in g[r] if v != bg]
            for i, v in enumerate(non_bg):
                result[r][i] = v
        return result

    raise ValueError(f"Unknown gravity direction: {direction!r}")


def gravity_blocked(g: Grid, direction: str = "down", bg: int = 0) -> Grid:
    """
    Each cell falls until it hits another non-bg cell (not the wall).
    Simulates stacking — cells stop on top of each other.
    """
    h, w = grid_shape(g)
    result = grid_copy(g)

    if direction == "down":
        for c in range(w):
            moved = True
            while moved:
                moved = False
                for r in range(h - 2, -1, -1):
                    if result[r][c] != bg and result[r+1][c] == bg:
                        result[r+1][c] = result[r][c]
                        result[r][c] = bg
                        moved = True

    elif direction == "up":
        for c in range(w):
            moved = True
            while moved:
                moved = False
                for r in range(1, h):
                    if result[r][c] != bg and result[r-1][c] == bg:
                        result[r-1][c] = result[r][c]
                        result[r][c] = bg
                        moved = True

    elif direction == "right":
        for r in range(h):
            moved = True
            while moved:
                moved = False
                for c in range(w - 2, -1, -1):
                    if result[r][c] != bg and result[r][c+1] == bg:
                        result[r][c+1] = result[r][c]
                        result[r][c] = bg
                        moved = True

    elif direction == "left":
        for r in range(h):
            moved = True
            while moved:
                moved = False
                for c in range(1, w):
                    if result[r][c] != bg and result[r][c-1] == bg:
                        result[r][c-1] = result[r][c]
                        result[r][c] = bg
                        moved = True

    return result


def shift_object(g: Grid, obj: dict, dx: int, dy: int, bg: int = 0) -> Grid:
    """
    Shifts a specifically extracted object dictionary rigidly by (dx, dy) 
    pixels. This guarantees strict structural continuity deduced by Z3.
    """
    h, w = grid_shape(g)
    result = grid_copy(g)
    
    # 1. Erase original object to strictly enforce background overlap matching
    r_min, c_min, _, _ = obj["bbox"]
    for lr, row in enumerate(obj["grid"]):
        for lc, val in enumerate(row):
            if val != 0:
                 gr, gc = r_min + lr, c_min + lc
                 if 0 <= gr < h and 0 <= gc < w:
                      result[gr][gc] = bg
                      
    # 2. Draw object at mathematically translated coordinate
    for lr, row in enumerate(obj["grid"]):
        for lc, val in enumerate(row):
             if val != 0:
                  gr, gc = r_min + lr + dy, c_min + lc + dx
                  if 0 <= gr < h and 0 <= gc < w:
                       result[gr][gc] = val
                       
    return result

# ═══════════════════════════════════════════════════════════════
# SYMMETRY
# ═══════════════════════════════════════════════════════════════

def complete_h_symmetry(g: Grid, bg: int = 0) -> Grid:
    """
    Make grid horizontally symmetric. Non-bg cells take priority over bg.
    Each row is made symmetric by mirroring non-bg values.
    """
    result = grid_copy(g)
    h, w = grid_shape(g)
    for r in range(h):
        for c in range(w):
            mirror = w - 1 - c
            left, right = result[r][c], result[r][mirror]
            if left != bg and right == bg:
                result[r][mirror] = left
            elif right != bg and left == bg:
                result[r][c] = right
    return result


def complete_v_symmetry(g: Grid, bg: int = 0) -> Grid:
    """Make grid vertically symmetric."""
    result = grid_copy(g)
    h, w = grid_shape(g)
    for r in range(h):
        mirror = h - 1 - r
        for c in range(w):
            top, bot = result[r][c], result[mirror][c]
            if top != bg and bot == bg:
                result[mirror][c] = top
            elif bot != bg and top == bg:
                result[r][c] = bot
    return result


def complete_rot180_symmetry(g: Grid, bg: int = 0) -> Grid:
    """Make grid 180°-rotation symmetric."""
    result = grid_copy(g)
    h, w = grid_shape(g)
    for r in range(h):
        for c in range(w):
            mr, mc = h - 1 - r, w - 1 - c
            v1, v2 = result[r][c], result[mr][mc]
            if v1 != bg and v2 == bg:
                result[mr][mc] = v1
            elif v2 != bg and v1 == bg:
                result[r][c] = v2
    return result


def complete_diagonal_symmetry(g: Grid, bg: int = 0) -> Grid:
    """Make square grid diagonally symmetric (transpose-symmetric)."""
    h, w = grid_shape(g)
    if h != w:
        return g
    result = grid_copy(g)
    for r in range(h):
        for c in range(r+1, w):
            v1, v2 = result[r][c], result[c][r]
            if v1 != bg and v2 == bg:
                result[c][r] = v1
            elif v2 != bg and v1 == bg:
                result[r][c] = v2
    return result


def enforce_h_symmetry(g: Grid) -> Grid:
    """Force exact horizontal symmetry by averaging left/right halves."""
    h, w = grid_shape(g)
    result = grid_copy(g)
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            # Left takes priority
            result[r][mc] = result[r][c]
    return result


def enforce_v_symmetry(g: Grid) -> Grid:
    """Force exact vertical symmetry (top half wins)."""
    h, w = grid_shape(g)
    result = grid_copy(g)
    for r in range(h // 2):
        result[h - 1 - r] = result[r][:]
    return result


# ═══════════════════════════════════════════════════════════════
# GRID SPLITTING AND RECOMBINING
# ═══════════════════════════════════════════════════════════════

def split_by_divider(g: Grid, divider_color: int,
                     direction: str = "h") -> list[Grid]:
    """
    Split grid at rows/cols that are entirely filled with divider_color.
    direction: "h" (split by rows) | "v" (split by cols)
    """
    h, w = grid_shape(g)

    if direction == "h":
        dividers = [r for r in range(h)
                    if all(g[r][c] == divider_color for c in range(w))]
        starts = [0] + [d+1 for d in dividers]
        ends   = dividers + [h]
        parts  = []
        for s, e in zip(starts, ends):
            if e > s:
                parts.append([row[:] for row in g[s:e]])
        return parts

    else:  # "v"
        dividers = [c for c in range(w)
                    if all(g[r][c] == divider_color for r in range(h))]
        starts = [0] + [d+1 for d in dividers]
        ends   = dividers + [w]
        parts  = []
        for s, e in zip(starts, ends):
            if e > s:
                parts.append([[g[r][c] for c in range(s, e)] for r in range(h)])
        return parts


def split_into_quadrants(g: Grid) -> list[Grid]:
    """Split grid into 4 equal quadrants [TL, TR, BL, BR]."""
    h, w = grid_shape(g)
    mh, mw = h // 2, w // 2
    return [
        [row[:mw] for row in g[:mh]],           # TL
        [row[mw:] for row in g[:mh]],            # TR
        [row[:mw] for row in g[mh:]],            # BL
        [row[mw:] for row in g[mh:]],            # BR
    ]


def split_grid(g: Grid, n_rows: int, n_cols: int) -> list[list[Grid]]:
    """
    Split grid into n_rows × n_cols equal sub-grids.
    Returns 2D list: result[r][c] = subgrid at row r, col c.
    """
    h, w = grid_shape(g)
    sh, sw = h // n_rows, w // n_cols
    result = []
    for ri in range(n_rows):
        row = []
        for ci in range(n_cols):
            r0, r1 = ri*sh, (ri+1)*sh
            c0, c1 = ci*sw, (ci+1)*sw
            row.append([g[r][c0:c1] for r in range(r0, r1)])
        result.append(row)
    return result


def join_grids(parts: list[list[Grid]]) -> Grid:
    """Inverse of split_grid: join 2D list of sub-grids."""
    return vstack(*[hstack(*row) for row in parts]) if parts else []


def hstack(*grids) -> Grid:
    """Stack any number of grids horizontally."""
    if not grids:
        return []
    result = [row[:] for row in grids[0]]
    for g in grids[1:]:
        for r in range(len(result)):
            result[r] += g[r] if r < len(g) else [0]*grid_width(g)
    return result


def vstack(*grids) -> Grid:
    """Stack any number of grids vertically."""
    result = []
    for g in grids:
        result.extend(row[:] for row in g)
    return result


# ═══════════════════════════════════════════════════════════════
# OBJECT OPERATIONS
# ═══════════════════════════════════════════════════════════════

def filter_objects_by_color(g: Grid, color: int, bg: int = 0) -> Grid:
    """Keep only objects of the given color, everything else → bg."""
    return [[c if c == color else bg for c in row] for row in g]


def filter_objects_by_size(g: Grid, min_size: int = 1,
                            max_size: int = 9999, bg: int = 0) -> Grid:
    """Keep only objects whose cell count is in [min_size, max_size]."""
    objs   = extract_objects(g, bg)
    result = empty_grid(*grid_shape(g), bg)
    for obj in objs:
        if min_size <= obj["size"] <= max_size:
            for r, c in obj["cells"]:
                result[r][c] = g[r][c]
    return result


def largest_object(g: Grid, bg: int = 0) -> Grid:
    """Return a grid containing only the largest object."""
    objs = extract_objects(g, bg)
    if not objs:
        return g
    biggest = max(objs, key=lambda o: o["size"])
    result  = empty_grid(*grid_shape(g), bg)
    for r, c in biggest["cells"]:
        result[r][c] = g[r][c]
    return result


def smallest_object(g: Grid, bg: int = 0) -> Grid:
    """Return a grid containing only the smallest object."""
    objs = extract_objects(g, bg)
    if not objs:
        return g
    tiny   = min(objs, key=lambda o: o["size"])
    result = empty_grid(*grid_shape(g), bg)
    for r, c in tiny["cells"]:
        result[r][c] = g[r][c]
    return result


def sort_objects_by(g: Grid, key: str = "size",
                    reverse: bool = False, bg: int = 0) -> list[dict]:
    """
    Return sorted list of object dicts.
    key: "size" | "color" | "row" | "col"
    """
    objs = extract_objects(g, bg)
    if key == "size":
        return sorted(objs, key=lambda o: o["size"], reverse=reverse)
    if key == "color":
        return sorted(objs, key=lambda o: o["color"], reverse=reverse)
    if key == "row":
        return sorted(objs, key=lambda o: o["bbox"][0], reverse=reverse)
    if key == "col":
        return sorted(objs, key=lambda o: o["bbox"][1], reverse=reverse)
    return objs


def color_objects_by_rank(g: Grid, key: str = "size",
                           colors: list = None, bg: int = 0) -> Grid:
    """
    Recolor objects by their rank (sorted by key).
    colors: list of colors to assign in rank order.
    If None, uses [1,2,3,...] (blue, red, green...).
    """
    objs   = sort_objects_by(g, key, reverse=False, bg=bg)
    colors = colors or list(range(1, len(objs)+1))
    result = empty_grid(*grid_shape(g), bg)
    for i, obj in enumerate(objs):
        c = colors[i % len(colors)]
        for r, col in obj["cells"]:
            result[r][col] = c
    return result


def place_object(base: Grid, obj_grid: Grid,
                 r: int, c: int, bg: int = 0) -> Grid:
    """Place obj_grid onto base at position (r, c), skipping bg cells."""
    result = grid_copy(base)
    oh, ow = grid_shape(obj_grid)
    for dr in range(oh):
        for dc in range(ow):
            v = obj_grid[dr][dc]
            if v != bg:
                nr, nc = r+dr, c+dc
                if 0 <= nr < grid_height(result) and 0 <= nc < grid_width(result):
                    result[nr][nc] = v
    return result


def move_object(g: Grid, from_r: int, from_c: int,
                to_r: int, to_c: int, bg: int = 0) -> Grid:
    """Move the object touching cell (from_r, from_c) to (to_r, to_c)."""
    objs = extract_objects(g, bg)
    target = None
    for obj in objs:
        if (from_r, from_c) in obj["cells"]:
            target = obj
            break
    if not target:
        return g
    result = grid_copy(g)
    # Erase
    for r, c in target["cells"]:
        result[r][c] = bg
    # Place
    r0, c0 = target["bbox"][0], target["bbox"][1]
    return place_object(result, target["grid"], to_r, to_c, bg)


def count_objects(g: Grid, bg: int = 0) -> int:
    """Count connected components (objects)."""
    return len(extract_objects(g, bg))


def objects_touching_border(g: Grid, bg: int = 0) -> Grid:
    """Return only objects that touch the border."""
    h, w   = grid_shape(g)
    objs   = extract_objects(g, bg)
    result = empty_grid(h, w, bg)
    for obj in objs:
        touches = any(
            r == 0 or r == h-1 or c == 0 or c == w-1
            for r, c in obj["cells"]
        )
        if touches:
            for r, c in obj["cells"]:
                result[r][c] = g[r][c]
    return result


def objects_not_touching_border(g: Grid, bg: int = 0) -> Grid:
    """Return only interior objects (not touching border)."""
    h, w   = grid_shape(g)
    objs   = extract_objects(g, bg)
    result = empty_grid(h, w, bg)
    for obj in objs:
        touches = any(
            r == 0 or r == h-1 or c == 0 or c == w-1
            for r, c in obj["cells"]
        )
        if not touches:
            for r, c in obj["cells"]:
                result[r][c] = g[r][c]
    return result


def align_objects_to_grid(g: Grid, template: Grid, bg: int = 0) -> Grid:
    """
    For each non-bg cell in template, place the corresponding object
    from g at the template position. Used for layout matching tasks.
    """
    # Placeholder: copy non-bg from g into positions marked by template
    h, w   = grid_shape(template)
    result = empty_grid(h, w, bg)
    for r in range(min(h, grid_height(g))):
        for c in range(min(w, grid_width(g))):
            if template[r][c] != bg:
                result[r][c] = g[r][c]
    return result


# ═══════════════════════════════════════════════════════════════
# MORPHOLOGY
# ═══════════════════════════════════════════════════════════════

def dilate(g: Grid, color: int = None, bg: int = 0,
           connectivity: int = 4) -> Grid:
    """
    Expand non-bg cells by one step (4 or 8 connected).
    If color is given, only dilate cells of that color.
    """
    h, w   = grid_shape(g)
    result = grid_copy(g)
    dirs4  = [(-1,0),(1,0),(0,-1),(0,1)]
    dirs8  = dirs4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    dirs   = dirs8 if connectivity == 8 else dirs4

    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v == bg:
                continue
            if color is not None and v != color:
                continue
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                    result[nr][nc] = v
    return result


def erode(g: Grid, bg: int = 0, connectivity: int = 4) -> Grid:
    """
    Shrink non-bg regions by one step. A cell survives erosion only
    if all its connected neighbours are also non-bg.
    """
    h, w   = grid_shape(g)
    result = empty_grid(h, w, bg)
    dirs4  = [(-1,0),(1,0),(0,-1),(0,1)]
    dirs8  = dirs4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    dirs   = dirs8 if connectivity == 8 else dirs4

    for r in range(h):
        for c in range(w):
            if g[r][c] == bg:
                continue
            if all(
                0 <= r+dr < h and 0 <= c+dc < w and g[r+dr][c+dc] != bg
                for dr, dc in dirs
            ):
                result[r][c] = g[r][c]
    return result


def outline(g: Grid, bg: int = 0, thickness: int = 1,
            outline_color: int = None) -> Grid:
    """
    Replace each non-bg object with just its border cells.
    outline_color: if None, keeps original color.
    """
    dilated = g
    for _ in range(thickness):
        dilated = dilate(dilated, bg=bg)
    eroded  = erode(g, bg=bg)
    h, w    = grid_shape(g)
    result  = empty_grid(h, w, bg)
    for r in range(h):
        for c in range(w):
            # Border = in g but not in eroded
            if g[r][c] != bg and eroded[r][c] == bg:
                result[r][c] = outline_color if outline_color else g[r][c]
    return result


def fill_holes(g: Grid, bg: int = 0) -> Grid:
    """
    Fill enclosed background regions (holes inside objects).
    Uses flood-fill from all border bg cells; remaining bg = holes.
    """
    h, w   = grid_shape(g)
    visited = [[False]*w for _ in range(h)]
    queue   = deque()

    # Seed from border bg cells
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and g[r][c] == bg:
                queue.append((r, c))
                visited[r][c] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc] == bg:
                visited[nr][nc] = True
                queue.append((nr, nc))

    result = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg and not visited[r][c]:
                result[r][c] = most_common_color(g, exclude={bg})
    return result


def convex_hull_fill(g: Grid, color: int = None, bg: int = 0) -> Grid:
    """
    Fill the convex hull of each object (simplified: fill bounding box).
    For a proper convex hull use the outline approach.
    """
    objs   = extract_objects(g, bg)
    result = grid_copy(g)
    for obj in objs:
        if color is not None and obj["color"] != color:
            continue
        r0, c0, r1, c1 = obj["bbox"]
        for r in range(r0, r1):
            for c in range(c0, c1):
                if result[r][c] == bg:
                    result[r][c] = obj["color"]
    return result


# ═══════════════════════════════════════════════════════════════
# COLOR LOGIC
# ═══════════════════════════════════════════════════════════════

def grid_xor(a: Grid, b: Grid, bg: int = 0) -> Grid:
    """XOR: cell is non-bg only where exactly one of a, b is non-bg."""
    h  = max(grid_height(a), grid_height(b))
    w  = max(grid_width(a),  grid_width(b))
    result = empty_grid(h, w, bg)
    for r in range(h):
        for c in range(w):
            va = grid_get(a, r, c, bg)
            vb = grid_get(b, r, c, bg)
            if (va != bg) != (vb != bg):
                result[r][c] = va if va != bg else vb
    return result


def grid_and(a: Grid, b: Grid, bg: int = 0) -> Grid:
    """AND: cell is non-bg only where both a and b are non-bg."""
    h  = max(grid_height(a), grid_height(b))
    w  = max(grid_width(a),  grid_width(b))
    result = empty_grid(h, w, bg)
    for r in range(h):
        for c in range(w):
            va = grid_get(a, r, c, bg)
            vb = grid_get(b, r, c, bg)
            if va != bg and vb != bg:
                result[r][c] = va
    return result


def grid_or(a: Grid, b: Grid, color_b_wins: bool = True,
             bg: int = 0) -> Grid:
    """OR: non-bg from either grid. b wins conflicts if color_b_wins."""
    h  = max(grid_height(a), grid_height(b))
    w  = max(grid_width(a),  grid_width(b))
    result = empty_grid(h, w, bg)
    for r in range(h):
        for c in range(w):
            va = grid_get(a, r, c, bg)
            vb = grid_get(b, r, c, bg)
            if va != bg and vb != bg:
                result[r][c] = vb if color_b_wins else va
            elif va != bg:
                result[r][c] = va
            elif vb != bg:
                result[r][c] = vb
    return result


def conditional_recolor(g: Grid, condition_grid: Grid,
                         from_color: int, to_color: int,
                         bg: int = 0) -> Grid:
    """
    Recolor cells in g from from_color to to_color only where
    condition_grid has a non-bg value.
    """
    result = grid_copy(g)
    h, w   = grid_shape(g)
    for r in range(h):
        for c in range(w):
            if (grid_get(condition_grid, r, c, bg) != bg and
                    result[r][c] == from_color):
                result[r][c] = to_color
    return result


def mask_apply(g: Grid, mask: Grid, fill_color: int,
               bg: int = 0) -> Grid:
    """Fill cells in g where mask is non-bg with fill_color."""
    result = grid_copy(g)
    h, w   = grid_shape(g)
    for r in range(h):
        for c in range(w):
            if grid_get(mask, r, c, bg) != bg:
                result[r][c] = fill_color
    return result


# ═══════════════════════════════════════════════════════════════
# PATTERN DETECTION AND COMPLETION
# ═══════════════════════════════════════════════════════════════

def detect_period(g: Grid, direction: str = "h") -> Optional[int]:
    """
    Detect smallest repeating period along rows (h) or cols (v).
    Returns period in cells, or None if not periodic.
    """
    if direction == "h":
        w = grid_width(g)
        for p in range(1, w // 2 + 1):
            if w % p != 0:
                continue
            if all(
                g[r][c] == g[r][c % p]
                for r in range(grid_height(g))
                for c in range(w)
            ):
                return p
        return None
    else:
        h = grid_height(g)
        for p in range(1, h // 2 + 1):
            if h % p != 0:
                continue
            if all(
                g[r][c] == g[r % p][c]
                for r in range(h)
                for c in range(grid_width(g))
            ):
                return p
        return None


def tile_to_size(g: Grid, target_h: int, target_w: int) -> Grid:
    """Tile g to exactly fill a (target_h × target_w) grid."""
    h, w   = grid_shape(g)
    result = empty_grid(target_h, target_w)
    for r in range(target_h):
        for c in range(target_w):
            result[r][c] = g[r % h][c % w]
    return result


def mosaic(parts: list[Grid], n_rows: int, n_cols: int) -> Grid:
    """
    Arrange parts (list of grids) in a n_rows × n_cols grid.
    parts must have length == n_rows * n_cols.
    """
    rows = []
    for ri in range(n_rows):
        row_grids = parts[ri*n_cols : (ri+1)*n_cols]
        rows.append(hstack(*row_grids))
    return vstack(*rows)


def repeat_pattern(g: Grid, n: int, direction: str = "h") -> Grid:
    """Repeat grid n times horizontally or vertically."""
    if direction == "h":
        return hstack(*([g] * n))
    return vstack(*([g] * n))


# ═══════════════════════════════════════════════════════════════
# PATH / TRACE
# ═══════════════════════════════════════════════════════════════

def shortest_path(g: Grid, start: tuple, end: tuple,
                  passable: set = None, bg: int = 0) -> list[tuple]:
    """
    BFS shortest path from start to end on passable cells.
    passable: set of colors that can be traversed. Default: {bg}.
    Returns list of (r, c) tuples, or [] if no path.
    """
    h, w      = grid_shape(g)
    passable  = passable if passable is not None else {bg}
    visited   = [[False]*w for _ in range(h)]
    prev      = {}
    queue     = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            # Reconstruct path
            path = []
            cur  = end
            while cur != start:
                path.append(cur)
                cur = prev[cur]
            path.append(start)
            return path[::-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (0<=nr<h and 0<=nc<w and not visited[nr][nc] and
                    g[nr][nc] in passable):
                visited[nr][nc] = True
                prev[(nr,nc)] = (r,c)
                queue.append((nr,nc))
    return []


def draw_path(g: Grid, path: list[tuple], color: int) -> Grid:
    """Draw a path (list of (r,c)) onto g with given color."""
    result = grid_copy(g)
    for r, c in path:
        result[r][c] = color
    return result


def connect_cells(g: Grid, color_a: int, color_b: int,
                  line_color: int, bg: int = 0) -> Grid:
    """
    Find one cell of color_a and one cell of color_b, draw
    an L-shaped or straight connector between them.
    """
    cells_a = [(r, c) for r in range(grid_height(g))
               for c in range(grid_width(g)) if g[r][c] == color_a]
    cells_b = [(r, c) for r in range(grid_height(g))
               for c in range(grid_width(g)) if g[r][c] == color_b]
    if not cells_a or not cells_b:
        return g
    ra, ca = cells_a[0]
    rb, cb = cells_b[0]
    result = grid_copy(g)
    # Horizontal then vertical
    for c in range(min(ca, cb), max(ca, cb)+1):
        if result[ra][c] == bg:
            result[ra][c] = line_color
    for r in range(min(ra, rb), max(ra, rb)+1):
        if result[r][cb] == bg:
            result[r][cb] = line_color
    return result


# ═══════════════════════════════════════════════════════════════
# ARITHMETIC / VALUE OPS
# ═══════════════════════════════════════════════════════════════

def grid_add(a: Grid, b: Grid, mod: int = 10) -> Grid:
    """Add corresponding cell values, mod 10."""
    h = max(grid_height(a), grid_height(b))
    w = max(grid_width(a),  grid_width(b))
    return [
        [(grid_get(a, r, c, 0) + grid_get(b, r, c, 0)) % mod
         for c in range(w)]
        for r in range(h)
    ]


def grid_subtract(a: Grid, b: Grid, mod: int = 10) -> Grid:
    """Subtract b from a, mod 10."""
    h = max(grid_height(a), grid_height(b))
    w = max(grid_width(a),  grid_width(b))
    return [
        [(grid_get(a, r, c, 0) - grid_get(b, r, c, 0)) % mod
         for c in range(w)]
        for r in range(h)
    ]


def threshold(g: Grid, value: int, above_color: int,
              below_color: int) -> Grid:
    """Map each cell: above_color if cell > value, else below_color."""
    return [[above_color if c > value else below_color for c in row] for row in g]


def normalize_colors(g: Grid) -> Grid:
    """
    Remap colors to lowest available values starting from 1.
    Preserves 0 (background). Useful for color-invariant matching.
    """
    present  = sorted(set(c for row in g for c in row if c != 0))
    mapping  = {c: i+1 for i, c in enumerate(present)}
    mapping[0] = 0
    return replace_colors(g, mapping)


# ═══════════════════════════════════════════════════════════════
# SHAPE UTILITIES
# ═══════════════════════════════════════════════════════════════

def centroid(obj: dict) -> tuple[float, float]:
    """Return (row, col) centroid of an object."""
    cells = obj["cells"]
    return (
        sum(r for r, _ in cells) / len(cells),
        sum(c for _, c in cells) / len(cells),
    )


def object_shape_signature(obj: dict, bg: int = 0) -> str:
    """Compact string describing object's shape (normalized bitmask)."""
    g    = obj["grid"]
    h, w = grid_shape(g)
    return "".join(
        "1" if g[r][c] != bg else "0"
        for r in range(h) for c in range(w)
    ) + f"@{h}x{w}"


def shapes_equal(obj_a: dict, obj_b: dict, bg: int = 0) -> bool:
    """True if two objects have the same shape (color-blind)."""
    return object_shape_signature(obj_a, bg) == object_shape_signature(obj_b, bg)


def find_matching_shape(template_obj: dict, g: Grid,
                         bg: int = 0) -> Optional[dict]:
    """Find an object in g that matches template_obj's shape."""
    sig  = object_shape_signature(template_obj, bg)
    objs = extract_objects(g, bg)
    for obj in objs:
        if object_shape_signature(obj, bg) == sig:
            return obj
    return None


def extract_bounding_box(g: Grid, bg: int = 0) -> Grid:
    """Return the smallest subgrid containing all non-bg cells."""
    h, w = grid_shape(g)
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if min_r > max_r or min_c > max_c:
        return g
    result = empty_grid(max_r - min_r + 1, max_c - min_c + 1, bg)
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            result[r - min_r][c - min_c] = g[r][c]
    return result


def flood_fill(g: Grid, r: int, c: int, target_color: int, replacement_color: int) -> Grid:
    """Algorithmically fill all connected target_color cells starting at (r,c) with replacement_color."""
    h, w = grid_shape(g)
    if not (0 <= r < h and 0 <= c < w) or g[r][c] != target_color or target_color == replacement_color:
        return g
    result = grid_copy(g)
    queue = deque([(r, c)])
    result[r][c] = replacement_color
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == target_color:
                result[nr][nc] = replacement_color
                queue.append((nr, nc))
    return result


def find_and_replace_pattern(g: Grid, template: Grid, replacement: Grid, bg: int = 0) -> Grid:
    """Search for exact matches of template (ignoring bg) and swap with replacement."""
    h, w = grid_shape(g)
    th, tw = grid_shape(template)
    rh, rw = grid_shape(replacement)
    result = grid_copy(g)
    if th != rh or tw != rw:
        return g  # Only support same-size replacements for now
    
    for r in range(h - th + 1):
        for c in range(w - tw + 1):
            match = True
            for tr in range(th):
                for tc in range(tw):
                    tv = template[tr][tc]
                    if tv != bg and result[r + tr][c + tc] != tv:
                        match = False
                        break
                if not match:
                    break
            if match:
                for tr in range(th):
                    for tc in range(tw):
                        rv = replacement[tr][tc]
                        if rv != bg:
                            result[r + tr][c + tc] = rv
    return result


def draw_line_until(g: Grid, r: int, c: int, direction: str, color: int, stop_colors: set = None) -> Grid:
    """Draw a line from (r,c) in direction ('up', 'down', 'left', 'right') until hitting a stop color or edge."""
    result = grid_copy(g)
    h, w = grid_shape(g)
    dr, dc = 0, 0
    if direction == 'up': dr = -1
    elif direction == 'down': dr = 1
    elif direction == 'left': dc = -1
    elif direction == 'right': dc = 1
    else: return g
    
    stop_colors = stop_colors or {color} # default stop on itself
    
    cr, cc = r + dr, c + dc
    while 0 <= cr < h and 0 <= cc < w:
        if result[cr][cc] in stop_colors:
            break
        result[cr][cc] = color
        cr += dr
        cc += dc
    return result


# ═══════════════════════════════════════════════════════════════
# FULL NAMESPACE EXPORT (for arc_solver exec globals)
# ═══════════════════════════════════════════════════════════════

EXT_DSL_NAMESPACE = {
    # Gravity
    "gravity":              gravity,
    "gravity_blocked":      gravity_blocked,
    # Symmetry
    "complete_h_symmetry":  complete_h_symmetry,
    "complete_v_symmetry":  complete_v_symmetry,
    "complete_rot180_symmetry": complete_rot180_symmetry,
    "complete_diagonal_symmetry": complete_diagonal_symmetry,
    "enforce_h_symmetry":   enforce_h_symmetry,
    "enforce_v_symmetry":   enforce_v_symmetry,
    # Splitting
    "split_by_divider":     split_by_divider,
    "split_into_quadrants": split_into_quadrants,
    "split_grid":           split_grid,
    "join_grids":           join_grids,
    "hstack":               hstack,
    "vstack":               vstack,
    # Object ops
    "filter_objects_by_color":  filter_objects_by_color,
    "filter_objects_by_size":   filter_objects_by_size,
    "largest_object":           largest_object,
    "smallest_object":          smallest_object,
    "sort_objects_by":          sort_objects_by,
    "color_objects_by_rank":    color_objects_by_rank,
    "place_object":             place_object,
    "move_object":              move_object,
    "count_objects":            count_objects,
    "objects_touching_border":  objects_touching_border,
    "objects_not_touching_border": objects_not_touching_border,
    "align_objects_to_grid":   align_objects_to_grid,
    # Morphology
    "dilate":               dilate,
    "erode":                erode,
    "outline":              outline,
    "fill_holes":           fill_holes,
    "convex_hull_fill":     convex_hull_fill,
    # Color logic
    "grid_xor":             grid_xor,
    "grid_and":             grid_and,
    "grid_or":              grid_or,
    "conditional_recolor":  conditional_recolor,
    "mask_apply":           mask_apply,
    # Pattern
    "detect_period":        detect_period,
    "tile_to_size":         tile_to_size,
    "mosaic":               mosaic,
    "repeat_pattern":       repeat_pattern,
    # Path
    "shortest_path":        shortest_path,
    "draw_path":            draw_path,
    "connect_cells":        connect_cells,
    # Arithmetic
    "grid_add":             grid_add,
    "grid_subtract":        grid_subtract,
    "threshold":            threshold,
    "normalize_colors":     normalize_colors,
    # Shape utils
    "centroid":             centroid,
    "object_shape_signature": object_shape_signature,
    "shapes_equal":         shapes_equal,
    "find_matching_shape":  find_matching_shape,
    "extract_bounding_box": extract_bounding_box,
    # New Phase 5 Primitives
    "flood_fill":           flood_fill,
    "find_and_replace_pattern": find_and_replace_pattern,
    "draw_line_until":      draw_line_until,
    # New Phase 6 Tensor Mechanics
    "int_einsum":           int_einsum,
    "einsum_affine":        einsum_affine,
    "einsum_color_map":     einsum_color_map,
    # Phase 8 Abstract Z3 Constraint Solving
    "shift_object":         shift_object,
}
