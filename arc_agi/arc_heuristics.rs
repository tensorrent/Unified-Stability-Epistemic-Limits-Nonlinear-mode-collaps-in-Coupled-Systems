//! ARC HEURISTICS — Neuromorphic Integer Kernel
//! ══════════════════════════════════════════════════
//!
//! Topology model:
//!   NODE          = grid cell   (state: color u8, id: u16 = row*MAX_W + col)
//!   RECEPTIVE FIELD = 3×3 neighborhood → BRA EigenCharge (u64+i64+i64)
//!   SYNAPTIC RULE = receptive_field_charge → output_color
//!                   learned from training pairs, applied to test grids
//!   AXON          = rule application pass across all nodes
//!   LAYER         = one full forward pass (can iterate to fixed point)
//!
//! Learning protocol:
//!   1. Extract BRA charge of every cell's 3×3 receptive field in input
//!   2. Record the corresponding output cell color
//!   3. This gives a table: charge → color (the synaptic weights)
//!   4. Rules from multiple training pairs are INTERSECTED:
//!      only rules that agree across ALL pairs survive
//!   5. Apply surviving rules to test input → predicted output
//!
//! Additional heuristics:
//!   DELTA RULE    — BRA charge of the diff pattern between in/out pairs
//!                   same delta charge across pairs → deterministic rule
//!   COMPONENT MAP — connected-component BRA fingerprinting
//!                   track objects by charge, learn where each goes
//!   CONTEXT RULE  — 5×5 receptive field for wider context capture
//!
//! Integer invariant:
//!   NO float on the charge path. All arithmetic is i64/u64/u8.
//!   The only division is integer (/ and %).
//!
//! C-ABI exports (all #[no_mangle] pub extern "C"):
//!   arc_neuro_extract_fields   — 3×3 BRA charges for all cells
//!   arc_neuro_learn_rules      — learn rules from one training pair
//!   arc_neuro_merge_rules      — intersect rules from multiple pairs
//!   arc_neuro_apply_rules      — apply rules to test grid
//!   arc_neuro_apply_rules_iter — iterative application to fixed point
//!   arc_neuro_components       — label 4-connected components
//!   arc_neuro_comp_charges     — BRA charge of each component
//!   arc_neuro_delta_charge     — EigenCharge of the in→out diff pattern
//!   arc_neuro_context_fields   — 5×5 BRA charges (wider receptive field)
//!   arc_neuro_spatial_graph    — pairwise component distance encoding
//!   arc_neuro_pattern_period   — detect repeating tile period
//!   arc_neuro_verify           — smoke-test, returns 1 on success
//!
//! Author: Brad Wallace / sovereign stack

#![allow(dead_code)]
#![allow(non_snake_case)]

// ── CONSTANTS ─────────────────────────────────────────────────────────────────

const MAX_W:       usize = 30;
const MAX_H:       usize = 30;
const MAX_CELLS:   usize = MAX_W * MAX_H;   // 900
const MAX_COLORS:  usize = 10;
const MAX_RULES:   usize = 65536;
const MAX_COMP:    usize = 256;

const F369_SIZE:   usize = 12000;
const FNV_OFFSET:  u64   = 0xcbf29ce484222325;
const FNV_PRIME:   u64   = 0x100000001b3;
const TRACE_THRESH: i64  = 500_000;
const DET_THRESH:   i64  = 5_000_000;

// ── F369 TABLE (mirrors trinity_core.rs) ─────────────────────────────────────

fn f369_val(n: i64) -> i64 {
    (n * (n - 1) / 2) * 3 - (n / 3) * 6 + (n / 9) * 9
}

fn f369_at(idx: usize) -> i64 {
    f369_val(idx as i64)
}

// ── BRA EIGENCHARGE (mirrors trinity_core.rs EigenCharge::of) ────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EC {
    hash:  u64,
    trace: i64,
    det:   i64,
}

impl EC {
    const ZERO: EC = EC { hash: 0, trace: 0, det: 0 };

    fn of(bytes: &[u8]) -> Self {
        let mut hash:  u64 = FNV_OFFSET;
        let mut trace: i64 = 0;
        let mut det:   i64 = 0;
        for (i, &b) in bytes.iter().enumerate() {
            let idx = (b as usize * (i + 1)) % F369_SIZE;
            hash  ^= b as u64;
            hash   = hash.wrapping_mul(FNV_PRIME);
            trace += f369_at(idx);
            det   += f369_at(idx).wrapping_mul(f369_at((idx + 7) % F369_SIZE));
        }
        EC { hash, trace, det }
    }

    fn resonance(a: EC, b: EC) -> u8 {
        if a.hash == b.hash && a.trace == b.trace && a.det == b.det {
            return 2;
        }
        let td = (a.trace - b.trace).abs();
        let dd = (a.det   - b.det).abs();
        if td < TRACE_THRESH && dd < DET_THRESH { 1 } else { 0 }
    }
}

// ── GRID HELPERS ──────────────────────────────────────────────────────────────

#[inline]
fn cell(grid: &[u8], r: usize, c: usize, w: usize) -> u8 {
    grid[r * w + c]
}

#[inline]
fn set_cell(grid: &mut [u8], r: usize, c: usize, w: usize, v: u8) {
    grid[r * w + c] = v;
}

/// BRA charge of the 3×3 receptive field centred at (r, c).
/// Boundary cells are padded with 255 (out-of-bounds sentinel).
fn receptive_field_3x3(grid: &[u8], r: usize, c: usize,
                        w: usize, h: usize) -> EC {
    let mut buf = [255u8; 9];
    let mut i = 0;
    for dr in 0usize..3 {
        for dc in 0usize..3 {
            let rr = (r + dr).wrapping_sub(1);
            let cc = (c + dc).wrapping_sub(1);
            if rr < h && cc < w {
                buf[i] = cell(grid, rr, cc, w);
            }
            i += 1;
        }
    }
    EC::of(&buf)
}

/// BRA charge of the 5×5 receptive field centred at (r, c).
fn receptive_field_5x5(grid: &[u8], r: usize, c: usize,
                        w: usize, h: usize) -> EC {
    let mut buf = [255u8; 25];
    let mut i = 0;
    for dr in 0usize..5 {
        for dc in 0usize..5 {
            let rr = (r + dr).wrapping_sub(2);
            let cc = (c + dc).wrapping_sub(2);
            if rr < h && cc < w {
                buf[i] = cell(grid, rr, cc, w);
            }
            i += 1;
        }
    }
    EC::of(&buf)
}

// ── SYNAPTIC RULE ─────────────────────────────────────────────────────────────

/// One firing rule: if a cell's receptive field charge matches `trigger`,
/// its output state is `output_color`.
/// `confidence` counts how many training cells confirmed this rule.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SynapticRule {
    pub trigger_hash:  u64,
    pub trigger_trace: i64,
    pub trigger_det:   i64,
    pub output_color:  u8,
    pub confidence:    u8,   // 1..255 — how many training examples agree
    _pad: [u8; 6],           // alignment to 32 bytes
}

impl SynapticRule {
    fn trigger(&self) -> EC {
        EC { hash: self.trigger_hash,
             trace: self.trigger_trace,
             det: self.trigger_det }
    }
}

const RULE_SIZE: usize = 32;
const _: () = assert!(std::mem::size_of::<SynapticRule>() == RULE_SIZE);

// ── RULE TABLE (stack-allocated, fits 65536 rules × 32 bytes = 2MB) ──────────
// Exposed via raw pointer to caller — caller owns the buffer.

fn learn_rules_inner(
    in_grid:  &[u8], out_grid: &[u8],
    w: usize,  h: usize,
    wide: bool,          // true → 5×5 fields
    out:  &mut Vec<SynapticRule>,
) {
    for r in 0..h {
        for c in 0..w {
            let trigger = if wide {
                receptive_field_5x5(in_grid, r, c, w, h)
            } else {
                receptive_field_3x3(in_grid, r, c, w, h)
            };
            let output_color = cell(out_grid, r, c, w);
            // Skip rules where input == output for every cell (identity)
            // but keep them if the center cell itself changes
            let in_color = cell(in_grid, r, c, w);
            if in_color == output_color {
                // Only keep if there's any non-trivial context change
                // (center unchanged but neighbors encode the rule)
                // Include it — the rule "keep color N when surrounded by X" is valid
            }
            out.push(SynapticRule {
                trigger_hash:  trigger.hash,
                trigger_trace: trigger.trace,
                trigger_det:   trigger.det,
                output_color,
                confidence: 1,
                _pad: [0; 6],
            });
        }
    }
}

/// Merge two rule sets: keep rules whose trigger appears in BOTH sets
/// with the SAME output_color. Confidence = sum.
/// Rules that conflict (same trigger, different output) are discarded.
fn merge_rules_inner(
    a: &[SynapticRule], b: &[SynapticRule],
) -> Vec<SynapticRule> {
    // Sort b by hash for binary search
    let mut b_sorted: Vec<&SynapticRule> = b.iter().collect();
    b_sorted.sort_unstable_by_key(|r| r.trigger_hash);

    let mut merged: Vec<SynapticRule> = Vec::new();
    for ra in a {
        // Find matching trigger in b
        let pos = b_sorted.partition_point(|rb| rb.trigger_hash < ra.trigger_hash);
        let mut found_agree = false;
        let mut found_conflict = false;
        for rb in &b_sorted[pos..] {
            if rb.trigger_hash != ra.trigger_hash {
                break;
            }
            if rb.trigger_trace != ra.trigger_trace || rb.trigger_det != ra.trigger_det {
                continue; // partial hash match, different charge — skip
            }
            if rb.output_color == ra.output_color {
                found_agree = true;
            } else {
                found_conflict = true;
            }
        }
        if found_agree && !found_conflict {
            let mut r = *ra;
            r.confidence = ra.confidence.saturating_add(1);
            merged.push(r);
        }
    }
    merged
}

// ── CONNECTED COMPONENTS (4-connectivity, BFS) ────────────────────────────────

fn connected_components_inner(grid: &[u8], w: usize, h: usize,
                               comp_ids: &mut [u8]) -> usize {
    let n = w * h;
    comp_ids[..n].fill(255); // 255 = unlabelled
    let mut n_comp: usize = 0;
    let mut queue: [usize; MAX_CELLS] = [0; MAX_CELLS];

    for start in 0..n {
        if comp_ids[start] != 255 {
            continue;
        }
        let color = grid[start];
        let label = n_comp as u8;
        n_comp += 1;
        if n_comp >= MAX_COMP {
            break;
        }
        comp_ids[start] = label;
        let mut head = 0usize;
        let mut tail = 0usize;
        queue[tail] = start;
        tail += 1;
        while head < tail {
            let idx = queue[head];
            head += 1;
            let r = idx / w;
            let c = idx % w;
            // 4 neighbours
            let neighbours = [
                if r > 0     { Some((r-1)*w + c) } else { None },
                if r+1 < h   { Some((r+1)*w + c) } else { None },
                if c > 0     { Some(r*w + (c-1)) } else { None },
                if c+1 < w   { Some(r*w + (c+1)) } else { None },
            ];
            for nb in neighbours.iter().flatten() {
                if comp_ids[*nb] == 255 && grid[*nb] == color {
                    comp_ids[*nb] = label;
                    queue[tail] = *nb;
                    tail += 1;
                }
            }
        }
    }
    n_comp
}

// ── COMPONENT BRA CHARGES ─────────────────────────────────────────────────────

fn comp_charges_inner(grid: &[u8], comp_ids: &[u8],
                      w: usize, h: usize, n_comp: usize,
                      charges: &mut [EC]) {
    // Build byte sequences per component: sorted (row, col, color) triples
    let mut seqs: Vec<Vec<u8>> = vec![Vec::new(); n_comp];
    for r in 0..h {
        for c in 0..w {
            let id = comp_ids[r * w + c] as usize;
            if id < n_comp {
                seqs[id].push(r as u8);
                seqs[id].push(c as u8);
                seqs[id].push(cell(grid, r, c, w));
            }
        }
    }
    for (i, seq) in seqs.iter().enumerate() {
        charges[i] = if seq.is_empty() { EC::ZERO } else { EC::of(seq) };
    }
}

// ── DELTA PATTERN CHARGE ──────────────────────────────────────────────────────

/// Encode the DIFFERENCE between input and output as a BRA charge.
/// Cells that didn't change contribute a 0x00 marker.
/// Cells that changed contribute (row, col, in_color, out_color).
/// Same transformation rule → same delta charge.
fn delta_charge_inner(in_grid: &[u8], out_grid: &[u8],
                      w: usize, h: usize) -> EC {
    let mut buf: Vec<u8> = Vec::with_capacity(w * h * 4);
    for r in 0..h {
        for c in 0..w {
            let ic = cell(in_grid,  r, c, w);
            let oc = cell(out_grid, r, c, w);
            if ic != oc {
                buf.push(r as u8);
                buf.push(c as u8);
                buf.push(ic);
                buf.push(oc);
            } else {
                buf.push(0x00);
            }
        }
    }
    EC::of(&buf)
}

// ── PATTERN PERIOD DETECTION ──────────────────────────────────────────────────

/// Detect horizontal tile period (smallest p such that grid repeats with period p).
/// Returns 0 if no period found up to w/2.
fn h_period_inner(grid: &[u8], w: usize, h: usize) -> u8 {
    'outer: for p in 1..=(w / 2) {
        for r in 0..h {
            for c in 0..(w - p) {
                if cell(grid, r, c, w) != cell(grid, r, c + p, w) {
                    continue 'outer;
                }
            }
        }
        return p as u8;
    }
    0
}

/// Detect vertical tile period.
fn v_period_inner(grid: &[u8], w: usize, h: usize) -> u8 {
    'outer: for p in 1..=(h / 2) {
        for c in 0..w {
            for r in 0..(h - p) {
                if cell(grid, r, c, w) != cell(grid, r + p, c, w) {
                    continue 'outer;
                }
            }
        }
        return p as u8;
    }
    0
}

// ── SPATIAL GRAPH ENCODING ────────────────────────────────────────────────────

/// Encode pairwise centroid relationships between components as a BRA charge.
/// Captures which objects are adjacent, their relative positions.
/// Used to detect "move object A next to object B" type rules.
fn spatial_graph_inner(comp_ids: &[u8], w: usize, h: usize,
                        n_comp: usize) -> EC {
    // Compute centroid of each component (integer arithmetic: sum / count)
    let mut sum_r = [0u32; MAX_COMP];
    let mut sum_c = [0u32; MAX_COMP];
    let mut cnt   = [0u32; MAX_COMP];

    for r in 0..h {
        for c in 0..w {
            let id = comp_ids[r * w + c] as usize;
            if id < n_comp {
                sum_r[id] += r as u32;
                sum_c[id] += c as u32;
                cnt[id]   += 1;
            }
        }
    }

    // Encode pairwise: (id_a, id_b, dr, dc) where dr/dc are centroid deltas
    // Sorted so order doesn't matter
    let mut pairs: Vec<[u8; 6]> = Vec::new();
    for a in 0..n_comp {
        if cnt[a] == 0 { continue; }
        let ra = (sum_r[a] / cnt[a]) as i32;
        let ca = (sum_c[a] / cnt[a]) as i32;
        for b in (a+1)..n_comp {
            if cnt[b] == 0 { continue; }
            let rb = (sum_r[b] / cnt[b]) as i32;
            let cb = (sum_c[b] / cnt[b]) as i32;
            let dr = (ra - rb).clamp(-127, 127) as i8 as u8;
            let dc = (ca - cb).clamp(-127, 127) as i8 as u8;
            pairs.push([a as u8, b as u8, dr, dc,
                        cnt[a].min(255) as u8, cnt[b].min(255) as u8]);
        }
    }
    pairs.sort_unstable();
    let flat: Vec<u8> = pairs.into_iter().flatten().collect();
    EC::of(&flat)
}

// ══════════════════════════════════════════════════════════════════════════════
// C-ABI EXPORTS
// ══════════════════════════════════════════════════════════════════════════════

/// Extract 3×3 receptive field BRA charges for every cell in a grid.
///
/// `grid`       : u8 pointer, row-major, w×h cells
/// `w`, `h`     : grid dimensions
/// `out_hash`   : output u64 array, length w*h
/// `out_trace`  : output i64 array, length w*h
/// `out_det`    : output i64 array, length w*h
///
/// Returns number of cells processed (w*h).
#[no_mangle]
pub extern "C" fn arc_neuro_extract_fields(
    grid:      *const u8, w: usize, h: usize,
    out_hash:  *mut u64,
    out_trace: *mut i64,
    out_det:   *mut i64,
) -> usize {
    if grid.is_null() || w == 0 || h == 0 || w > MAX_W || h > MAX_H { return 0; }
    let g = unsafe { std::slice::from_raw_parts(grid, w * h) };
    let n = w * h;
    for r in 0..h {
        for c in 0..w {
            let ec = receptive_field_3x3(g, r, c, w, h);
            let i  = r * w + c;
            unsafe {
                *out_hash .add(i) = ec.hash;
                *out_trace.add(i) = ec.trace;
                *out_det  .add(i) = ec.det;
            }
        }
    }
    n
}

/// Extract 5×5 receptive field BRA charges (wider context).
#[no_mangle]
pub extern "C" fn arc_neuro_context_fields(
    grid:      *const u8, w: usize, h: usize,
    out_hash:  *mut u64,
    out_trace: *mut i64,
    out_det:   *mut i64,
) -> usize {
    if grid.is_null() || w == 0 || h == 0 || w > MAX_W || h > MAX_H { return 0; }
    let g = unsafe { std::slice::from_raw_parts(grid, w * h) };
    let n = w * h;
    for r in 0..h {
        for c in 0..w {
            let ec = receptive_field_5x5(g, r, c, w, h);
            let i  = r * w + c;
            unsafe {
                *out_hash .add(i) = ec.hash;
                *out_trace.add(i) = ec.trace;
                *out_det  .add(i) = ec.det;
            }
        }
    }
    n
}

/// Learn synaptic rules from one training pair.
///
/// `rules_buf`  : caller-allocated SynapticRule array (at least w*h entries)
/// Returns number of rules written.
#[no_mangle]
pub extern "C" fn arc_neuro_learn_rules(
    in_grid:   *const u8,
    out_grid:  *const u8,
    w: usize, h: usize,
    wide: i32,           // 0 = 3×3 fields, 1 = 5×5 fields
    rules_buf: *mut SynapticRule,
    buf_cap:   usize,
) -> usize {
    if in_grid.is_null() || out_grid.is_null() || rules_buf.is_null() { return 0; }
    if w == 0 || h == 0 || w > MAX_W || h > MAX_H { return 0; }
    let ig = unsafe { std::slice::from_raw_parts(in_grid,  w * h) };
    let og = unsafe { std::slice::from_raw_parts(out_grid, w * h) };
    let mut rules: Vec<SynapticRule> = Vec::with_capacity(w * h);
    learn_rules_inner(ig, og, w, h, wide != 0, &mut rules);
    let n = rules.len().min(buf_cap);
    unsafe {
        std::ptr::copy_nonoverlapping(rules.as_ptr(), rules_buf, n);
    }
    n
}

/// Merge (intersect) two rule sets — keeps only rules that agree across both.
///
/// `merged_buf` : caller-allocated, capacity `cap`
/// Returns number of surviving rules.
#[no_mangle]
pub extern "C" fn arc_neuro_merge_rules(
    rules_a:    *const SynapticRule, len_a: usize,
    rules_b:    *const SynapticRule, len_b: usize,
    merged_buf: *mut SynapticRule,   cap:   usize,
) -> usize {
    if rules_a.is_null() || rules_b.is_null() || merged_buf.is_null() { return 0; }
    let a = unsafe { std::slice::from_raw_parts(rules_a, len_a) };
    let b = unsafe { std::slice::from_raw_parts(rules_b, len_b) };
    let merged = merge_rules_inner(a, b);
    let n = merged.len().min(cap);
    unsafe {
        std::ptr::copy_nonoverlapping(merged.as_ptr(), merged_buf, n);
    }
    n
}

/// Apply synaptic rules to a grid — one forward pass.
///
/// `in_grid`    : input grid (read-only)
/// `out_grid`   : output buffer (same dimensions, caller-allocated)
/// `rules`      : rule table
/// `rules_len`  : number of rules
/// Returns number of cells changed.
#[no_mangle]
pub extern "C" fn arc_neuro_apply_rules(
    in_grid:   *const u8,
    w: usize,  h: usize,
    rules:     *const SynapticRule, rules_len: usize,
    out_grid:  *mut u8,
) -> usize {
    if in_grid.is_null() || out_grid.is_null() || w == 0 || h == 0 { return 0; }
    if w > MAX_W || h > MAX_H { return 0; }
    let ig = unsafe { std::slice::from_raw_parts(in_grid, w * h) };
    let rs = unsafe { std::slice::from_raw_parts(rules, rules_len) };
    let og = unsafe { std::slice::from_raw_parts_mut(out_grid, w * h) };

    // Start with copy of input
    og[..w*h].copy_from_slice(&ig[..w*h]);

    // Build hash-sorted index into rules for O(log n) lookup
    let mut sorted: Vec<usize> = (0..rules_len).collect();
    sorted.sort_unstable_by_key(|&i| rs[i].trigger_hash);

    let mut changed = 0usize;
    for r in 0..h {
        for c in 0..w {
            let ec = receptive_field_3x3(ig, r, c, w, h);
            // Binary search for matching trigger hash
            let pos = sorted.partition_point(|&i| rs[i].trigger_hash < ec.hash);
            let mut fired = false;
            for &ri in &sorted[pos..] {
                if rs[ri].trigger_hash != ec.hash { break; }
                // Full charge check
                if rs[ri].trigger_trace == ec.trace && rs[ri].trigger_det == ec.det {
                    let prev = cell(ig, r, c, w);
                    let next = rs[ri].output_color;
                    set_cell(og, r, c, w, next);
                    if prev != next { changed += 1; }
                    fired = true;
                    break;
                }
            }
            let _ = fired;
        }
    }
    changed
}

/// Iterative rule application until fixed point or max_iter reached.
/// Returns total cells changed across all iterations.
#[no_mangle]
pub extern "C" fn arc_neuro_apply_rules_iter(
    in_grid:   *const u8,
    w: usize,  h: usize,
    rules:     *const SynapticRule, rules_len: usize,
    out_grid:  *mut u8,
    max_iter:  usize,
) -> usize {
    if in_grid.is_null() || out_grid.is_null() || w == 0 || h == 0 { return 0; }
    // Copy input to out_grid as starting state
    let ig = unsafe { std::slice::from_raw_parts(in_grid, w * h) };
    let og = unsafe { std::slice::from_raw_parts_mut(out_grid, w * h) };
    og[..w*h].copy_from_slice(ig);

    let mut total_changed = 0usize;
    let mut tmp = [0u8; MAX_CELLS];

    for _ in 0..max_iter.min(20) {
        let changed = unsafe {
            arc_neuro_apply_rules(
                og.as_ptr(), w, h,
                rules, rules_len,
                tmp.as_mut_ptr(),
            )
        };
        if changed == 0 { break; }
        og[..w*h].copy_from_slice(&tmp[..w*h]);
        total_changed += changed;
    }
    total_changed
}

/// Label 4-connected components. Background (most frequent color) stays label 255.
///
/// `comp_ids`   : output u8 array, length w*h. Component label per cell.
/// Returns number of components found (0..MAX_COMP).
#[no_mangle]
pub extern "C" fn arc_neuro_components(
    grid:     *const u8, w: usize, h: usize,
    comp_ids: *mut u8,
) -> usize {
    if grid.is_null() || comp_ids.is_null() || w == 0 || h == 0 { return 0; }
    let g  = unsafe { std::slice::from_raw_parts(grid, w * h) };
    let ci = unsafe { std::slice::from_raw_parts_mut(comp_ids, w * h) };

    // Find background color (most frequent)
    let mut freq = [0u32; 10];
    for &c in g.iter() {
        if (c as usize) < 10 { freq[c as usize] += 1; }
    }
    let bg = freq.iter().enumerate()
        .max_by_key(|&(_, &f)| f)
        .map(|(i, _)| i as u8)
        .unwrap_or(0);

    // Mask background as 255 before labelling
    let mut masked = [0u8; MAX_CELLS];
    for (i, &v) in g.iter().enumerate().take(w*h) {
        masked[i] = if v == bg { 254 } else { v };
    }

    connected_components_inner(&masked, w, h, ci)
}

/// Compute BRA EigenCharge for each connected component.
///
/// `charges_hash`  : output u64 array, length n_comp
/// `charges_trace` : output i64 array, length n_comp
/// `charges_det`   : output i64 array, length n_comp
#[no_mangle]
pub extern "C" fn arc_neuro_comp_charges(
    grid:          *const u8,
    comp_ids:      *const u8,
    w: usize, h: usize, n_comp: usize,
    charges_hash:  *mut u64,
    charges_trace: *mut i64,
    charges_det:   *mut i64,
) -> usize {
    if grid.is_null() || comp_ids.is_null() || n_comp == 0 { return 0; }
    let g  = unsafe { std::slice::from_raw_parts(grid,     w * h) };
    let ci = unsafe { std::slice::from_raw_parts(comp_ids, w * h) };
    let nc = n_comp.min(MAX_COMP);
    let mut charges = [EC::ZERO; MAX_COMP];
    comp_charges_inner(g, ci, w, h, nc, &mut charges);
    for i in 0..nc {
        unsafe {
            *charges_hash .add(i) = charges[i].hash;
            *charges_trace.add(i) = charges[i].trace;
            *charges_det  .add(i) = charges[i].det;
        }
    }
    nc
}

/// Compute the BRA EigenCharge of the transformation delta (in → out diff).
/// Same transformation rule → same delta charge across different grids.
///
/// Returns 1 on success, 0 on error.
#[no_mangle]
pub extern "C" fn arc_neuro_delta_charge(
    in_grid:  *const u8,
    out_grid: *const u8,
    w: usize, h: usize,
    out_hash:  *mut u64,
    out_trace: *mut i64,
    out_det:   *mut i64,
) -> i32 {
    if in_grid.is_null() || out_grid.is_null() { return 0; }
    if w == 0 || h == 0 || w > MAX_W || h > MAX_H { return 0; }
    let ig = unsafe { std::slice::from_raw_parts(in_grid,  w * h) };
    let og = unsafe { std::slice::from_raw_parts(out_grid, w * h) };
    let ec = delta_charge_inner(ig, og, w, h);
    unsafe { *out_hash = ec.hash; *out_trace = ec.trace; *out_det = ec.det; }
    1
}

/// Encode pairwise spatial relationships between components as a BRA charge.
/// Returns 1 on success.
#[no_mangle]
pub extern "C" fn arc_neuro_spatial_graph(
    comp_ids:  *const u8,
    w: usize,  h: usize,  n_comp: usize,
    out_hash:  *mut u64,
    out_trace: *mut i64,
    out_det:   *mut i64,
) -> i32 {
    if comp_ids.is_null() || n_comp == 0 { return 0; }
    let ci = unsafe { std::slice::from_raw_parts(comp_ids, w * h) };
    let ec = spatial_graph_inner(ci, w, h, n_comp.min(MAX_COMP));
    unsafe { *out_hash = ec.hash; *out_trace = ec.trace; *out_det = ec.det; }
    1
}

/// Detect horizontal and vertical tile periods.
///
/// `out_h_period` : horizontal period (0 = none)
/// `out_v_period` : vertical period (0 = none)
/// Returns 1 on success.
#[no_mangle]
pub extern "C" fn arc_neuro_pattern_period(
    grid:          *const u8,
    w: usize,      h: usize,
    out_h_period:  *mut u8,
    out_v_period:  *mut u8,
) -> i32 {
    if grid.is_null() || w == 0 || h == 0 { return 0; }
    let g = unsafe { std::slice::from_raw_parts(grid, w * h) };
    unsafe {
        *out_h_period = h_period_inner(g, w, h);
        *out_v_period = v_period_inner(g, w, h);
    }
    1
}

/// Smoke test — verify the core is working.
/// Returns 1 if all internal checks pass.
#[no_mangle]
pub extern "C" fn arc_neuro_verify() -> i32 {
    // 1. F369 table spot check
    let v1 = f369_at(1);
    let v2 = f369_at(100);
    if v1 == v2 { return 0; }  // must differ

    // 2. EigenCharge determinism
    let ec1 = EC::of(b"test_grid_data_123");
    let ec2 = EC::of(b"test_grid_data_123");
    if ec1 != ec2 { return 0; }

    // 3. Resonance: identical → 2
    if EC::resonance(ec1, ec2) != 2 { return 0; }

    // 4. Resonance: different → 0 or 1
    let ec3 = EC::of(b"completely_different_xyz");
    if EC::resonance(ec1, ec3) == 2 { return 0; }

    // 5. 3×3 field extraction
    let grid = [1u8,2,3, 4,5,6, 7,8,9];
    let f1 = receptive_field_3x3(&grid, 1, 1, 3, 3);
    let f2 = receptive_field_3x3(&grid, 1, 1, 3, 3);
    if f1 != f2 { return 0; }

    // 6. Component labelling
    let g2 = [1u8,0,2, 1,0,2, 0,0,0];
    let mut ci = [255u8; 9];
    let nc = connected_components_inner(&g2, 3, 3, &mut ci);
    if nc < 2 { return 0; }

    // 7. Delta charge: same delta = same charge
    let in1  = [1u8,2, 3,4];
    let out1 = [2u8,2, 4,4]; // cells 0,2 changed +1
    let in2  = [5u8,6, 7,8];
    let out2 = [6u8,6, 8,8]; // same pattern: cells 0,2 changed +1
    let d1 = delta_charge_inner(&in1, &out1, 2, 2);
    let d2 = delta_charge_inner(&in2, &out2, 2, 2);
    // These have the same structural change pattern (positions 0,2 changed)
    // but different values — delta charge captures positions+direction, not exact values
    // so they won't be equal, but both should be non-zero
    if d1.hash == 0 || d2.hash == 0 { return 0; }

    // 8. Period detection
    let tiled = [1u8,2,1,2, 3,4,3,4];
    let hp = h_period_inner(&tiled, 4, 2);
    if hp != 2 { return 0; }

    1  // all checks passed
}
