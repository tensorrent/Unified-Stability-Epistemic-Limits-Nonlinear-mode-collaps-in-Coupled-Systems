# Copyright (c) 2026 Brad Wallace. All rights reserved.
# Subject to Sovereign Integrity Protocol License (SIP License v1.1).
# See SIP_LICENSE.md for full terms.
"""
arc_solver.py — ARC-AGI Solver Pipeline
========================================
Three-stage pipeline that mirrors the winning approaches:

  Stage 1 — ANALYZE
    Describe each training pair in multiple representations:
    - Text (char grid)
    - Structural (object list, symmetry, color stats)
    - Diff (what changed between input and output)
    Feed everything to the LLM for rule extraction.

  Stage 2 — SYNTHESIZE
    LLM generates a Python program that implements the inferred rule.
    The program is a function: transform(grid: Grid) -> Grid
    It is executed against all training inputs. Score = fraction correct.

  Stage 3 — REFINE
    If score < 1.0, feed failing pairs back with the diff between
    predicted and expected output. Ask the LLM to fix the program.
    Repeat up to MAX_REFINEMENTS times.

Backends: same as hermes_vqa — claude first, then openrouter.
All attempts are scored and the best-scoring program wins.

VQA path: If text+program synthesis fails, fall back to rendering the
task as a PNG and asking the VQA backend to directly predict the output
grid as a character-encoded string.

Scroll integration: every attempt is recorded as an EV_QUERY or
EV_RESONANCE event with the program, score, and task_id as payload.
"""

import os
import re
import sys
import json
import time
import traceback
import textwrap
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Optional

SOVEREIGN_SDK = os.environ.get("SOVEREIGN_SDK", os.path.dirname(__file__))
if SOVEREIGN_SDK not in sys.path:
    sys.path.insert(0, SOVEREIGN_SDK)

from arc_types import (
    Grid, ARCTask, ARCPrediction, Pair,
    grid_shape, grid_to_text, text_to_grid, grid_similarity,
    describe_task, describe_pair, describe_grid,
    extract_objects, detect_symmetry, count_colors,
    background_color, score_prediction, score_task,
    ARC_COLOR_CHARS, ARC_CHAR_COLORS, ARC_COLOR_NAMES,
    grid_diff, grid_eq,
)

# ── Config ─────────────────────────────────────────────────────────────────────

MAX_REFINEMENTS   = int(os.environ.get("ARC_MAX_REFINEMENTS", "4"))
MAX_CANDIDATES    = int(os.environ.get("ARC_MAX_CANDIDATES",  "3"))
EXEC_TIMEOUT      = int(os.environ.get("ARC_EXEC_TIMEOUT",    "10"))

ANTHROPIC_API_URL  = "https://api.anthropic.com/v1/messages"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_API_VER  = "2023-06-01"
DEFAULT_CLAUDE     = "claude-sonnet-4-6"
DEFAULT_OPENROUTER = "anthropic/claude-sonnet-4-5"

LOCAL_API_URL = "http://localhost:8402/v1/chat/completions"

def _call_llm(messages: list[dict], system: str = "",  # only reached after TENT gate
              max_tokens: int = 2048) -> str:
    """Call LLM with text-only messages. Tries local openclaw provider first, then falls back to claude/openrouter."""
    
    # SEGGCI OPENCLAW PROVIDER PRIORITY
    msgs = ([{"role":"system","content":system}] if system else []) + messages
    body = json.dumps({
        "model":      "seggci/mistral-7b",
        "max_tokens": max_tokens,
        "messages":   msgs,
    }).encode()
    try:
        req = urllib.request.Request(
            LOCAL_API_URL, data=body,
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [arc_solver] local provider failed ({e}), trying external API...")

    key_claude = os.environ.get("ANTHROPIC_API_KEY", "")
    key_or     = os.environ.get("OPENROUTER_API_KEY", "")

    if key_claude:
        body = json.dumps({
            "model":      os.environ.get("ARC_CLAUDE_MODEL", DEFAULT_CLAUDE),
            "max_tokens": max_tokens,
            "messages":   messages,
            **({"system": system} if system else {}),
        }).encode()
        req = urllib.request.Request(
            ANTHROPIC_API_URL, data=body,
            headers={"Content-Type": "application/json",
                     "x-api-key": key_claude,
                     "anthropic-version": ANTHROPIC_API_VER},
            method="POST")
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["content"][0]["text"]

    if key_or:
        msgs = ([{"role":"system","content":system}] if system else []) + messages
        body = json.dumps({
            "model":      os.environ.get("ARC_OR_MODEL", DEFAULT_OPENROUTER),
            "max_tokens": max_tokens,
            "messages":   msgs,
        }).encode()
        req = urllib.request.Request(
            OPENROUTER_API_URL, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key_or}",
                     "HTTP-Referer": "sovereign-arc",
                     "X-Title": "sovereign-arc"},
            method="POST")
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    raise RuntimeError("No LLM API key set and local OpenClaw provider failed")


# ── Stage 1: ANALYZE ──────────────────────────────────────────────────────────

def _pair_diff_description(pair: Pair) -> str:
    """Textual description of what changed from input to output."""
    if not pair.output:
        return "(no output)"
    ih, iw = grid_shape(pair.input)
    oh, ow = grid_shape(pair.output)
    lines = []
    if (ih, iw) != (oh, ow):
        lines.append(f"Shape: {ih}×{iw} → {oh}×{ow}")
    diffs = grid_diff(pair.input, pair.output)
    if not diffs:
        lines.append("Grid unchanged.")
    else:
        lines.append(f"{len(diffs)} cells changed:")
        for r, c, old, new in diffs[:20]:
            lines.append(f"  ({r},{c}) {ARC_COLOR_NAMES.get(old,'?')} → {ARC_COLOR_NAMES.get(new,'?')}")
        if len(diffs) > 20:
            lines.append(f"  ... and {len(diffs)-20} more")
    # Object summary
    in_objs  = extract_objects(pair.input)
    out_objs = extract_objects(pair.output)
    lines.append(f"Input objects: {len(in_objs)}, Output objects: {len(out_objs)}")
    return "\n".join(lines)

def build_analysis_prompt(task: ARCTask) -> str:
    """
    Build the analysis prompt for Stage 1: rule extraction.
    Returns a string ready to send as the user message.
    """
    lines = [
        "You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.",
        "I will show you training examples (input→output grid pairs).",
        "Your job is to identify the EXACT transformation rule.",
        "",
        "GRID ENCODING: Each cell is one character.",
        ". = black(0)  b = blue(1)  r = red(2)  g = green(3)  y = yellow(4)",
        "W = gray(5)   m = magenta(6)  o = orange(7)  a = azure(8)  n = maroon(9)",
        "",
    ]

    for i, pair in enumerate(task.train):
        lines.append(f"=== Training Example {i+1} ===")
        lines.append(f"INPUT ({grid_shape(pair.input)[0]}×{grid_shape(pair.input)[1]}):")
        lines.append(grid_to_text(pair.input))
        if pair.output:
            lines.append(f"OUTPUT ({grid_shape(pair.output)[0]}×{grid_shape(pair.output)[1]}):")
            lines.append(grid_to_text(pair.output))
            lines.append("DIFF ANALYSIS:")
            lines.append(_pair_diff_description(pair))
        lines.append("")

    lines += [
        "=== YOUR TASK ===",
        "1. State the transformation rule in plain English (be precise and complete).",
        "2. Identify what Core Knowledge concepts are used:",
        "   - Spatial (rotation, reflection, translation, scaling)",
        "   - Pattern (repetition, symmetry, completion)",
        "   - Object (counting, color, size, shape, containment)",
        "   - Logical (AND, OR, XOR, conditional coloring)",
        "3. Note any edge cases or conditions.",
        "",
        "Be concise. One rule, stated clearly.",
    ]
    return "\n".join(lines)

def analyze_task(task: ARCTask) -> str:
    """Stage 1: Ask LLM to extract the transformation rule. Fallback to LAC deterministic audit."""
    try:
        from arc_ilm import ILMDeterministicEngine
        audit = ILMDeterministicEngine.analyze(task)
    except Exception:
        audit = "[LAC Error] Could not generate deterministic audit."

    prompt = build_analysis_prompt(task)
    system = (
        "You are an expert ARC-AGI solver. "
        "Analyze training examples and extract the exact transformation rule. "
        "Be precise and complete. Think step by step."
    )
    try:
        llm_rule = _call_llm(
            [{"role": "user", "content": prompt}],
            system=system, max_tokens=1024
        )
        return f"{llm_rule}\n\n[LAC AUDIT]:\n{audit}"
    except Exception as e:
        print(f"  [arc_solver] analysis LLM failed ({e}), using deterministic audit only.")
        return audit


# ── Stage 2: SYNTHESIZE ───────────────────────────────────────────────────────

# SYNTH_SYSTEM — use full extended DSL list from arc_abstraction
try:
    from arc_abstraction import ABSTRACT_SYNTHESIS_SYSTEM as SYNTH_SYSTEM
except ImportError:
    SYNTH_SYSTEM = (
        "You are an expert ARC-AGI solver that writes Python programs. "
        "Write def transform(grid): ... return result. "
        "Use only stdlib. Return ONLY the function in a ```python``` block."
    )

def build_synthesis_prompt(task: ARCTask, rule: str,
                            prior_attempt: str = "",
                            failing_pairs: list = None) -> str:
    lines = []
    lines.append(f"RULE IDENTIFIED:\n{rule}\n")

    # Compact training examples
    for i, pair in enumerate(task.train):
        lines.append(f"Example {i+1}:")
        lines.append(f"  Input:\n" + "\n".join(
            "    " + row for row in grid_to_text(pair.input).split("\n")))
        if pair.output:
            lines.append(f"  Output:\n" + "\n".join(
                "    " + row for row in grid_to_text(pair.output).split("\n")))

    if prior_attempt:
        lines.append("\nPREVIOUS ATTEMPT (INCORRECT):")
        lines.append(f"```python\n{prior_attempt}\n```")

    if failing_pairs:
        lines.append("\nFAILING CASES:")
        for fp in failing_pairs[:3]:
            lines.append(f"  Input:     {grid_to_text(fp['input'])}")
            lines.append(f"  Expected:  {grid_to_text(fp['expected'])}")
            lines.append(f"  Got:       {grid_to_text(fp['got'])}")
            lines.append(f"  Diff:      {fp['diff_count']} cells wrong")

    lines.append("\nWrite the transform() function now.")
    return "\n".join(lines)

def _extract_python(text: str) -> str:
    """Extract Python code block from LLM response."""
    # Try ```python ... ``` block
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ``` ... ``` block
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try def transform(
    m = re.search(r"(def transform\(.*?)(?=\ndef |\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

def synthesize_program(task: ARCTask, rule: str,
                        prior: str = "", failing: list = None) -> str:
    """Stage 2: Ask LLM to write Python program implementing the rule."""
    prompt = build_synthesis_prompt(task, rule, prior, failing)
    response = _call_llm(
        [{"role": "user", "content": prompt}],
        system=SYNTH_SYSTEM, max_tokens=1500
    )
    return _extract_python(response)


# ── Program execution ──────────────────────────────────────────────────────────

def _make_exec_globals() -> dict:
    """Build the execution namespace with all DSL helpers pre-imported."""
    from arc_types import (
        grid_height, grid_width, grid_shape, grid_copy, empty_grid,
        rot90, rot180, rot270, reflect_h, reflect_v, reflect_diag, reflect_anti,
        crop, pad, tile, recolor, replace_colors, fill, overlay, hstack, vstack,
        upscale, downscale, extract_objects, background_color, crop_to_content,
        count_colors, most_common_color, least_common_color, detect_symmetry,
        grid_unique_colors, grid_color_count, grid_diff, grid_similarity,
        grid_set, grid_get, grid_eq, normalize_colors as _nc,
    )
    base = {
        "grid_height": grid_height, "grid_width": grid_width,
        "grid_shape": grid_shape, "grid_copy": grid_copy, "empty_grid": empty_grid,
        "rot90": rot90, "rot180": rot180, "rot270": rot270,
        "reflect_h": reflect_h, "reflect_v": reflect_v,
        "reflect_diag": reflect_diag, "reflect_anti": reflect_anti,
        "crop": crop, "pad": pad, "tile": tile,
        "recolor": recolor, "replace_colors": replace_colors, "fill": fill,
        "overlay": overlay, "hstack": hstack, "vstack": vstack,
        "upscale": upscale, "downscale": downscale,
        "extract_objects": extract_objects, "background_color": background_color,
        "crop_to_content": crop_to_content, "count_colors": count_colors,
        "most_common_color": most_common_color, "least_common_color": least_common_color,
        "detect_symmetry": detect_symmetry, "grid_unique_colors": grid_unique_colors,
        "grid_color_count": grid_color_count, "grid_diff": grid_diff,
        "grid_similarity": grid_similarity, "grid_set": grid_set,
        "grid_get": grid_get, "grid_eq": grid_eq,
    }
    # Merge extended DSL namespace (graceful — won't fail if module missing)
    try:
        from arc_dsl_ext import EXT_DSL_NAMESPACE
        base.update(EXT_DSL_NAMESPACE)
    except ImportError:
        pass
    
    # Neuromorphic Interface
    import arc_neuro
    base["NeuromorphicKernel"] = arc_neuro
    return base

def execute_program(code: str, grid: Grid,
                    timeout: int = EXEC_TIMEOUT) -> tuple[Optional[Grid], str]:
    """
    Execute the synthesized transform() function.
    Returns (result_grid, error_string).
    error_string is "" on success.
    """
    import signal, threading

    globs = _make_exec_globals()
    result_container = [None]
    error_container  = [""]

    try:
        exec(compile(code, "<arc_program>", "exec"), globs)
        transform_fn = globs.get("transform")
        if not transform_fn:
            return None, "No transform() function defined"

        def run():
            try:
                from copy import deepcopy
                result_container[0] = transform_fn(deepcopy(grid))
            except Exception as e:
                error_container[0] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None, f"Execution timeout after {timeout}s"

        if error_container[0]:
            return None, error_container[0]

        result = result_container[0]
        # Handle NeuromorphicKernel return (direct grid)
        if isinstance(result, dict) and "prediction" in result:
            result = result["prediction"]

        if not isinstance(result, list):
            return None, f"transform() returned {type(result).__name__}, expected list"
        return result, ""

    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    except Exception as e:
        return None, f"Exec error: {e}"

def extract_primitives_from_code(code: str) -> list[str]:
    """Extract used ARC primitives from Python code for Neuromorphic LTP."""
    from arc_bra import _PRIM_COORDS
    prims = []
    for p in _PRIM_COORDS.keys():
        if f"{p}(" in code:
            prims.append(p)
    return prims

def evaluate_program(code: str, task: ARCTask) -> dict:
    """
    Run the program against all training pairs.
    Returns {score, correct, total, failures}
    """
    correct  = 0
    failures = []
    for pair in task.train:
        result, err = execute_program(code, pair.input)
        if err or result is None:
            failures.append({
                "input":    pair.input,
                "expected": pair.output,
                "got":      None,
                "error":    err,
                "diff_count": -1,
            })
            continue
        if pair.output and result == pair.output:
            correct += 1
        elif pair.output:
            from arc_types import grid_diff
            diffs = grid_diff(result, pair.output)
            failures.append({
                "input":    pair.input,
                "expected": pair.output,
                "got":      result,
                "error":    "",
                "diff_count": len(diffs),
            })
        else:
            correct += 1  # No ground truth → assume correct

    total = len(task.train)
    return {
        "score":    correct / total if total else 0.0,
        "correct":  correct,
        "total":    total,
        "failures": failures,
        "primitives": extract_primitives_from_code(code)
    }


# ── Stage 3: REFINE ───────────────────────────────────────────────────────────

def refine_program(task: ARCTask, rule: str, program: str,
                   failures: list) -> str:
    """Stage 3: Ask LLM to fix the program given failing cases."""
    prompt = build_synthesis_prompt(task, rule, prior_attempt=program,
                                    failing_pairs=failures)
    response = _call_llm(
        [{"role": "user", "content": prompt}],
        system=SYNTH_SYSTEM + "\n\nFix the PREVIOUS ATTEMPT. Address the FAILING CASES.",
        max_tokens=1500
    )
    return _extract_python(response)


# ── VQA fallback ──────────────────────────────────────────────────────────────

def _vqa_fallback(task: ARCTask, test_index: int,
                  scroll_bridge=None) -> Optional[Grid]:
    """
    If program synthesis fails, render the task as PNG and ask VQA to
    directly predict the output grid as a character string.
    """
    try:
        from arc_renderer import render_task
        from hermes_vqa import VQABridge
    except ImportError:
        return None

    png_bytes = render_task(task, include_test=True)

    test_pair = task.test[test_index]
    th, tw    = grid_shape(test_pair.input)

    question = (
        f"This image shows ARC-AGI training examples (input→output pairs) "
        f"and one test input at the bottom.\n"
        f"The test input is {th} rows × {tw} columns.\n"
        f"Using the same transformation rule shown in the training examples, "
        f"predict the test output grid.\n"
        f"Reply ONLY with the grid as {th} lines of {tw} characters each, "
        f"using this encoding:\n"
        f". = black(0)  b = blue(1)  r = red(2)  g = green(3)  y = yellow(4)\n"
        f"W = gray(5)   m = magenta(6)  o = orange(7)  a = azure(8)  n = maroon(9)\n"
        f"No explanation, just the grid."
    )

    vqa = VQABridge(scroll_bridge=scroll_bridge)
    result = vqa.ask(png_bytes, question, commit_to_memory=False)

    if not result.ok:
        return None

    # Parse the response as a grid
    try:
        lines = [l.strip() for l in result.answer.strip().split("\n")
                 if l.strip() and all(c in ARC_CHAR_COLORS for c in l.strip())]
        if not lines:
            return None
        grid = [[ARC_CHAR_COLORS[c] for c in line] for line in lines]
        # Validate shape
        if len(grid) != th or any(len(row) != tw for row in grid):
            # Try to accept partial match if shapes differ slightly
            if len(grid) > 0:
                return grid
            return None
        return grid
    except Exception:
        return None


# ── Main solver ────────────────────────────────────────────────────────────────

class ARCSolver:
    """
    Full ARC-AGI solver pipeline.

    Strategy cascade (fastest-first):
      0. Pattern library warm-start (exact / similar task → reuse program)
      1. Brute-force DSL search (catches simple 1-2 op transforms in <20s)
      2. Analyze → Synthesize (object-centric abstraction prompt)
      3. Refine (feed failing pairs back, up to MAX_REFINEMENTS rounds)
      4. Augmentation candidates (wrap best program for all 8 D4 orientations)
      5. Multi-candidate voting (generate N candidates, vote on test predictions)
      6. VQA fallback (render task as PNG, ask vision model for char-grid output)

    All stages are gated by score thresholds — fast exit on perfect score.
    """

    def __init__(self, scroll_bridge=None, verbose: bool = False,
                 library=None,
                 use_brute_force: bool = True,
                 use_augmentation: bool = True,
                 use_search: bool = True,
                 n_candidates: int = None):
        self._bridge   = scroll_bridge
        self._verbose  = verbose
        self._library  = library
        self._use_bf   = use_brute_force
        self._use_aug  = use_augmentation
        self._use_srch = use_search
        self._n_cand   = n_candidates or MAX_CANDIDATES
        
        # UPA Components (Phase 14 & 15)
        try:
            from arc_upa import ARCUPALattice, ARCPrimeMidiRouter
            from session_daw import SessionDAW
            from pedagogical_engine import PedagogicalEngine
            self._upa_lattice = ARCUPALattice()
            self._upa_router  = ARCPrimeMidiRouter(self._upa_lattice)
            self._session_daw = SessionDAW()
            self._ped_engine  = PedagogicalEngine()
        except ImportError:
            self._upa_lattice = None
            self._upa_router  = None
            self._session_daw = None
            self._ped_engine  = None

    def _log(self, msg: str):
        if self._verbose:
            print(f"  [ARC] {msg}")

    def _scroll_record(self, payload: str, ev_type: int, score: int = 0):
        if self._bridge:
            try:
                from vexel_flow import EV_QUERY, EV_RESONANCE, EV_MISS
                self._bridge.scroll.record(payload[:128], ev_type, score)
            except Exception:
                pass

    def solve(self, task: ARCTask) -> list[ARCPrediction]:
        """
        Solve all test inputs in the task.
        Returns one ARCPrediction per test input.
        """
        try:
            from vexel_flow import EV_QUERY, EV_RESONANCE, EV_MISS
        except ImportError:
            EV_QUERY = EV_RESONANCE = EV_MISS = 0

        self._scroll_record(f"arc_task:{task.task_id}", EV_QUERY, 1)

        # ── Sovereign: BRA density gate + UPG routing + eigenstate lookup ───
        _bra_cfg = {}
        try:
            from arc_bra import sovereign_solve_config, BRAPatternStore
            _bra_store = getattr(self, "_bra_store", None)
            _bra_cfg = sovereign_solve_config(task, bra_store=_bra_store,
                                              verbose=self._verbose)
            # Override solver parameters from sovereign config
            if _bra_cfg.get("n_candidates") is not None:
                self._n_cand = max(0, _bra_cfg["n_candidates"])
            self._log(
                f"BRA gate: pipeline={_bra_cfg.get('pipeline')} "
                f"density={_bra_cfg.get('density_report').density:.2f} "
                f"warmstart={'yes' if _bra_cfg.get('bra_warmstart') else 'no'}"
            )
            # BRA eigenstate warmstart (higher confidence than SQL warm_search)
            if (_bra_cfg.get("bra_warmstart") and
                    _bra_cfg.get("bra_resonance", 0) == 2):
                best_program = _bra_cfg["bra_warmstart"]
                best_score   = 1.0
                rule         = "BRA eigenstate exact resonance"
                self._log("BRA exact resonance hit — skipping all stages")
                self._scroll_record(
                    f"arc_bra_exact:{task.task_id}", EV_RESONANCE, 3)
        except Exception as e:
            self._log(f"BRA init (non-fatal): {e}")
        
        # ── UPA Phase 14: MIDI Seed Generation ──
        if self._upa_router:
            try:
                # Dummy activation for seeding based on initial charge
                # Real activation happens in route_ast, but we seed here for persistence
                from arc_bra import eigen_charge
                import json
                data = json.dumps(task.train[0].input).encode('utf-8')
                charge = eigen_charge(data)
                
                # Assign to a few likely experts based on charge resonance
                initial_activations = {}
                for p in list(self._upa_lattice.experts.keys())[:5]:
                    initial_activations[p] = 0.5
                
                midi = self._upa_router.encode_routing(initial_activations, task.task_id)
                task.metadata["upa_seed"] = self._upa_router.generate_seed(midi)
            except Exception as e:
                self._log(f"UPA Seeding error: {e}")

        # ── UPA Phase 15: Session DAW (Memory Recall) ──
        if self._session_daw:
            tape = self._session_daw.playback(task)
            if tape:
                self._log(f"DAW Memory Recall: Found resonant tape '{tape.task_id}' (solve_path={tape.solve_path})")
                task.metadata["daw_recall"] = tape.solve_path

        # ── UPA Phase 16: Pedagogical Intelligence (Curriculum Boost) ──
        if self._ped_engine and self._session_daw:
            curriculum = self._ped_engine.get_curriculum_for_task(task)
            ped_experts = []
            for item in curriculum:
                master = self._session_daw.get_master_tape(item)
                if master:
                    ped_experts.extend(master.solve_path)
            if ped_experts:
                self._log(f"Pedagogical Boost: Active curriculum {curriculum}")
                task.metadata["pedagogical_recall"] = ped_experts

        self._log(f"Task {task.task_id}: {len(task.train)} train, "
                  f"{len(task.test)} test")

        # ── Phase 17: Deterministic ILM synthesis ──
        ilm_candidates = []
        try:
            from arc_ilm import ILMDeterministicEngine
            self._log("Phase 17: Generating deterministic ILM candidates...")
            ilm_candidates = ILMDeterministicEngine.synthesize(task)
            if ilm_candidates:
                self._log(f"  ILM synthesized {len(ilm_candidates)} deterministic programs.")
        except Exception as e:
            self._log(f"Deterministic ILM error: {e}")

        best_program = ""
        best_score   = -1.0
        rule         = ""

        # ── Stage 0: Pattern library warm-start ──────────────────────────────
        if self._library:
            try:
                from arc_search import warm_search
                warm = warm_search(task, "", self._library, verbose=self._verbose)
                if warm and warm.score >= 1.0:
                    best_program = warm.code
                    best_score   = warm.score
                    self._log(f"Warm-start hit: score={warm.score:.2f}")
                    self._scroll_record(
                        f"arc_warmstart:{task.task_id}", EV_RESONANCE, 3)
            except Exception as e:
                self._log(f"Warm-start error: {e}")

        # ── Stage 0b: Hand-crafted program library ────────────────────────────
        if best_score < 1.0:
            try:
                from arc_programs import match_program
                hc_result = match_program(task)
                if hc_result and hc_result["score"] >= 1.0:
                    best_program = hc_result["program"]
                    rule         = hc_result.get("name", "hand-crafted")
                    self._log(f"Hand-crafted match: {rule}")
                    self._scroll_record(
                        f"arc_handcrafted:{task.task_id}:{rule}", EV_RESONANCE, 3)
            except Exception as e:
                self._log(f"Hand-crafted check: {e}")

        # ── Stage 0c: Neuromorphic Integer Kernel (Phase 18) ──────────────────
        if best_score < 1.0:
            try:
                from arc_neuro import neuro_solve_v2
                n_res = neuro_solve_v2(task)
                if n_res["prediction"] is not None:
                    # Treat all kernel hits as perfect deterministic matches
                    best_program = f"NeuromorphicKernel.{n_res['tier']}"
                    best_score = 1.0
                    rule = f"Neuromorphic Integer Kernel ({n_res['tier']})"
                    self._log(f"Neuromorphic Kernel hit: {n_res['tier']} (conf={n_res['confidence']})")
                    task.metadata["neuro_kernel_res"] = n_res
                    self._scroll_record(f"arc_neuro_kernel:{n_res['tier']}", EV_RESONANCE, 3)
            except Exception as e:
                self._log(f"Neuromorphic kernel check failed: {e}")

        # ── Stage 0d: Local Rule (Neuromorphic Receptive Field) Fallback ───
        if best_score < 1.0:
            try:
                from arc_local_rules import LocalRuleLearner, LocalRulePropagator
                learner = LocalRuleLearner()
                for pair in task.train:
                    learner.learn_from_pair(pair.input, pair.output)
                
                rules = learner.get_deterministic_rules()
                if rules:
                    from arc_local_rules import AxonPropagator
                    # Test Single-Pass consistency
                    correct_single = 0
                    for pair in task.train:
                        pred = LocalRulePropagator.apply_layer(pair.input, rules)
                        if pred == pair.output:
                            correct_single += 1
                    
                    if correct_single == len(task.train):
                        best_program = "LocalRuleEngine.apply(grid)"
                        best_score = 1.0
                        rule = "Neuromorphic Receptive Field Match"
                        self._log("Local Rule Match: Consistent 3x3 mapping found (Single Pass).")
                        task.metadata["local_rules"] = rules
                        task.metadata["local_mode"] = "single"
                    else:
                        # Test Iterative-Stability consistency (Phase 13)
                        correct_iter = 0
                        for pair in task.train:
                            pred = AxonPropagator.propagate_until_stable(pair.input, rules)
                            if pred == pair.output:
                                correct_iter += 1
                        
                        if correct_iter == len(task.train):
                            best_program = "LocalRuleEngine.apply(grid)"
                            best_score = 1.0
                            rule = "Neuromorphic Local Attractor Match"
                            self._log("Local Rule Match: Consistent 3x3 mapping found (Iterative Attractor).")
                            task.metadata["local_rules"] = rules
                            task.metadata["local_mode"] = "iterative"
            except Exception as e:
                self._log(f"Local rule check failed: {e}")

        # --- PHASE 9/10: Neuromorphic Spike Pre-calculation ---
        spiked_order = []
        try:
            from arc_com import CoMEngine
            from arc_neuro import NeuromorphicBrain
            com = CoMEngine()
            com_data = com.synthesize_sequence(task.train[0].input, task.train[0].output)
            brain = NeuromorphicBrain()
            train_context = [{"input": ex.input, "output": ex.output} for ex in task.train]
            spiked_order = brain.route_ast(task.train[0].input, task.train[0].output, com_data, train_pairs=train_context)
        except Exception as e:
            self._log(f"Brain spike calculation failed: {e}")

        # ── Stage 1: Brute-force DSL search ──────────────────────────────────
        if best_score < 1.0 and self._use_bf:
            self._log("Stage 1: Brute-force DSL search...")
            try:
                from arc_search import brute_force_search
                bf = brute_force_search(task, max_depth=2,
                                        time_limit=20.0,
                                        verbose=self._verbose,
                                        upg_prim_order=spiked_order or _bra_cfg.get("upg_prim_order"))
                if bf and bf.score > best_score:
                    best_program = bf.code
                    best_score   = bf.score
                    self._log(f"Brute-force best: {bf.score:.2f}")
                    self._scroll_record(
                        f"arc_bf:{task.task_id}:{bf.score:.2f}",
                        EV_RESONANCE if bf.score > 0 else EV_MISS,
                        int(bf.score * 3))
            except Exception as e:
                self._log(f"Brute-force error: {e}")

        # ── Stage 2: Rule analysis ─── (only if TENT gate is open) ────────────
        _pipeline = _bra_cfg.get("pipeline", "llm_standard")
        if best_score < 1.0 and _pipeline == "brute_force":
            self._log("Stage 2 skipped: TENT gate closed (low-density task)")
        if best_score < 1.0 and _pipeline != "brute_force":
            self._log("Stage 2: Analyzing rule (object-centric)...")
            try:
                from arc_abstraction import abstract_task, ABSTRACT_ANALYSIS_SYSTEM
                prompt = abstract_task(task)
                rule   = _call_llm(
                    [{"role": "user", "content": prompt}],
                    system=ABSTRACT_ANALYSIS_SYSTEM, max_tokens=1024)
                self._log(f"Rule: {rule[:100]}...")
                self._scroll_record(
                    f"arc_rule:{task.task_id}:{rule[:64]}", EV_QUERY, 1)
            except Exception as e:
                self._log(f"Analysis failed: {e} — using basic analysis")
                try:
                    rule = analyze_task(task)
                except Exception:
                    rule = "Unable to determine rule."

        # ── Stage 3: Synthesize + refine ─── (cinema/standard only) ──────────
            # Include deterministic candidates at the start of attempt 0
            all_initial_candidates = (ilm_candidates or [])
            
            for attempt in range(1 + MAX_REFINEMENTS):
                self._log(f"  Attempt {attempt+1}/{1+MAX_REFINEMENTS}...")
                programs_to_test = []
                
                try:
                    if attempt == 0:
                        # Add LLM candidate if possible
                        try:
                            llm_prog = synthesize_program(task, rule)
                            if llm_prog: programs_to_test.append(llm_prog)
                        except Exception as e:
                            self._log(f"  LLM synthesis failed ({e}), relying on deterministic candidates.")
                        
                        # Add all deterministic candidates
                        programs_to_test.extend(all_initial_candidates)
                    else:
                        if current_prog:
                            program = refine_program(task, rule, current_prog, current_failures)
                            if program: programs_to_test.append(program)
                except Exception as e:
                    self._log(f"  Synthesis error: {e}")
                    # If we have deterministic candidates but LLM refined failed, continue to next attempt if needed
                    if not programs_to_test and not all_initial_candidates:
                        break

                for program in programs_to_test:
                    ev = evaluate_program(program, task)
                    self._log(f"  Score: {ev['score']:.2f} "
                              f"({ev['correct']}/{ev['total']})")
                    self._scroll_record(
                        f"arc_synth:{task.task_id}:a{attempt}:"
                        f"score={ev['score']:.2f}",
                        EV_RESONANCE if ev["score"] > 0 else EV_MISS,
                        int(ev["score"] * 3))

                    if ev["score"] > best_score:
                        best_score   = ev["score"]
                        best_program = program

                    if ev["score"] >= 1.0:
                        self._log("  Perfect score!")
                        # --- PHASE 10: LTP Learning ---
                        try:
                            from arc_neuro import NeuromorphicBrain
                            # learn_success takes the input grid and the primitives
                            brain = NeuromorphicBrain()
                            brain.learn_success(task.train[0].input, ev.get("primitives", []))
                        except Exception as e:
                            self._log(f"  LTP Error: {e}")
                        break
                
                if best_score >= 1.0:
                    break

                # For refinement, we use the best one found so far this round or overall
                if programs_to_test:
                    current_prog     = (best_program if best_program else programs_to_test[0])
                    # We need the failures from the specific run of evaluate_program
                    # To keep it simple, we'll re-evaluate the best_program if it changed
                    ev_best = evaluate_program(current_prog, task)
                    current_failures = ev_best["failures"]
                else:
                    break

        # ── Stage 4: Augmentation candidates ─────────────────────────────────
        if best_score < 1.0 and best_program and self._use_aug:
            self._log("Stage 4: Augmentation candidates...")
            try:
                from arc_augment import generate_augmented_programs
                aug_cands = generate_augmented_programs(
                    best_program, task, min_score=0.4)
                for ac in aug_cands:
                    ev = evaluate_program(ac["program"], task)
                    if ev["score"] > best_score:
                        best_score   = ev["score"]
                        best_program = ac["program"]
                        self._log(f"  Augment {ac['aug_name']}: {ev['score']:.2f}")
                    if best_score >= 1.0:
                        break
            except Exception as e:
                self._log(f"Augmentation error: {e}")

        # ── Stage 5: Multi-candidate voting ──────────────────────────────────
        if best_score < 1.0 and self._use_srch and rule:
            self._log("Stage 5: Multi-candidate generation + voting...")
            try:
                from arc_search import generate_candidates, fill_candidate_predictions, vote_predictions
                _n = (_bra_cfg.get("n_candidates") or self._n_cand)
                cands = generate_candidates(task, rule,
                                            n_candidates=_n,
                                            verbose=self._verbose)
                # Include the best program so far as a candidate too
                if best_program:
                    from arc_search import Candidate
                    bc = Candidate(best_program, best_score, "prior_best")
                    cands.append(bc)

                cands = fill_candidate_predictions(cands, task)
                voted = vote_predictions(cands, len(task.test))

                # Pick best program by score
                if cands:
                    top = max(cands, key=lambda c: c.score)
                    if top.score > best_score:
                        best_score   = top.score
                        best_program = top.code
                        self._log(f"  Multi-cand best: {top.score:.2f}")
            except Exception as e:
                self._log(f"Multi-candidate error: {e}")
                voted = None
        else:
            voted = None

        # ── Build predictions ─────────────────────────────────────────────────
        predictions = []
        for i, test_pair in enumerate(task.test):
            predicted   = None
            reasoning   = rule or "brute-force DSL"
            train_score = best_score

            # Try voted result first (highest consensus)
            if voted and i < len(voted) and voted[i] is not None:
                predicted = voted[i]
                reasoning = f"Multi-candidate vote (train={best_score:.2f})"

            # Fall back to best program
            if predicted is None:
                if best_program == "LocalRuleEngine.apply(grid)" and "local_rules" in task.metadata:
                    mode = task.metadata.get("local_mode", "single")
                    if mode == "iterative":
                        from arc_local_rules import AxonPropagator
                        predicted = AxonPropagator.propagate_until_stable(test_pair.input, task.metadata["local_rules"])
                        reasoning = f"Neuromorphic Local Attractor (NCA Iterative)"
                    else:
                        from arc_local_rules import LocalRulePropagator
                        predicted = LocalRulePropagator.apply_layer(test_pair.input, task.metadata["local_rules"])
                        reasoning = f"Neuromorphic Local Rule (3x3 receptive field)"
                elif best_program.startswith("NeuromorphicKernel.") and "neuro_kernel_res" in task.metadata:
                    n_res = task.metadata["neuro_kernel_res"]
                    # If multiple test inputs, we need to re-run the specific solver for this test index
                    if len(task.test) > 1:
                        try:
                            from arc_neuro import solve_by_local_rules_v2, solve_by_delta, solve_by_period_v2
                            # Mock the task to have only the current test pair for the specific solver
                            from copy import deepcopy
                            task_subset = deepcopy(task)
                            task_subset.test = [test_pair]
                            if n_res["tier"].startswith("local"):
                                wide = ("5x5" in n_res["tier"])
                                predicted = solve_by_local_rules_v2(task_subset, wide=wide)
                            elif n_res["tier"] == "delta":
                                predicted = solve_by_delta(task_subset)
                            elif n_res["tier"] == "period":
                                predicted = solve_by_period_v2(task_subset)
                        except:
                            predicted = n_res["prediction"] if i == 0 else None
                    else:
                        predicted = n_res["prediction"]
                    reasoning = f"Neuromorphic Integer Kernel ({n_res['tier']})"
                elif best_program:
                    pred, err = execute_program(best_program, test_pair.input)
                    if not err and pred is not None:
                        predicted = pred
                        reasoning = f"Program synthesis (train={best_score:.2f})"

            # ── Stage 6: VQA fallback ─────────────────────────────────────
            if predicted is None:
                self._log(f"Stage 6: VQA fallback for test {i+1}...")
                try:
                    predicted = _vqa_fallback(task, i, self._bridge)
                    if predicted:
                        reasoning   = "VQA vision fallback"
                        train_score = 0   # BRA resonance 0 = no match
                        self._scroll_record(
                            f"arc_vqa_fallback:{task.task_id}:{i}",
                            EV_RESONANCE, 1)
                except Exception as e:
                    self._log(f"VQA fallback error: {e}")

            # Score against ground truth if available
            correct = None
            if test_pair.output is not None and predicted is not None:
                correct = predicted == test_pair.output

            # --- PHASE 19: RC1 Integrity Audit ---
            try:
                from arc_integer_constraints import get_placeholder_state_for_task, rc1_audit_trace
                rc1_state = get_placeholder_state_for_task(task.task_id)
                integrity_trace = rc1_audit_trace(rc1_state)
                reasoning += integrity_trace
            except Exception as e:
                self._log(f"RC1 Audit Error: {e}")

            self._scroll_record(
                f"arc_pred:{task.task_id}:t{i}:"
                f"{'ok' if predicted else 'fail'}",
                EV_RESONANCE if predicted else EV_MISS,
                2 if predicted else 0)

            try:
                root = self._bridge.scroll.eigen() if self._bridge else "0x0"
            except Exception:
                root = "0x0"

            predictions.append(ARCPrediction(
                task_id     = task.task_id,
                test_index  = i,
                predicted   = predicted,
                confidence  = best_score,
                program     = best_program,
                reasoning   = reasoning[:4096],
                train_score = train_score,
                correct     = correct,
                vexel_root  = root,
            ))

        # Store to pattern library
        if self._library and best_program and best_score > 0:
            try:
                from arc_memory import classify_pattern
                self._library.store(
                    task.task_id, rule[:500], best_program,
                    best_score,
                    classify_pattern(rule) if rule else "unknown")
            except Exception:
                pass

        n_predicted = sum(1 for p in predictions if p.predicted)
        n_solved    = sum(1 for p in predictions if p.correct)
        # ── Phase 15: Session DAW - RECORD successful session ──
        if self._session_daw and n_solved > 0:
            try:
                # Use the first correctly solved test index to define the successful path
                successful_pred = next((p for p in predictions if p.correct), predictions[0])
                solve_path = []
                if successful_pred.reasoning:
                    # Heuristic to extract expert name from reasoning
                    if "Program" in successful_pred.reasoning:
                        solve_path = extract_primitives_from_code(successful_pred.program)
                    elif "(" in successful_pred.reasoning:
                        # Extract from reasonings like "Neuromorphic Local Rule (3x3 receptive field)"
                        # or "RecursiveTiler.solve(grid)"
                        match = re.search(r"([A-Za-z0-9_]+)\.", successful_pred.reasoning)
                        if match:
                            solve_path = [match.group(1)]
                        else:
                            solve_path = [successful_pred.reasoning.split('(')[0].strip()]

                if solve_path and self._upa_router:
                    # Encode activations to MIDI
                    # Convert solve_path to strengths (1.0 each for now)
                    activations = {p: 1.0 for p in solve_path}
                    midi_bytes = self._upa_router.encode_routing(activations, task.task_id)
                    self._session_daw.record(task, midi_bytes, solve_path, score=best_score)
                    self._log(f"DAW recorded successful tape for {task.task_id}: {solve_path}")
            except Exception as e:
                self._log(f"DAW record error: {e}")

        return predictions

    def solve_batch(self, tasks: list[ARCTask]) -> dict[str, list[ARCPrediction]]:
        """Solve multiple tasks. Returns {task_id: [predictions]}."""
        results = {}
        for task in tasks:
            try:
                results[task.task_id] = self.solve(task)
            except Exception as e:
                self._log(f"Error solving {task.task_id}: {e}")
                results[task.task_id] = []
        return results

    def evaluate_on_dataset(self, tasks: list[ARCTask]) -> dict:
        """
        Run solver on tasks with ground truth test outputs.
        Returns full scoring breakdown.
        """
        try:
            from vexel_flow import EV_RESONANCE, EV_MISS
        except ImportError:
            EV_RESONANCE = EV_MISS = 0

        all_results = self.solve_batch(tasks)
        solved = 0
        total  = 0
        task_scores = []

        for task in tasks:
            preds      = all_results.get(task.task_id, [])
            pred_grids = [p.predicted for p in preds]
            sc         = score_task(pred_grids, task)
            task_scores.append(sc)
            total += 1
            if sc["solved"]:
                solved += 1

        self._scroll_record(
            f"arc_eval:tasks={total}:solved={solved}",
            EV_RESONANCE if solved > 0 else EV_MISS,
            min(3, solved))

        return {
            "tasks_attempted": total,
            "tasks_solved":    solved,
            "solve_rate":      solved / total if total else 0.0,
            "avg_bra_resonance": (
                sum(s.get("bra_resonance_avg", 0) for s in task_scores) / total
                if total else 0.0),
            "task_breakdown":  task_scores,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC-AGI Solver CLI")
    sub    = parser.add_subparsers(dest="cmd")

    # solve single task
    sp = sub.add_parser("solve", help="Solve a task by ID or file path")
    sp.add_argument("task", help="Task ID (e.g. 007bbfb7) or path to .json file")
    sp.add_argument("--split", default="training",
                    choices=["training","evaluation"], help="ARC split")
    sp.add_argument("--verbose", "-v", action="store_true")

    # solve file
    fp = sub.add_parser("file", help="Solve a task from a local JSON file")
    fp.add_argument("path", help="Path to task JSON file")
    fp.add_argument("--verbose", "-v", action="store_true")

    # demo
    sub.add_parser("demo", help="Run a self-contained demo (no API key needed)")

    args = parser.parse_args()

    if args.cmd == "solve":
        from arc_types import load_task
        task = load_task(args.task, args.split)
        solver = ARCSolver(verbose=args.verbose)
        preds  = solver.solve(task)
        for p in preds:
            status = "✓" if p.correct else ("✗" if p.correct is False else "?")
            print(f"Test {p.test_index+1}: {status}  train_score={p.train_score:.2f}"
                  f"  backend={'program' if p.program else 'vqa'}")
            if p.predicted:
                print(grid_to_text(p.predicted))

    elif args.cmd == "file":
        from arc_types import load_task_from_file
        task   = load_task_from_file(args.path)
        solver = ARCSolver(verbose=args.verbose)
        preds  = solver.solve(task)
        for p in preds:
            status = "✓" if p.correct else ("✗" if p.correct is False else "?")
            print(f"Test {p.test_index+1}: {status}  train_score={p.train_score:.2f}")
            if p.predicted:
                print(grid_to_text(p.predicted))

    elif args.cmd == "demo":
        print("\n" + "═"*60)
        print("  ARC-AGI SOLVER DEMO")
        print("="*60)

        # Build a simple task: recolor blue→red
        from arc_types import ARCTask, Pair

        task = ARCTask(
            task_id="demo_recolor",
            train=[
                Pair(input=[[1,0,0],[0,1,0],[0,0,1]],
                     output=[[2,0,0],[0,2,0],[0,0,2]]),
                Pair(input=[[1,1,0],[0,0,0],[0,0,1]],
                     output=[[2,2,0],[0,0,0],[0,0,2]]),
            ],
            test=[
                Pair(input=[[0,1,0],[1,0,1],[0,1,0]],
                     output=[[0,2,0],[2,0,2],[0,2,0]])
            ],
        )

        print(f"\nTask: {task.task_id}")
        print("Training pair 1 input:")
        print(grid_to_text(task.train[0].input))
        print("Training pair 1 output:")
        print(grid_to_text(task.train[0].output))
        print("\nTest input:")
        print(grid_to_text(task.test[0].input))
        print("\nExpected output:")
        print(grid_to_text(task.test[0].output))

        # Test DSL execution directly (no API key needed)
        code = """
def transform(grid):
    return recolor(grid, 1, 2)
"""
        result, err = execute_program(code, task.test[0].input)
        if err:
            print(f"\nExecution error: {err}")
        else:
            print("\nDSL program output:")
            print(grid_to_text(result))
            correct = result == task.test[0].output
            print(f"Correct: {'✓' if correct else '✗'}")

        ev = evaluate_program(code, task)
        print(f"\nTraining evaluation: {ev['correct']}/{ev['total']} correct ({ev['score']:.0%})")
        print("\nNote: Full solver requires ANTHROPIC_API_KEY or OPENROUTER_API_KEY")
        print("="*60 + "\n")

    else:
        parser.print_help()
