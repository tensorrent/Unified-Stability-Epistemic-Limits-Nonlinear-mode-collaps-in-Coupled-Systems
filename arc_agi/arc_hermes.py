"""
arc_hermes.py — ARC-AGI ↔ Sovereign Stack Integration
=======================================================
Wires the ARC solver into the Hermes + Vexel + VQA stack.

Provides:
  1. Tool interception for hermes_hooks.py (arc_solve, arc_pattern_search,
     arc_pattern_stats, arc_render tools)
  2. ARCHermesAgent — a ready-to-use agent pre-configured with ARC tools
  3. Standalone evaluation runner (no Hermes required)
  4. Stack health check (verifies all dependencies)

ARC tool → scroll event mapping:
  arc_solve(task)              → EV_QUERY (task received)
                                 EV_RESONANCE score=3 (solved)
                                 EV_MISS (failed)
  arc_pattern_search(query)    → EV_QUERY score=1
  arc_pattern_stats()          → EV_QUERY score=1
  arc_render(task, test_index) → EV_QUERY score=1 (image generated)

Usage (programmatic):
    from arc_hermes import ARCHermesAgent
    agent = ARCHermesAgent(bridge=hermes_bridge)
    result = agent.solve_file("path/to/task.json")

Usage (CLI):
    python arc_hermes.py solve 007bbfb7
    python arc_hermes.py file task.json
    python arc_hermes.py eval --split training --limit 10
    python arc_hermes.py health
    python arc_hermes.py demo
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

SOVEREIGN_SDK = os.environ.get("SOVEREIGN_SDK", os.path.dirname(__file__))
if SOVEREIGN_SDK not in sys.path:
    sys.path.insert(0, SOVEREIGN_SDK)

# ── Tool interception (called from hermes_hooks.py) ───────────────────────────

def handle_arc_tool(tool_name: str, tool_input: dict,
                    scroll_bridge=None) -> Optional[dict]:
    """
    Entry point for hermes_hooks interception of ARC tools.
    Returns result dict for the model, or None if tool not recognised.
    """
    try:
        from arc_memory import (
            handle_arc_solve_tool,
            handle_arc_pattern_search_tool,
            handle_arc_pattern_stats_tool,
        )
    except ImportError as e:
        return {"error": f"ARC modules not available: {e}"}

    if tool_name == "arc_solve":
        result = handle_arc_solve_tool(tool_input, scroll_bridge)
        return _wrap_result(result, "arc_solve")

    elif tool_name == "arc_pattern_search":
        result = handle_arc_pattern_search_tool(tool_input, scroll_bridge)
        return _wrap_result(result, "arc_pattern_search")

    elif tool_name == "arc_pattern_stats":
        result = handle_arc_pattern_stats_tool(tool_input, scroll_bridge)
        return _wrap_result(result, "arc_pattern_stats")

    elif tool_name == "arc_render":
        result = _handle_arc_render_tool(tool_input, scroll_bridge)
        return _wrap_result(result, "arc_render")

    return None  # Not an ARC tool


def _wrap_result(result: dict, tool_name: str) -> dict:
    if "error" in result:
        return {"event": "MISS", "tool": tool_name, **result}
    return {"event": "RESONANCE", "tool": tool_name, "ok": True, **result}


def _handle_arc_render_tool(tool_input: dict, scroll_bridge=None) -> dict:
    """Render an ARC task as PNG and return base64 data URI."""
    try:
        from arc_types import ARCTask, load_task, load_task_from_file
        from arc_renderer import render_task, render_pair
        import base64
    except ImportError as e:
        return {"error": f"arc_render deps not available: {e}"}

    task_ref   = tool_input.get("task", "")
    mode       = tool_input.get("mode", "task")   # "task" | "pair"
    pair_index = int(tool_input.get("pair_index", 0))

    if not task_ref:
        return {"error": "arc_render: task is required"}

    try:
        p = Path(task_ref)
        if p.exists():
            task = load_task_from_file(str(p))
        elif task_ref.startswith("{"):
            task = ARCTask.from_dict(json.loads(task_ref), "inline")
        else:
            task = load_task(task_ref)
    except Exception as e:
        return {"error": f"Could not load task: {e}"}

    if mode == "pair" and 0 <= pair_index < len(task.train):
        pair = task.train[pair_index]
        png  = render_pair(pair.input, pair.output or pair.input)
    else:
        png = render_task(task, include_test=True)

    b64 = base64.b64encode(png).decode()

    if scroll_bridge:
        from vexel_flow import EV_QUERY
        scroll_bridge.scroll.record(
            f"arc_render:{task.task_id}:{mode}", EV_QUERY, 1)

    return {
        "task_id":   task.task_id,
        "data_uri":  f"data:image/png;base64,{b64}",
        "png_bytes": len(png),
        "mode":      mode,
    }


# ── ARC Hermes Agent ──────────────────────────────────────────────────────────

class ARCHermesAgent:
    """
    Ready-to-use ARC solver that integrates with the Hermes scroll bridge.
    Wraps ARCSolver + ARCPatternLibrary + VQA in one convenient API.
    """

    def __init__(self, bridge=None, verbose: bool = True,
                 pattern_db: str = None):
        from arc_memory import ARCSession
        self._session = ARCSession(
            bridge=bridge, verbose=verbose, pattern_db=pattern_db)
        self._bridge  = bridge
        self._verbose = verbose

    def solve_task_id(self, task_id: str,
                      split: str = "training") -> dict:
        from arc_types import load_task
        task = load_task(task_id, split)
        preds = self._session.solve_task(task)
        return self._format_results(task, preds)

    def solve_file(self, path: str) -> dict:
        from arc_types import load_task_from_file
        task  = load_task_from_file(path)
        preds = self._session.solve_task(task)
        return self._format_results(task, preds)

    def solve_json(self, task_json: str, task_id: str = "inline") -> dict:
        from arc_types import ARCTask
        task  = ARCTask.from_dict(json.loads(task_json), task_id)
        preds = self._session.solve_task(task)
        return self._format_results(task, preds)

    def run_evaluation(self, task_ids: list[str],
                       split: str = "training") -> dict:
        return self._session.load_and_run(task_ids, split)

    def run_from_files(self, paths: list[str]) -> dict:
        return self._session.run_from_files(paths)

    def pattern_stats(self) -> dict:
        return self._session._library.stats()

    def _format_results(self, task, preds) -> dict:
        from arc_types import grid_to_text, grid_shape
        output = []
        for p in preds:
            output.append({
                "test_index":    p.test_index,
                "train_score":   round(p.train_score, 3),
                "confidence":    round(p.confidence, 3),
                "reasoning":     p.reasoning[:300],
                "program_lines": len(p.program.split("\n")) if p.program else 0,
                "predicted_text": grid_to_text(p.predicted) if p.predicted else None,
                "predicted_shape": grid_shape(p.predicted) if p.predicted else None,
                "correct":       p.correct,
                "vexel_root":    p.vexel_root,
            })
        return {
            "task_id":     task.task_id,
            "predictions": output,
            "solved":      all(o["correct"] for o in output if o["correct"] is not None),
            "pattern_stats": self._session._library.stats(),
        }

    def close(self):
        self._session.close()


# ── Stack health check ────────────────────────────────────────────────────────

def health_check() -> dict:
    """Verify all ARC dependencies are available."""
    checks = {}

    # arc_types
    try:
        from arc_types import ARCTask, grid_to_text, rot90, extract_objects
        checks["arc_types"] = {"ok": True}
    except Exception as e:
        checks["arc_types"] = {"ok": False, "error": str(e)}

    # arc_renderer
    try:
        from arc_renderer import render_grid
        checks["arc_renderer"] = {"ok": True}
    except Exception as e:
        checks["arc_renderer"] = {"ok": False, "error": str(e)}

    # arc_solver
    try:
        from arc_solver import execute_program, evaluate_program
        checks["arc_solver"] = {"ok": True}
    except Exception as e:
        checks["arc_solver"] = {"ok": False, "error": str(e)}

    # arc_memory
    try:
        from arc_memory import ARCPatternLibrary, classify_pattern
        checks["arc_memory"] = {"ok": True}
    except Exception as e:
        checks["arc_memory"] = {"ok": False, "error": str(e)}

    # VQA bridge
    try:
        from hermes_vqa import VQABridge, probe_backends
        infos = probe_backends()
        available = [b.name for b in infos if b.available]
        checks["hermes_vqa"] = {"ok": True, "backends": available}
    except Exception as e:
        checks["hermes_vqa"] = {"ok": False, "error": str(e)}

    # Trinity library
    trinity_lib = os.environ.get("TRINITY_LIB", "/app/libtrinity.so")
    if Path(trinity_lib).exists():
        try:
            import ctypes
            lib = ctypes.CDLL(trinity_lib)
            lib.bra_verify_f369_table.restype = ctypes.c_int32
            ok = bool(lib.bra_verify_f369_table())
            checks["trinity"] = {"ok": ok, "path": trinity_lib}
        except Exception as e:
            checks["trinity"] = {"ok": False, "error": str(e)}
    else:
        checks["trinity"] = {"ok": False, "error": f"Not found: {trinity_lib}"}

    # LLM API
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_or     = bool(os.environ.get("OPENROUTER_API_KEY"))
    checks["llm_api"] = {
        "ok": has_claude or has_or,
        "claude":      has_claude,
        "openrouter":  has_or,
    }

    # ARC dataset cache
    cache = Path.home() / ".arc_cache"
    training_count = len(list((cache/"training").glob("*.json"))) if (cache/"training").exists() else 0
    eval_count     = len(list((cache/"evaluation").glob("*.json"))) if (cache/"evaluation").exists() else 0
    checks["arc_cache"] = {
        "ok":         training_count > 0 or eval_count > 0,
        "training":   training_count,
        "evaluation": eval_count,
        "cache_dir":  str(cache),
    }

    all_ok = all(v["ok"] for v in checks.values())
    return {"healthy": all_ok, "checks": checks}


# ── Hermes hooks patch ─────────────────────────────────────────────────────────
# Patch hermes_hooks.intercept_tool_call to also handle ARC tools.
# Called once at import time if HERMES_ARC_ENABLED=1.

def _patch_hermes_hooks():
    try:
        import hermes_hooks as hk
        original = hk.intercept_tool_call

        def patched(tool_name, tool_input, tool_result=None,
                    session_id=None, success=True):
            # Try ARC tools first
            if tool_name.startswith("arc_"):
                bridge = hk._get_bridge(session_id)
                result = handle_arc_tool(tool_name, tool_input,
                                         scroll_bridge=bridge)
                if result is not None:
                    return result
            # Fall through to original
            return original(tool_name, tool_input, tool_result,
                            session_id, success)

        hk.intercept_tool_call = patched
        return True
    except ImportError:
        return False


# Auto-patch if enabled
if os.environ.get("HERMES_ARC_ENABLED", "1") == "1":
    _patch_hermes_hooks()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ARC-AGI Sovereign Stack CLI")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    # solve task by ID
    sp = sub.add_parser("solve", help="Solve a task by ID")
    sp.add_argument("task_id")
    sp.add_argument("--split", default="training",
                    choices=["training","evaluation"])

    # solve from file
    fp = sub.add_parser("file", help="Solve a task from local JSON file")
    fp.add_argument("path")

    # evaluate on multiple tasks
    ep = sub.add_parser("eval", help="Evaluate on multiple tasks")
    ep.add_argument("--split", default="training",
                    choices=["training","evaluation"])
    ep.add_argument("--limit", type=int, default=5,
                    help="Max tasks to evaluate")
    ep.add_argument("--ids", nargs="*",
                    help="Specific task IDs (if omitted, loads from cache)")

    # health check
    sub.add_parser("health", help="Check all dependencies")

    # demo
    sub.add_parser("demo", help="Full end-to-end demo")

    # pattern stats
    sub.add_parser("stats", help="Pattern library statistics")

    args = parser.parse_args()

    # ── health ──
    if args.cmd == "health":
        result = health_check()
        print(f"\nStack health: {'✓ ALL OK' if result['healthy'] else '✗ ISSUES FOUND'}\n")
        for name, info in result["checks"].items():
            mark = "✓" if info["ok"] else "✗"
            extra = ""
            if name == "hermes_vqa" and "backends" in info:
                extra = f"  backends: {info['backends']}"
            elif name == "arc_cache":
                extra = f"  training={info['training']}, eval={info['evaluation']}"
            elif name == "llm_api":
                keys = []
                if info.get("claude"):     keys.append("claude")
                if info.get("openrouter"): keys.append("openrouter")
                extra = f"  keys: {keys or 'none'}"
            elif not info["ok"] and "error" in info:
                extra = f"  error: {info['error']}"
            print(f"  {mark} {name}{extra}")
        print()

    # ── solve ──
    elif args.cmd == "solve":
        agent  = ARCHermesAgent(verbose=args.verbose)
        result = agent.solve_task_id(args.task_id, args.split)
        print(json.dumps(result, indent=2))
        agent.close()

    # ── file ──
    elif args.cmd == "file":
        agent  = ARCHermesAgent(verbose=args.verbose)
        result = agent.solve_file(args.path)
        print(json.dumps(result, indent=2))
        agent.close()

    # ── eval ──
    elif args.cmd == "eval":
        from arc_types import list_cached_tasks
        ids = args.ids
        if not ids:
            ids = list_cached_tasks(args.split)[:args.limit]
        if not ids:
            print(f"No cached tasks for split '{args.split}'. "
                  f"Run: python arc_hermes.py solve <task_id> first.")
            sys.exit(1)
        agent  = ARCHermesAgent(verbose=args.verbose)
        result = agent.run_evaluation(ids[:args.limit], args.split)
        print(f"\nResults: {result['tasks_solved']}/{result['tasks_total']} solved"
              f" ({result['solve_rate']:.1%})")
        print(json.dumps(result["pattern_stats"], indent=2))
        agent.close()

    # ── stats ──
    elif args.cmd == "stats":
        agent = ARCHermesAgent(verbose=False)
        print(json.dumps(agent.pattern_stats(), indent=2))
        agent.close()

    # ── demo ──
    elif args.cmd == "demo":
        print("\n" + "═"*64)
        print("  ARC-AGI SOVEREIGN STACK — FULL DEMO")
        print("═"*64)

        # 1. Health check
        print("\n[1] Stack health check:")
        h = health_check()
        for name, info in h["checks"].items():
            mark = "✓" if info["ok"] else "✗"
            print(f"    {mark} {name}")

        # 2. Toy task — pure DSL, no API key needed
        print("\n[2] DSL execution (no API key required):")
        from arc_types import ARCTask, Pair, grid_to_text
        from arc_solver import execute_program, evaluate_program

        task = ARCTask(
            task_id="demo_rot90",
            train=[
                Pair(input=[[1,2,0],[0,0,0],[0,0,3]],
                     output=[[0,0,1],[0,0,2],[3,0,0]]),
            ],
            test=[
                Pair(input=[[0,0,4],[0,5,0],[6,0,0]],
                     output=[[6,0,0],[0,5,0],[0,0,4]])
            ],
        )

        code = "def transform(grid): return rot90(grid)"
        ev   = evaluate_program(code, task)
        print(f"    Rule: rotate 90° clockwise")
        print(f"    Program: {code}")
        print(f"    Train score: {ev['correct']}/{ev['total']}")
        result, err = execute_program(code, task.test[0].input)
        correct = result == task.test[0].output if result else False
        print(f"    Test prediction correct: {'✓' if correct else '✗'}")

        # 3. Pattern library demo
        print("\n[3] Pattern library:")
        import tempfile
        tmp_db = Path(tempfile.mkdtemp()) / "arc_demo.db"
        from arc_memory import ARCPatternLibrary
        lib = ARCPatternLibrary(db_path=str(tmp_db))
        lib.store("demo_rot90", "Rotate 90° clockwise", code, 1.0, "spatial")
        lib.store("demo_color", "Recolor blue to red",
                  "def transform(grid): return recolor(grid,1,2)", 1.0, "color")
        st = lib.stats()
        print(f"    Patterns stored: {st['total']}")
        print(f"    By class: {st['by_class']}")
        best = lib.best_program_for("rotate")
        print(f"    best_program_for('rotate'): {best}")
        lib.close()

        # 4. Renderer demo
        print("\n[4] PNG renderer:")
        from arc_renderer import render_task
        png = render_task(task, include_test=True)
        print(f"    Task PNG: {len(png)} bytes (pure Python, no Pillow)")

        # 5. VQA tool schema
        print("\n[5] ARC tools registered:")
        from arc_memory import ARC_TOOL_SCHEMAS
        for schema in ARC_TOOL_SCHEMAS:
            print(f"    • {schema['name']}: {schema['description'][:60]}...")

        arc_render_schema = {
            "name": "arc_render",
            "description": "Render an ARC task as a PNG image for visual inspection."
        }
        print(f"    • {arc_render_schema['name']}: {arc_render_schema['description']}")

        print(f"\n[6] Scroll event mapping:")
        print("    arc_solve(task)       → EV_QUERY + EV_RESONANCE/EV_MISS")
        print("    arc_pattern_search(q) → EV_QUERY")
        print("    arc_pattern_stats()   → EV_QUERY")
        print("    arc_render(task)      → EV_QUERY")

        print("\n" + "─"*64)
        print("  Set ANTHROPIC_API_KEY to enable full LLM synthesis pipeline.")
        print("  Run: python arc_hermes.py solve 007bbfb7")
        print("─"*64 + "\n")

    else:
        parser.print_help()
