"""
arc_eval.py — ARC-AGI Full Evaluation Harness
==============================================
Manages end-to-end evaluation on the public ARC-AGI dataset.

Features:
  - Download and cache the full 400+400 task dataset
  - Configurable evaluation (subset, full training, full evaluation)
  - Parallel evaluation (ThreadPoolExecutor) with timeout per task
  - JSON result file with task-level breakdown
  - Live progress display
  - Vexel scroll integration — every solved task is a scroll event
  - Pattern library integration — learns from every solved task

ARC-AGI scoring:
  A task is solved if and only if ALL test outputs are EXACTLY correct.
  Score = n_solved / n_attempted

Usage:
    # CLI
    python arc_eval.py run --split training --limit 50
    python arc_eval.py run --split evaluation --limit 400
    python arc_eval.py stats results.json
    python arc_eval.py download

    # Programmatic
    evaluator = ARCEvaluator(bridge=hermes_bridge)
    results   = evaluator.run(split="training", limit=20)
    print(results.summary())
"""

from __future__ import annotations
import json
import os
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SOVEREIGN_SDK = os.environ.get("SOVEREIGN_SDK", os.path.dirname(__file__))
if SOVEREIGN_SDK not in sys.path:
    sys.path.insert(0, SOVEREIGN_SDK)

from arc_types import (
    ARCTask, ARCPrediction, Grid,
    grid_to_text, grid_shape, score_task, list_cached_tasks,
    load_task, load_task_from_file,
)

# ── Config ─────────────────────────────────────────────────────────────────────

ARC_CACHE_DIR    = Path(os.environ.get("ARC_CACHE_DIR",
                        Path.home() / ".arc_cache"))
TASK_TIMEOUT_SEC = int(os.environ.get("ARC_TASK_TIMEOUT", "120"))
MAX_WORKERS      = int(os.environ.get("ARC_WORKERS", "4"))

ARC_GITHUB_BASE  = (
    "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
)
ARC_TASK_INDEX   = {
    "training":   f"{ARC_GITHUB_BASE}/training",
    "evaluation": f"{ARC_GITHUB_BASE}/evaluation",
}

# Known 400-task IDs (training set, first 50 shown; full list downloaded)
_KNOWN_TRAINING_IDS_PARTIAL = [
    "007bbfb7","00d62c1b","017c7c7b","025d127b","045e512c",
    "0520fde7","05269061","05f2a901","06df4c85","08ed6ac7",
    "09629fb3","0962bcdd","0a938d79","0b148d64","0ca9ddb6",
    "0d3d703e","0e206a2e","10fcaaa3","11852cab","1190e5a7",
    "137eaa0f","150deff5","178fcbfb","1a07d186","1b2d62fb",
    "1bfc4729","1c786137","1caeab9d","1cf80156","1e0a9b12",
    "1e32b0e9","1f0c79e5","1f642eb9","1f85a75f","1f876c06",
    "1fad071e","2013d3e2","2204b7a8","22168020","2281f1f4",
    "228f6490","23581191","239be575","23b5c85d","253bf280",
    "256b0a75","25d487eb","25d8a9c8","25ff71a9","264363fd",
]


# ── Task result dataclass ──────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id:      str
    split:        str
    solved:       bool
    n_test:       int
    n_correct:    int
    bra_resonance: int  # 0/1/2 — BRA integer resonance (exact/near/none)
    train_score:  float
    program:      str
    reasoning:    str
    elapsed_sec:  float
    error:        str = ""
    predictions:  list = field(default_factory=list)
    vexel_root:   str = "0x0000000000000000"

    @property
    def ok(self) -> bool:
        return not bool(self.error)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("predictions")    # keep output lean
        return d


# ── Evaluation result ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    split:         str
    n_attempted:   int
    n_solved:      int
    n_errors:      int
    solve_rate:    float
    avg_bra_resonance: float  # mean BRA resonance across tasks (0.0–2.0)
    elapsed_total: float
    task_results:  list[TaskResult]
    pattern_stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*56}",
            f"  ARC-AGI Evaluation Results",
            f"{'─'*56}",
            f"  Split:        {self.split}",
            f"  Attempted:    {self.n_attempted}",
            f"  Solved:       {self.n_solved}  ({self.solve_rate:.1%})",
            f"  Errors:       {self.n_errors}",
            f"  Avg BRA res:  {self.avg_bra_resonance:.2f}/2.0",
            f"  Time:         {self.elapsed_total:.1f}s "
            f"  ({self.elapsed_total/max(1,self.n_attempted):.1f}s/task)",
        ]
        if self.pattern_stats:
            lines.append(f"  Patterns:     {self.pattern_stats.get('total',0)} stored")
        lines.append(f"{'═'*56}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "split":          self.split,
            "n_attempted":    self.n_attempted,
            "n_solved":       self.n_solved,
            "n_errors":       self.n_errors,
            "solve_rate":     self.solve_rate,
            "avg_bra_resonance": self.avg_bra_resonance,
            "elapsed_total":  self.elapsed_total,
            "pattern_stats":  self.pattern_stats,
            "task_results":   [t.to_dict() for t in self.task_results],
        }

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "EvalResult":
        d = json.loads(Path(path).read_text())
        trs = [TaskResult(**t) for t in d.pop("task_results", [])]
        return cls(**d, task_results=trs)


# ── Dataset downloader ────────────────────────────────────────────────────────

def download_task_list(split: str = "training") -> list[str]:
    """
    Fetch the list of all task IDs for a split from GitHub.
    Falls back to partial known list if network unavailable.
    """
    # GitHub directory listing isn't a simple JSON — try fetching the
    # directory listing page and extract .json filenames
    api_url = (
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}"
    )
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "sovereign-arc/1.0",
                     "Accept": "application/vnd.github.v3+json"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            items = json.loads(r.read())
        return [item["name"].replace(".json", "")
                for item in items if item["name"].endswith(".json")]
    except Exception:
        # Fall back to cached local list
        cached = list_cached_tasks(split, ARC_CACHE_DIR)
        if cached:
            return cached
        if split == "training":
            return _KNOWN_TRAINING_IDS_PARTIAL
        return []


def download_dataset(split: str = "training",
                     verbose: bool = True) -> int:
    """
    Download all tasks for a split to the local cache.
    Returns number of tasks successfully downloaded.
    """
    ids   = download_task_list(split)
    total = len(ids)
    ok    = 0

    if verbose:
        print(f"Downloading {total} {split} tasks...")

    for i, tid in enumerate(ids):
        try:
            load_task(tid, split, ARC_CACHE_DIR)
            ok += 1
            if verbose and (i+1) % 50 == 0:
                print(f"  {i+1}/{total}...")
        except Exception as e:
            if verbose:
                print(f"  FAIL {tid}: {e}")

    if verbose:
        print(f"Downloaded {ok}/{total} tasks to {ARC_CACHE_DIR}/{split}/")
    return ok


# ── Core evaluator ─────────────────────────────────────────────────────────────

class ARCEvaluator:
    """
    Full evaluation harness.
    Ties together: solver, augmentation, search, memory, scroll.
    """

    def __init__(self,
                 bridge=None,
                 verbose: bool = True,
                 workers: int = None,
                 task_timeout: int = None,
                 use_augmentation: bool = True,
                 use_brute_force: bool = True,
                 use_search: bool = True,
                 n_candidates: int = 3):
        self._bridge     = bridge
        self._verbose    = verbose
        self._workers    = workers or MAX_WORKERS
        self._timeout    = task_timeout or TASK_TIMEOUT_SEC
        self._use_aug    = use_augmentation
        self._use_bf     = use_brute_force
        self._use_search = use_search
        self._n_cand     = n_candidates
        self._lock       = threading.Lock()
        self._progress   = {"done": 0, "solved": 0, "total": 0}

    def _log(self, msg: str):
        if self._verbose:
            print(f"  [eval] {msg}")

    def _solve_one(self, task: ARCTask,
                   library=None) -> TaskResult:
        """Solve a single task and return TaskResult."""
        t0 = time.time()

        try:
            # Import solver pipeline
            from arc_solver import ARCSolver, analyze_task
            from arc_search import search_and_vote, warm_search
            from arc_augment import solve_with_augmentation

            # Warm-start from library
            warm_cand = None
            if library:
                from arc_search import warm_search as ws
                warm_cand = ws(task, "", library, verbose=self._verbose)

            # Analyze rule
            try:
                rule = analyze_task(task)
            except Exception:
                rule = ""

            # Warm start: reuse library program if score=1.0
            if warm_cand and warm_cand.score >= 1.0:
                program    = warm_cand.code
                train_sc   = warm_cand.score
                predictions = []
                from arc_solver import execute_program
                for pair in task.test:
                    pred, err = execute_program(program, pair.input)
                    predictions.append(pred if not err else None)
            else:
                # Full search + vote
                if self._use_search:
                    result = search_and_vote(
                        task, rule,
                        n_llm_candidates = self._n_cand,
                        brute_force_first = self._use_bf,
                        brute_time_limit  = 15.0,
                        verbose           = self._verbose,
                    )
                    program     = result["best_program"]
                    train_sc    = result["best_score"]
                    predictions = result["voted_predictions"]
                else:
                    solver = ARCSolver(
                        scroll_bridge=self._bridge,
                        verbose=self._verbose
                    )
                    preds       = solver.solve(task)
                    program     = preds[0].program if preds else ""
                    train_sc    = preds[0].train_score if preds else 0.0
                    predictions = [p.predicted for p in preds]

                # Augmentation consistency check
                if self._use_aug and train_sc >= 0.5 and predictions:
                    from arc_solver import execute_program
                    from arc_augment import solve_with_augmentation

                    def prog_solver(aug_task: ARCTask):
                        results = []
                        for pair in aug_task.test:
                            pred, err = execute_program(program, pair.input)
                            results.append(pred if not err else None)
                        return results

                    aug_preds = solve_with_augmentation(
                        task, prog_solver,
                        d4_subset=["identity","rot90","rot180","rot270"],
                        max_views=4,
                        verbose=self._verbose,
                    )
                    # If augmented voted predictions agree with original
                    # use the augmented result (higher confidence)
                    if aug_preds and any(p is not None for p in aug_preds):
                        predictions = aug_preds

            # Score against ground truth
            sc = score_task(predictions, task)
            sim = sc.get("bra_resonance_avg", 0)
            n_correct = sum(1 for r in sc["results"] if r.get("correct"))

            # Store in library
            if library and program and train_sc > 0:
                from arc_memory import ARCPatternLibrary, classify_pattern
                library.store(
                    task_id     = task_id if 'task_id' in locals() else task.task_id,
                    rule        = rule[:500],
                    program     = program,
                    train_score = train_sc,
                    pattern_class = classify_pattern(rule),
                )
                
            # --- PHASE 10: LTP Learning ---
            if sc["solved"] and program:
                try:
                    from arc_neuro import NeuromorphicBrain
                    from arc_solver import extract_primitives_from_code
                    brain = NeuromorphicBrain()
                    prims = extract_primitives_from_code(program)
                    brain.learn_success(task.train[0].input, prims)
                except Exception as e:
                    self._log(f"LTP Learning failed: {e}")

            # Scroll
            if self._bridge:
                from vexel_flow import EV_RESONANCE, EV_MISS
                ev = EV_RESONANCE if sc["solved"] else EV_MISS
                self._bridge.scroll.record(
                    f"arc_eval:{task.task_id}:{'solved' if sc['solved'] else 'fail'}",
                    ev, 3 if sc["solved"] else 1
                )

            root = (self._bridge.scroll.eigen()
                    if self._bridge else "0x0000000000000000")

            return TaskResult(
                task_id     = task.task_id,
                split       = task.source,
                solved      = sc["solved"],
                n_test      = len(task.test),
                n_correct   = n_correct,
                bra_resonance = int(round(float(sim) * 2)),  # normalise to 0/1/2
                train_score = train_sc,
                program     = program[:2000],  # cap for storage
                reasoning   = rule[:4096],
                elapsed_sec = time.time() - t0,
                predictions = [grid_to_text(p) if p else ""
                               for p in predictions],
                vexel_root  = root,
            )

        except Exception as e:
            import traceback
            return TaskResult(
                task_id     = task.task_id,
                split       = "",
                solved      = False,
                n_test      = len(task.test),
                n_correct   = 0,
                bra_resonance = 0,
                train_score = 0,
                program     = "",
                reasoning   = "",
                elapsed_sec = time.time() - t0,
                error       = f"{type(e).__name__}: {e}",
            )

    def _update_progress(self, result: TaskResult):
        with self._lock:
            self._progress["done"]   += 1
            self._progress["solved"] += int(result.solved)
            done   = self._progress["done"]
            solved = self._progress["solved"]
            total  = self._progress["total"]
            rate   = solved / done if done else 0.0
            status = "✓" if result.solved else ("✗" if result.ok else "E")
            if self._verbose:
                print(
                    f"  [{done:3d}/{total}] {status} {result.task_id:<12} "
                    f"train={result.train_score:.2f}  "
                    f"bra={result.bra_resonance}/2  "
                    f"rate={rate:.1%}  "
                    f"{result.elapsed_sec:.1f}s"
                    + (f"  ERR: {result.error[:40]}" if result.error else "")
                )

    def run(self,
            split: str = "training",
            limit: int = None,
            task_ids: list[str] = None,
            output_file: str = None,
            library=None) -> EvalResult:
        """
        Run evaluation on a split.

        split: "training" | "evaluation"
        limit: max tasks to evaluate
        task_ids: specific IDs (overrides split+limit)
        output_file: save JSON results here
        library: ARCPatternLibrary instance (optional)
        """
        t0 = time.time()

        # Build task list
        if task_ids:
            ids = task_ids
        else:
            ids = list_cached_tasks(split, ARC_CACHE_DIR)
            if not ids:
                self._log(f"No cached {split} tasks — downloading...")
                download_dataset(split, verbose=self._verbose)
                ids = list_cached_tasks(split, ARC_CACHE_DIR)
            if limit:
                ids = ids[:limit]

        self._progress = {"done": 0, "solved": 0, "total": len(ids)}
        self._log(f"Evaluating {len(ids)} tasks from split={split!r}")

        # Load tasks
        tasks = []
        for tid in ids:
            try:
                t = load_task(tid, split, ARC_CACHE_DIR)
                tasks.append(t)
            except Exception as e:
                self._log(f"Could not load {tid}: {e}")

        # Run with thread pool
        task_results = []
        if self._workers > 1:
            with ThreadPoolExecutor(max_workers=self._workers) as ex:
                futures = {
                    ex.submit(self._solve_one, t, library): t
                    for t in tasks
                }
                for fut in as_completed(futures):
                    try:
                        result = fut.result(timeout=self._timeout)
                    except TimeoutError:
                        t = futures[fut]
                        result = TaskResult(
                            task_id=t.task_id, split=split,
                            solved=False, n_test=len(t.test),
                            n_correct=0, bra_resonance=0, train_score=0,
                            program="", reasoning="",
                            elapsed_sec=self._timeout,
                            error="timeout",
                        )
                    except Exception as e:
                        t = futures[fut]
                        result = TaskResult(
                            task_id=t.task_id, split=split,
                            solved=False, n_test=len(t.test),
                            n_correct=0, bra_resonance=0, train_score=0,
                            program="", reasoning="",
                            elapsed_sec=0.0,
                            error=str(e),
                        )
                    task_results.append(result)
                    self._update_progress(result)
        else:
            # Sequential (easier for debugging)
            for t in tasks:
                result = self._solve_one(t, library)
                task_results.append(result)
                self._update_progress(result)

        # Aggregate
        n_solved  = sum(1 for r in task_results if r.solved)
        n_errors  = sum(1 for r in task_results if r.error)
        n_att     = len(task_results)
        avg_bra_r = (sum(r.bra_resonance for r in task_results) / n_att
                     if n_att else 0.0)
        elapsed   = time.time() - t0

        pat_stats = library.stats() if library else {}

        result = EvalResult(
            split          = split,
            n_attempted    = n_att,
            n_solved       = n_solved,
            n_errors       = n_errors,
            solve_rate     = n_solved / n_att if n_att else 0.0,
            avg_bra_resonance = avg_bra_r,
            elapsed_total  = elapsed,
            task_results   = task_results,
            pattern_stats  = pat_stats,
        )

        if output_file:
            result.save(output_file)
            self._log(f"Results saved to {output_file}")

        if self._verbose:
            print(result.summary())

        return result

    def run_from_files(self, paths: list[str],
                       output_file: str = None,
                       library=None) -> EvalResult:
        """Run evaluation on local task JSON files."""
        tasks = []
        for p in paths:
            try:
                tasks.append(load_task_from_file(p))
            except Exception as e:
                self._log(f"Could not load {p}: {e}")

        t0           = time.time()
        task_results = []
        self._progress = {"done": 0, "solved": 0, "total": len(tasks)}

        for t in tasks:
            result = self._solve_one(t, library)
            task_results.append(result)
            self._update_progress(result)

        n_solved = sum(1 for r in task_results if r.solved)
        n_errors = sum(1 for r in task_results if r.error)
        n_att    = len(task_results)
        avg_bra_r = sum(r.bra_resonance for r in task_results) / max(1, n_att)
        elapsed  = time.time() - t0
        pat_stats = library.stats() if library else {}

        result = EvalResult(
            split="files", n_attempted=n_att,
            n_solved=n_solved, n_errors=n_errors,
            solve_rate=n_solved/max(1,n_att),
            avg_bra_resonance=avg_bra_r,
            elapsed_total=elapsed,
            task_results=task_results,
            pattern_stats=pat_stats,
        )
        if output_file:
            result.save(output_file)
        if self._verbose:
            print(result.summary())
        return result


# ── Result analysis ────────────────────────────────────────────────────────────

def analyze_results(result: EvalResult) -> dict:
    """Deep analysis of evaluation results."""
    solved   = [r for r in result.task_results if r.solved]
    failed   = [r for r in result.task_results if not r.solved and not r.error]
    errored  = [r for r in result.task_results if r.error]

    # Timing
    times = [r.elapsed_sec for r in result.task_results if r.ok]
    avg_t = sum(times) / len(times) if times else 0.0

    # Train score distribution for unsolved
    fail_train = [r.train_score for r in failed]
    near_solved = [r for r in failed if r.train_score >= 0.5]

    return {
        "summary": {
            "solved":    len(solved),
            "failed":    len(failed),
            "errored":   len(errored),
            "solve_rate": result.solve_rate,
        },
        "timing": {
            "avg_sec":   round(avg_t, 1),
            "total_sec": round(result.elapsed_total, 1),
        },
        "failure_analysis": {
            "near_solved_count": len(near_solved),
            "avg_train_score_on_fails": (
                round(sum(fail_train)/len(fail_train), 3)
                if fail_train else 0.0),
            "near_solved_tasks": [r.task_id for r in near_solved[:10]],
        },
        "error_types": _count_errors(errored),
    }


def _count_errors(errored: list) -> dict:
    types: dict[str, int] = {}
    for r in errored:
        err_type = r.error.split(":")[0] if ":" in r.error else r.error[:20]
        types[err_type] = types.get(err_type, 0) + 1
    return dict(sorted(types.items(), key=lambda x: -x[1]))


def print_failure_report(result: EvalResult, top_n: int = 10):
    """Print the top failing tasks with debug info."""
    failed = [r for r in result.task_results
              if not r.solved and not r.error]
    failed_sorted = sorted(failed, key=lambda r: -r.train_score)

    print(f"\nTop {top_n} near-solved tasks:")
    for r in failed_sorted[:top_n]:
        print(f"  {r.task_id}  train_score={r.train_score:.2f}  "
              f"bra={r.bra_resonance}/2  {r.elapsed_sec:.1f}s")
        if r.reasoning:
            print(f"    Rule: {r.reasoning[:80]}...")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC-AGI Evaluation Harness")
    sub    = parser.add_subparsers(dest="cmd")

    # download
    dp = sub.add_parser("download", help="Download ARC dataset to local cache")
    dp.add_argument("--split", default="training",
                    choices=["training","evaluation","both"])

    # run
    rp = sub.add_parser("run", help="Run evaluation")
    rp.add_argument("--split", default="training",
                    choices=["training","evaluation"])
    rp.add_argument("--limit", type=int, default=None,
                    help="Max tasks to evaluate")
    rp.add_argument("--ids", nargs="*", help="Specific task IDs")
    rp.add_argument("--output", "-o", default=None,
                    help="Save results to JSON file")
    rp.add_argument("--workers", type=int, default=1,
                    help="Parallel workers")
    rp.add_argument("--no-augment", action="store_true")
    rp.add_argument("--no-bf", action="store_true",
                    help="Skip brute-force search")
    rp.add_argument("--candidates", type=int, default=3,
                    help="LLM candidates per task")
    rp.add_argument("--verbose", "-v", action="store_true")

    # stats
    sp = sub.add_parser("stats", help="Analyze a results JSON file")
    sp.add_argument("file", help="Path to results JSON")
    sp.add_argument("--failures", action="store_true",
                    help="Print failure report")

    # demo
    sub.add_parser("demo", help="Run a quick demo (no API key needed)")

    args = parser.parse_args()

    if args.cmd == "download":
        splits = (["training","evaluation"] if args.split == "both"
                  else [args.split])
        for s in splits:
            download_dataset(s, verbose=True)

    elif args.cmd == "run":
        from arc_memory import ARCPatternLibrary
        lib = ARCPatternLibrary()
        ev  = ARCEvaluator(
            verbose           = args.verbose or True,
            workers           = args.workers,
            use_augmentation  = not args.no_augment,
            use_brute_force   = not args.no_bf,
            n_candidates      = args.candidates,
        )
        result = ev.run(
            split       = args.split,
            limit       = args.limit,
            task_ids    = args.ids,
            output_file = args.output,
            library     = lib,
        )
        lib.close()

    elif args.cmd == "stats":
        result  = EvalResult.load(args.file)
        print(result.summary())
        analysis = analyze_results(result)
        print(json.dumps(analysis, indent=2))
        if args.failures:
            print_failure_report(result)

    elif args.cmd == "demo":
        print("\n" + "═"*60)
        print("  ARC-AGI EVALUATOR DEMO")
        print("═"*60)

        # Build toy tasks
        from arc_types import ARCTask, Pair
        from arc_solver import execute_program, evaluate_program

        tasks = [
            ARCTask("demo_001",
                train=[Pair([[1,0],[0,1]], [[2,0],[0,2]]),
                       Pair([[1,1,0],[0,0,1]], [[2,2,0],[0,0,2]])],
                test=[Pair([[0,1,0],[1,0,1],[0,1,0]],
                           [[0,2,0],[2,0,2],[0,2,0]])]),
            ARCTask("demo_002",
                train=[Pair([[1,2,3],[0,0,0]], [[3,2,1],[0,0,0]]),
                       Pair([[0,1,0],[2,3,4]], [[0,1,0],[4,3,2]])],
                test=[Pair([[1,0,2],[3,4,0]], [[2,0,1],[0,4,3]])]),
        ]

        programs = {
            "demo_001": "def transform(grid): return recolor(grid, 1, 2)",
            "demo_002": "def transform(grid): return [row[::-1] for row in grid]",
        }

        print("\n  Testing DSL execution on demo tasks:")
        for task in tasks:
            code = programs[task.task_id]
            ev   = evaluate_program(code, task)
            print(f"  {task.task_id}: train_score={ev['correct']}/{ev['total']}")

        print("\n  Brute-force search demo:")
        from arc_search import brute_force_search
        bf = brute_force_search(tasks[0], max_depth=1, time_limit=5.0, verbose=True)
        if bf:
            print(f"  Found: score={bf.score:.2f}")
            print(f"  Code:  {bf.code}")

        print("\n  Augmentation demo:")
        from arc_augment import generate_augmented_views
        views = generate_augmented_views(tasks[0], d4_subset=["identity","rot90","rot180"])
        print(f"  Generated {len(views)} augmented views")
        for v in views:
            print(f"    {v.aug_name}: task_id={v.task.task_id}")

        print("\n  Pattern library demo:")
        import tempfile
        from arc_memory import ARCPatternLibrary
        tmp_db = Path(tempfile.mkdtemp()) / "eval_demo.db"
        lib    = ARCPatternLibrary(db_path=str(tmp_db))
        for task in tasks:
            code = programs[task.task_id]
            lib.store(task.task_id, "color/geometric transform", code, 1.0)
        print(f"  Stored {lib.stats()['total']} patterns")
        lib.close()

        print(f"\n{'─'*60}")
        print("  Full eval: python arc_eval.py run --split training --limit 20")
        print(f"{'─'*60}\n")

    else:
        parser.print_help()
