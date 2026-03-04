"""
arc_memory.py — ARC Cross-Task Pattern Library
===============================================
Stores learned transformation patterns in Hermes MEMORY.md so the agent
builds up a library of solved concepts across sessions. Every pattern
write is a vexel scroll event — the knowledge graph grows and is
cryptographically anchored.

Memory architecture for ARC:
  MEMORY.md   — solved programs, indexed by pattern class
  SKILL.md    — reusable DSL snippets for common transformations
  vexel_sessions.vexel_root — proof of which session learned which pattern

Pattern classes (Core Knowledge taxonomy):
  spatial    — rotation, reflection, translation, scaling
  pattern    — repetition, tiling, symmetry completion
  object     — counting, filtering, containment, gravity
  color      — recolor, palette swap, conditional coloring
  logic      — XOR, AND/OR between grids, masking
  shape      — outline, fill, resize, morphology
  sequence   — ordering, sorting, ranking objects

Usage:
    lib = ARCPatternLibrary(bridge=hermes_bridge)
    lib.store(task_id, rule, program, score, pattern_class)
    candidates = lib.lookup(task_id, rule_description)
    program = lib.best_program_for(rule_description)
"""

import os
import re
import sys
import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional

SOVEREIGN_SDK = os.environ.get("SOVEREIGN_SDK", os.path.dirname(__file__))
if SOVEREIGN_SDK not in sys.path:
    sys.path.insert(0, SOVEREIGN_SDK)

from arc_types import (
    ARCTask, ARCPrediction, Grid, grid_to_text, grid_shape,
    ARC_COLOR_NAMES, describe_task,
)

# ── Pattern classes ────────────────────────────────────────────────────────────

PATTERN_CLASSES = [
    "spatial", "pattern", "object", "color",
    "logic", "shape", "sequence", "unknown"
]

PATTERN_KEYWORDS = {
    "spatial":  ["rotat", "reflect", "flip", "mirror", "translat", "shift",
                 "scale", "zoom", "resize"],
    "pattern":  ["tile", "repeat", "symmetric", "pattern", "period", "tile"],
    "object":   ["count", "object", "connect", "contain", "largest", "smallest",
                 "gravity", "fall", "sort"],
    "color":    ["recolor", "color", "replace", "map", "palette"],
    "logic":    ["xor", "and", "or", "mask", "overlap", "union", "intersect"],
    "shape":    ["outline", "border", "interior", "hollow", "fill", "expand",
                 "contract", "morph"],
    "sequence": ["order", "rank", "sort", "sequence", "index", "position"],
}

def classify_pattern(rule_text: str) -> str:
    """Heuristically classify a rule into a pattern class."""
    text = rule_text.lower()
    scores = {}
    for cls, keywords in PATTERN_KEYWORDS.items():
        scores[cls] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ── Pattern record ─────────────────────────────────────────────────────────────

class PatternRecord:
    """One stored pattern: task, rule, program, score, class."""

    def __init__(self, task_id: str, rule: str, program: str,
                 train_score: float, pattern_class: str,
                 vexel_root: str = "", ts: float = None,
                 bra_hash: int = 0, bra_trace: int = 0, bra_det: int = 0,
                 str_hash: int = 0, str_trace: int = 0, str_det: int = 0):
        self.task_id       = task_id
        self.rule          = rule
        self.program       = program
        self.train_score   = train_score
        self.pattern_class = pattern_class
        self.vexel_root    = vexel_root
        self.ts            = ts or time.time()
        # Identity charge — task_charge() includes task_id, unique fingerprint
        self.bra_hash  = bra_hash
        self.bra_trace = bra_trace
        self.bra_det   = bra_det
        # Structure charge — task_structure_charge(), excludes task_id
        # Same training examples → same str_hash regardless of task_id
        self.str_hash  = str_hash
        self.str_trace = str_trace
        self.str_det   = str_det

    def to_dict(self) -> dict:
        return {
            "task_id":       self.task_id,
            "rule":          self.rule,
            "program":       self.program,
            "train_score":   self.train_score,
            "pattern_class": self.pattern_class,
            "vexel_root":    self.vexel_root,
            "ts":            self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatternRecord":
        return cls(**d)

    def memory_entry(self) -> str:
        """Format as a MEMORY.md entry (compact)."""
        rule_short = self.rule[:120].replace("\n", " ").strip()
        prog_lines = self.program.strip().split("\n")
        prog_short = prog_lines[0][:80] if prog_lines else ""
        return (
            f"ARC-PATTERN [{self.pattern_class}] task={self.task_id} "
            f"score={self.train_score:.2f}: {rule_short} | "
            f"prog: {prog_short}"
        )

    def skill_content(self) -> str:
        """Format as a SKILL.md for Hermes skill library."""
        return textwrap.dedent(f"""\
            ---
            name: arc_{self.task_id}_{self.pattern_class}
            description: ARC pattern [{self.pattern_class}] from task {self.task_id}
            pattern_class: {self.pattern_class}
            train_score: {self.train_score}
            vexel_root: "{self.vexel_root}"
            ---

            # ARC Pattern: {self.pattern_class}
            **Task:** {self.task_id}
            **Train Score:** {self.train_score:.2f}

            ## Rule
            {self.rule}

            ## Program
            ```python
            {self.program}
            ```
        """)


import textwrap


# ── SQLite pattern store ───────────────────────────────────────────────────────

PATTERN_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS arc_patterns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       TEXT NOT NULL,
    pattern_class TEXT NOT NULL,
    rule          TEXT NOT NULL,
    program       TEXT NOT NULL,
    train_score   REAL NOT NULL,
    vexel_root    TEXT DEFAULT '',
    ts            REAL NOT NULL,
    bra_hash      INTEGER DEFAULT 0,
    bra_trace     INTEGER DEFAULT 0,
    bra_det       INTEGER DEFAULT 0,
    bra_resonance INTEGER DEFAULT 0,
    str_hash      INTEGER DEFAULT 0,
    str_trace     INTEGER DEFAULT 0,
    str_det       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_arc_class  ON arc_patterns(pattern_class);
CREATE INDEX IF NOT EXISTS idx_arc_task   ON arc_patterns(task_id);
CREATE INDEX IF NOT EXISTS idx_arc_score  ON arc_patterns(train_score DESC);
CREATE INDEX IF NOT EXISTS idx_arc_bra_h  ON arc_patterns(bra_hash);
CREATE INDEX IF NOT EXISTS idx_arc_str_h  ON arc_patterns(str_hash);
"""
# NOTE: FTS5 removed. Pattern lookup uses BRA integer charge comparison.
# bra_hash / bra_trace / bra_det: EigenCharge of the stored task.
# Query: fetch all candidates, filter by bra_resonance_score() in Python.
# This is deterministic integer comparison — no text tokenisation.


def _classify_from_keyword(query: str) -> Optional[str]:
    """Map a keyword string to a pattern class without text tokenisation."""
    q = query.lower()
    for cls, words in _CLASS_KEYWORDS.items():
        if any(w in q for w in words):
            return cls
    return None

_CLASS_KEYWORDS = {
    "spatial":    ["rotat", "reflect", "flip", "mirror", "transposi", "scale",
                   "upscale", "downscale", "resize", "crop"],
    "color":      ["recolor", "color", "colour", "palette", "swap", "invert",
                   "hue", "tint"],
    "object":     ["object", "count", "largest", "smallest", "border", "sort",
                   "rank", "interior"],
    "morphology": ["dilat", "erod", "outline", "fill", "hole", "convex",
                   "morph"],
    "gravity":    ["gravity", "fall", "float", "drop", "sink"],
    "logic":      ["xor", "and", "or", "logic", "mask", "overlay"],
    "pattern":    ["tile", "repeat", "period", "mosaic", "pattern"],
    "symmetry":   ["symmetr", "mirror", "complet", "enforc"],
}

class PatternDB:
    """SQLite-backed pattern store with BRA integer charge indexing.
    No FTS5 text search. Lookup = integer EigenCharge resonance comparison."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(PATTERN_DB_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def insert(self, rec: PatternRecord) -> int:
        # Use BRA task charge stored on the record (set by ARCPatternLibrary.store())
        # Convert u64 hash to signed i64 for SQLite INTEGER column
        def _to_i64(n): return n if n < (1<<63) else n - (1<<64)
        bra_hash  = _to_i64(rec.bra_hash)
        bra_trace = rec.bra_trace
        bra_det   = rec.bra_det
        str_hash  = _to_i64(rec.str_hash)
        str_trace = rec.str_trace
        str_det   = rec.str_det
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO arc_patterns
                   (task_id, pattern_class, rule, program, train_score,
                    vexel_root, ts, bra_hash, bra_trace, bra_det,
                    str_hash, str_trace, str_det)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.task_id, rec.pattern_class, rec.rule,
                 rec.program, rec.train_score, rec.vexel_root, rec.ts,
                 bra_hash, bra_trace, bra_det,
                 str_hash, str_trace, str_det)
            )
            self._conn.commit()
            return cur.lastrowid

    def search(self, query: str, limit: int = 5) -> list[PatternRecord]:
        """
        Category-filtered search — deterministic, no text tokenisation.
        query is a category keyword (spatial/color/object/morphology/pattern/logic/gravity).
        For task-level BRA resonance lookup, use search_by_charge().
        """
        # Classify the query keyword into a category
        cls = _classify_from_keyword(query)
        if cls:
            rows = self._conn.execute(
                """SELECT task_id, rule, program, train_score,
                          pattern_class, vexel_root, ts
                   FROM arc_patterns WHERE pattern_class=?
                   ORDER BY train_score DESC LIMIT ?""",
                (cls, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT task_id, rule, program, train_score,
                          pattern_class, vexel_root, ts
                   FROM arc_patterns ORDER BY train_score DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [PatternRecord(*r) for r in rows]

    def search_by_charge(self, bra_hash: int, bra_trace: int, bra_det: int,
                         limit: int = 5, min_resonance: int = 2,
                         use_structure: bool = False) -> list[tuple[int, 'PatternRecord']]:
        """
        BRA integer charge resonance lookup.
        Fetches all rows, filters by bra_resonance_score >= min_resonance.
        Returns list of (resonance_score, PatternRecord) sorted by resonance DESC.
        Deterministic: pure integer comparison, no text processing.
        """
        try:
            from arc_bra import EigenCharge, arc_task_resonance
        except ImportError:
            return []

        # Both query and stored values are i64 from SQLite — convert to u64 for BRA
        def _to_u64(n): return int(n) if int(n) >= 0 else int(n) + (1 << 64)
        query_ec = EigenCharge(hash=_to_u64(bra_hash), trace=bra_trace, det=bra_det)

        if use_structure:
            rows = self._conn.execute(
                """SELECT task_id, rule, program, train_score,
                          pattern_class, vexel_root, ts,
                          str_hash, str_trace, str_det
                   FROM arc_patterns ORDER BY train_score DESC"""
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT task_id, rule, program, train_score,
                          pattern_class, vexel_root, ts,
                          bra_hash, bra_trace, bra_det
                   FROM arc_patterns ORDER BY train_score DESC"""
            ).fetchall()

        results = []
        for row in rows:
            (task_id, rule, program, train_score,
             pattern_class, vexel_root, ts, bh, bt, bd) = row
            stored_ec = EigenCharge(hash=_to_u64(bh), trace=bt, det=bd)
            r = arc_task_resonance(query_ec, stored_ec)
            if r >= min_resonance:
                results.append((r, PatternRecord(task_id, rule, program,
                                                 train_score, pattern_class,
                                                 vexel_root, ts)))
        results.sort(key=lambda x: (-x[0], -x[1].train_score))
        return results[:limit]

    def by_class(self, cls: str, limit: int = 10,
                 min_score: float = 0.5) -> list[PatternRecord]:
        rows = self._conn.execute(
            """SELECT task_id, rule, program,
                      train_score, pattern_class, vexel_root, ts
               FROM arc_patterns
               WHERE pattern_class=? AND train_score>=?
               ORDER BY train_score DESC LIMIT ?""",
            (cls, min_score, limit)
        ).fetchall()
        return [PatternRecord(*r) for r in rows]

    def best_for_task(self, task_id: str) -> Optional[PatternRecord]:
        row = self._conn.execute(
            """SELECT task_id, rule, program,
                      train_score, pattern_class, vexel_root, ts
               FROM arc_patterns WHERE task_id=?
               ORDER BY train_score DESC LIMIT 1""",
            (task_id,)
        ).fetchone()
        return PatternRecord(*row) if row else None

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM arc_patterns").fetchone()[0]
        by_cls = {
            row[0]: row[1]
            for row in self._conn.execute(
                "SELECT pattern_class, COUNT(*) FROM arc_patterns GROUP BY pattern_class"
            ).fetchall()
        }
        avg_score = self._conn.execute(
            "SELECT AVG(train_score) FROM arc_patterns"
        ).fetchone()[0] or 0.0
        perfect = self._conn.execute(
            "SELECT COUNT(*) FROM arc_patterns WHERE train_score=1.0"
        ).fetchone()[0]
        return {
            "total":          total,
            "by_class":       by_cls,
            "avg_train_score": round(avg_score, 3),
            "perfect_score":  perfect,
        }

    def close(self):
        self._conn.close()


# ── ARCPatternLibrary ──────────────────────────────────────────────────────────

ARC_PATTERN_DB_PATH = os.environ.get(
    "ARC_PATTERN_DB",
    os.path.join(os.environ.get("HERMES_DIR", str(Path.home()/".hermes")),
                 "arc_patterns.db")
)

class ARCPatternLibrary:
    """
    Cross-session pattern store backed by SQLite + Hermes MEMORY.md.

    Every stored pattern:
    - Is inserted into arc_patterns.db (searchable by FTS5)
    - Gets a MEMORY.md entry (visible to Hermes agent in future sessions)
    - If score=1.0, optionally becomes a SKILL.md (reusable procedure)
    - Every write is recorded as a vexel EV_RESONANCE event
    """

    def __init__(self, bridge=None,
                 db_path: str = None,
                 auto_skill: bool = True):
        """
        bridge:     HermesScrollBridge instance (optional)
        db_path:    SQLite path (default: HERMES_DIR/arc_patterns.db)
        auto_skill: If True, perfect-score programs become SKILL.md entries
        """
        self._bridge     = bridge
        self._db         = PatternDB(db_path or ARC_PATTERN_DB_PATH)
        self._auto_skill = auto_skill

    def store(self, task_id: str, rule: str, program: str,
              train_score: float,
              pattern_class: str = None,
              task=None) -> PatternRecord:
              # task: optional ARCTask — if provided, BRA task_charge() stored
        """
        Store a pattern in the library.
        Returns the stored PatternRecord.
        """
        cls = pattern_class or classify_pattern(rule)
        root = self._bridge.scroll.eigen() if self._bridge else "0x0"

        rec = PatternRecord(
            task_id       = task_id,
            rule          = rule,
            program       = program,
            train_score   = train_score,
            pattern_class = cls,
            vexel_root    = root,
        )

        # Compute BRA charges and store on record BEFORE insert
        if task is not None:
            try:
                from arc_bra import task_charge, task_structure_charge
                # Identity charge — includes task_id, unique per task
                tc             = task_charge(task)
                rec.bra_hash   = tc.hash
                rec.bra_trace  = tc.trace
                rec.bra_det    = tc.det
                # Structure charge — excludes task_id, matches same pattern elsewhere
                sc             = task_structure_charge(task)
                rec.str_hash   = sc.hash
                rec.str_trace  = sc.trace
                rec.str_det    = sc.det
            except ImportError:
                pass
        self._db.insert(rec)

        # Write to MEMORY.md
        if self._bridge:
            from vexel_flow import EV_RESONANCE, EV_MISS
            entry = rec.memory_entry()
            self._bridge.memory_add(entry, "MEMORY.md")

            # Perfect program → SKILL.md
            if train_score >= 1.0 and self._auto_skill:
                skill_name    = f"arc_{task_id}_{cls}"
                skill_content = rec.skill_content()
                try:
                    self._bridge.skill_create(skill_name, f"ARC {cls} pattern from {task_id}",
                                              skill_content)
                except Exception:
                    pass  # skill already exists or bridge doesn't support it

        return rec

    def lookup(self, query, limit: int = 5) -> list[PatternRecord]:
        """
        BRA charge lookup when query is an ARCTask; category filter otherwise.
        ARCTask  → bra_resonance_score() comparison (integer, deterministic)
        str      → category keyword filter (not text tokenisation)
        """
        try:
            # ARCTask: compute task_charge, do BRA resonance scan
            from arc_bra import task_charge
            if hasattr(query, "train"):  # duck-type ARCTask
                tc = task_charge(query)
                def _to_i64(n): return n if n < (1<<63) else n - (1<<64)
                # Use structure charge for cross-task matching
                # (same training data = same pattern, regardless of task_id)
                from arc_bra import task_structure_charge
                sc = task_structure_charge(query)
                hits = self._db.search_by_charge(
                    _to_i64(sc.hash), sc.trace, sc.det, limit,
                    min_resonance=2, use_structure=True)
                return [rec for _, rec in hits]
        except ImportError:
            pass
        # String: category-filtered lookup
        return self._db.search(query if isinstance(query, str) else "", limit)

    def by_class(self, cls: str, limit: int = 10,
                 min_score: float = 0.5) -> list[PatternRecord]:
        return self._db.by_class(cls, limit, min_score)

    def best_program_for(self, query: str) -> Optional[str]:
        """
        Find the best matching program for a rule description.
        Returns Python source or None.
        """
        results = self.lookup(query, limit=3)
        if not results:
            return None
        # Pick highest train_score
        best = max(results, key=lambda r: r.train_score)
        return best.program if best.train_score > 0.5 else None

    def warm_synthesizer(self, task: ARCTask,
                         rule: str) -> Optional[str]:
        """
        Look up if this task or similar rule already has a perfect program.
        If found, return it — skip synthesis entirely.
        """
        # Exact task match
        exact = self._db.best_for_task(task.task_id)
        if exact and exact.train_score >= 1.0:
            return exact.program

        # Rule similarity search
        cls     = classify_pattern(rule)
        similar = self.by_class(cls, limit=5, min_score=0.9)
        if similar:
            # Return the highest-scoring one
            return similar[0].program
        return None

    def stats(self) -> dict:
        return self._db.stats()

    def close(self):
        self._db.close()


# ── ARC session manager (ties everything together) ────────────────────────────

class ARCSession:
    """
    Manages a full ARC evaluation session:
    - Loads tasks
    - Runs solver with warm-start from pattern library
    - Stores results back to library
    - Records everything in vexel scroll + MEMORY.md
    """

    def __init__(self,
                 bridge=None,
                 verbose: bool = True,
                 pattern_db: str = None,
                 cache_dir: str = None):
        from arc_solver import ARCSolver
        self._bridge  = bridge
        self._solver  = ARCSolver(scroll_bridge=bridge, verbose=verbose)
        self._library = ARCPatternLibrary(bridge=bridge, db_path=pattern_db)
        self._cache   = Path(cache_dir or os.environ.get(
            "ARC_CACHE_DIR", Path.home() / ".arc_cache"))
        self._verbose = verbose

    def _log(self, msg: str):
        if self._verbose:
            print(f"  [ARCSession] {msg}")

    def solve_task(self, task: ARCTask) -> list[ARCPrediction]:
        """Solve one task with warm-start from pattern library."""
        from vexel_flow import EV_QUERY, EV_RESONANCE, EV_MISS

        # Check if we already have a perfect program
        existing = self._library._db.best_for_task(task.task_id)
        if existing and existing.train_score >= 1.0:
            self._log(f"Warm-start: reusing program for {task.task_id} (score=1.0)")
            if self._bridge:
                self._bridge.scroll.record(
                    f"arc_warmstart:{task.task_id}", EV_QUERY, 1)

        preds = self._solver.solve(task)

        # Store results in pattern library
        for pred in preds:
            if pred.program and pred.train_score > 0.0:
                self._library.store(
                    task_id     = task.task_id,
                    rule        = pred.reasoning,
                    program     = pred.program,
                    train_score = pred.train_score,
                )

        return preds

    def run_evaluation(self, tasks: list[ARCTask]) -> dict:
        """
        Evaluate on a list of tasks.
        Stores patterns from successful solves.
        """
        results     = []
        solved      = 0
        total       = len(tasks)

        for i, task in enumerate(tasks):
            self._log(f"Task {i+1}/{total}: {task.task_id}")
            try:
                preds = self.solve_task(task)
                # Score
                all_correct = all(
                    p.correct for p in preds if p.correct is not None)
                if all_correct and preds:
                    solved += 1
                results.append({
                    "task_id":    task.task_id,
                    "solved":     all_correct,
                    "predictions": [p.to_dict() if hasattr(p,'to_dict') else vars(p)
                                    for p in preds],
                })
            except Exception as e:
                self._log(f"Error on {task.task_id}: {e}")
                results.append({
                    "task_id": task.task_id,
                    "solved":  False,
                    "error":   str(e),
                })

        solve_rate = solved / total if total else 0.0
        self._log(f"Solved {solved}/{total} ({solve_rate:.1%})")

        summary = {
            "tasks_total":   total,
            "tasks_solved":  solved,
            "solve_rate":    solve_rate,
            "pattern_stats": self._library.stats(),
            "task_results":  results,
        }
        return summary

    def load_and_run(self, task_ids: list[str],
                     split: str = "training") -> dict:
        """Load tasks from ARC dataset and run evaluation."""
        from arc_types import load_task
        tasks = []
        for tid in task_ids:
            try:
                tasks.append(load_task(tid, split, self._cache))
            except Exception as e:
                self._log(f"Could not load {tid}: {e}")
        return self.run_evaluation(tasks)

    def run_from_files(self, paths: list[str]) -> dict:
        """Load tasks from local JSON files and evaluate."""
        from arc_types import load_task_from_file
        tasks = []
        for p in paths:
            try:
                tasks.append(load_task_from_file(p))
            except Exception as e:
                self._log(f"Could not load {p}: {e}")
        return self.run_evaluation(tasks)

    def close(self):
        self._library.close()


# ── Hermes tool integration ───────────────────────────────────────────────────
#
# arc_solve(task_json_or_id, commit=False) tool for Hermes
# arc_pattern_search(query) tool for Hermes
# arc_pattern_stats() tool for Hermes

ARC_TOOL_SCHEMAS = [
    {
        "name": "arc_solve",
        "description": (
            "Solve an ARC-AGI visual reasoning task. "
            "Accepts a task JSON string, local file path, or task ID from the public dataset. "
            "Returns predicted output grids and confidence scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task JSON string, file path, or task ID (e.g. '007bbfb7')"
                },
                "split": {
                    "type": "string",
                    "enum": ["training", "evaluation"],
                    "default": "training",
                    "description": "ARC dataset split (used when task is an ID)"
                },
                "commit": {
                    "type": "boolean",
                    "default": False,
                    "description": "Commit learned pattern to MEMORY.md"
                },
            },
            "required": ["task"]
        }
    },
    {
        "name": "arc_pattern_search",
        "description": (
            "Search the ARC pattern library for previously learned transformation rules. "
            "Returns matching programs and their training scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Rule description or keywords to search for"
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum results to return"
                },
            },
            "required": ["query"]
        }
    },
    {
        "name": "arc_pattern_stats",
        "description": "Return statistics on the ARC pattern library.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
]


def handle_arc_solve_tool(tool_input: dict, scroll_bridge=None) -> dict:
    from arc_types import ARCTask, load_task, load_task_from_file
    from arc_solver import ARCSolver

    task_ref = tool_input.get("task", "")
    split    = tool_input.get("split", "training")
    commit   = tool_input.get("commit", False)

    if not task_ref:
        return {"error": "arc_solve: task is required"}

    # Load task
    try:
        p = Path(task_ref)
        if p.exists():
            task = load_task_from_file(str(p))
        elif task_ref.startswith("{"):
            task = ARCTask.from_dict(json.loads(task_ref), "inline")
        else:
            task = load_task(task_ref, split)
    except Exception as e:
        return {"error": f"Could not load task: {e}"}

    solver  = ARCSolver(scroll_bridge=scroll_bridge)
    library = ARCPatternLibrary(bridge=scroll_bridge)

    try:
        preds = solver.solve(task)
    except Exception as e:
        return {"error": f"Solver error: {e}"}

    output = []
    for p in preds:
        from arc_types import grid_to_text
        output.append({
            "test_index":  p.test_index,
            "predicted":   grid_to_text(p.predicted) if p.predicted else None,
            "train_score": round(p.train_score, 3),
            "reasoning":   p.reasoning[:200],
            "vexel_root":  p.vexel_root,
            "correct":     p.correct,
        })

        if commit and p.program and p.train_score > 0:
            library.store(task.task_id, p.reasoning, p.program, p.train_score)

    library.close()
    return {
        "task_id":     task.task_id,
        "predictions": output,
        "solved":      all(o["correct"] for o in output if o["correct"] is not None),
    }


def handle_arc_pattern_search_tool(tool_input: dict, scroll_bridge=None) -> dict:
    query = tool_input.get("query", "")
    limit = int(tool_input.get("limit", 5))
    library = ARCPatternLibrary(bridge=scroll_bridge)
    results = library.lookup(query, limit)
    library.close()
    return {
        "query":   query,
        "results": [
            {"task_id": r.task_id, "pattern_class": r.pattern_class,
             "train_score": r.train_score, "rule": r.rule[:200],
             "program_preview": r.program.split("\n")[0][:100]}
            for r in results
        ],
        "count": len(results),
    }


def handle_arc_pattern_stats_tool(tool_input: dict = None, scroll_bridge=None) -> dict:
    library = ARCPatternLibrary(bridge=scroll_bridge)
    stats   = library.stats()
    library.close()
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC Pattern Library CLI")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("stats", help="Show pattern library statistics")

    sp = sub.add_parser("search", help="Search pattern library")
    sp.add_argument("query")

    dp = sub.add_parser("demo", help="Demo pattern storage and retrieval")

    args = parser.parse_args()

    if args.cmd == "stats":
        lib = ARCPatternLibrary()
        print(json.dumps(lib.stats(), indent=2))
        lib.close()

    elif args.cmd == "search":
        lib = ARCPatternLibrary()
        results = lib.lookup(args.query)
        for r in results:
            print(f"  [{r.pattern_class}] {r.task_id} score={r.train_score:.2f}")
            print(f"    Rule: {r.rule[:100]}")
        lib.close()

    elif args.cmd == "demo":
        print("\n" + "="*60)
        print("  ARC Pattern Library Demo")
        print("="*60)
        import tempfile

        tmp_db = Path(tempfile.mkdtemp()) / "arc_test.db"
        lib = ARCPatternLibrary(db_path=str(tmp_db))

        # Store a few patterns
        lib.store("demo_001", "Recolor blue cells to red", "def transform(grid): return recolor(grid, 1, 2)",
                  1.0, "color")
        lib.store("demo_002", "Rotate grid 90 degrees clockwise", "def transform(grid): return rot90(grid)",
                  0.8, "spatial")
        lib.store("demo_003", "Reflect grid horizontally", "def transform(grid): return reflect_h(grid)",
                  1.0, "spatial")

        print(f"\nStored 3 patterns")
        print(json.dumps(lib.stats(), indent=2))

        # Search
        results = lib.lookup("rotate reflect spatial")
        print(f"\nSearch 'rotate reflect spatial' → {len(results)} results:")
        for r in results:
            print(f"  [{r.pattern_class}] {r.task_id} score={r.train_score}")

        best = lib.best_program_for("recolor blue to red")
        print(f"\nbest_program_for('recolor blue to red'):\n  {best}")

        lib.close()
        print("\nDemo complete.")
        print("="*60 + "\n")
