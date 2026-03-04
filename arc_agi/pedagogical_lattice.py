from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
from arc_bra import _PRIM_COORDS

@dataclass
class ReasoningNode:
    name: str
    level: int    # 0-20 (K to Post-Grad)
    domain: int   # 0-5 (Arithmetic, Logic, Geometry, Algebra, Algorithms, Topology)
    complexity: int # 0-3 (Observation, Deduction, Induction, Abstraction)
    prime: int    # Associated prime ID
    coordinate: Tuple[int, int, int]

class PedagogicalLattice:
    """
    3D Lattice mapping educational milestones to semantic coordinates.
    Extends the UPA concept for meta-reasoning.
    """
    def __init__(self):
        self.nodes: Dict[str, ReasoningNode] = {}
        self._initialize_lattice()

    def _initialize_lattice(self):
        # Educational levels: K=0, 12=12th, 16=BS, 18=MS, 20=PhD
        # Domains: Arithmetic=0, Logic=1, Geometry=2, Algebra=3, Algorithms=4, Topology=5
        
        milestones = [
            # K-5: Basic Observation & Arithmetic
            ("Visual Counting", 0, 0, 0, 2),
            ("Shape Recognition", 1, 2, 0, 3),
            ("Color Invariance", 2, 0, 0, 5),
            ("Grid Neighbors", 3, 2, 1, 7),
            
            # 6-8: Logical Deduction & Basic Algebra
            ("Symmetry Detection", 6, 2, 1, 11),
            ("Translation Heuristic", 7, 2, 1, 13),
            ("Variable Color Swap", 8, 3, 1, 17),
            
            # 9-12: Pattern Induction & Algorithms
            ("Iterative Filling", 10, 4, 2, 19),
            ("Recursive Tiling", 11, 4, 3, 23),
            ("Pathfinding Logic", 12, 4, 2, 29),
            
            # College: Advanced Abstraction
            ("Group Symmetry", 14, 3, 3, 31),
            ("Affine Transformations", 15, 2, 3, 37),
            ("Dynamic Programming", 16, 4, 3, 41),
            
            # Graduate+: Meta-Reasoning
            ("Topology Invariants", 18, 5, 3, 43),
            ("Cellular Automata (NCA)", 19, 4, 3, 47),
            ("Universal Prime Routing", 20, 1, 3, 53),
        ]
        
        for name, lvl, dom, comp, p in milestones:
            # Coordinate = (Level, Domain, Complexity)
            coords = (lvl, dom, comp)
            self.nodes[name] = ReasoningNode(name, lvl, dom, comp, p, coords)

    def get_nearest_heuristics(self, target_coord: Tuple[int, int, int], radius: float = 5.0) -> List[ReasoningNode]:
        """Find reasoning nodes near a task's semantic signature."""
        tx, ty, tz = target_coord
        results = []
        for node in self.nodes.values():
            nx, ny, nz = node.coordinate
            dist = math.sqrt((tx-nx)**2 + (ty-ny)**2 + (tz-nz)**2)
            if dist <= radius:
                results.append(node)
        return sorted(results, key=lambda n: n.level)

    def get_prime_for(self, heuristic_name: str) -> int:
        node = self.nodes.get(heuristic_name)
        return node.prime if node else 1 # Default to 1 (neutral)
