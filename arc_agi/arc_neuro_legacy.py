import json
import os
import threading
from arc_types import Grid
from arc_bra import _PRIM_COORDS, eigen_charge

SYNAPSE_FILE = "/Volumes/Seagate 4tb/unified_field_theory/arc_agi/synapses.json"
_persist_lock = threading.Lock()

class NeuromorphicBrain:
    def __init__(self, spike_threshold: float = 0.6):
        self.base_threshold = spike_threshold
        self.spike_threshold = spike_threshold
        # Each primitive has a resting membrane potential of 0.0
        self.membrane_potentials = {node: 0.0 for node in _PRIM_COORDS.keys()}
        # Synaptic weights for specific task-feature pairings
        self.synapses = self._load_synapses()

    def _load_synapses(self) -> dict:
        if os.path.exists(SYNAPSE_FILE):
            with _persist_lock:
                try:
                    with open(SYNAPSE_FILE, 'r') as f:
                        return json.load(f)
                except:
                    return {}
        return {}

    def _save_synapses(self):
        with _persist_lock:
            try:
                with open(SYNAPSE_FILE, 'w') as f:
                    json.dump(self.synapses, f, indent=2)
            except Exception as e:
                print(f"Error saving synapses: {e}")

    def _apply_stimulus(self, axis_weights: tuple[float, float, float], strength: float = 1.0, task_charge=None):
        """
        Inject electrical stimulus into the UPG topology.
        Signals decay inversely proportional to Euclidean distance in the graph.
        Also applies Synaptic Boost from LTP if a resonance charge exists.
        """
        sx, sy, sz = axis_weights
        
        # Check for persistent synaptic memory if charge is provided
        task_key = str(task_charge.hash) if task_charge else None
        memory_boosts = self.synapses.get(task_key, {}) if task_key else {}

        for prim, coords in _PRIM_COORDS.items():
            cx, cy, cz = coords
            # Simple inverse Euclidean distance propagation
            dist_sq = (cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2
            if dist_sq == 0:
                potential = strength * 1.5  # Hub spike
            else:
                potential = strength / dist_sq
            
            # Apply LTP Boost
            boost = memory_boosts.get(prim, 0.0)
            self.membrane_potentials[prim] += potential + boost

    def learn_success(self, task_grid: Grid, effective_primitives: list[str]):
        """
        Long-Term Potentiation (LTP): Strepthen synapses for successful solve paths.
        Uses EigenCharge to index the task.
        """
        from arc_bra import eigen_charge
        import json
        
        # Serialize grid for hash
        data = json.dumps(task_grid).encode('utf-8')
        charge = eigen_charge(data)
        task_key = str(charge.hash)
        
        if task_key not in self.synapses:
            self.synapses[task_key] = {}
            
        for prim in effective_primitives:
            # Strengthen synapse
            current = self.synapses[task_key].get(prim, 0.0)
            self.synapses[task_key][prim] = current + 0.5 # Potentiation constant
            
        self._save_synapses()

    def neuromodulate(self, candidate_count: int):
        """
        Homeostatic Neuro-Modulation: Adjust threshold based on noise.
        """
        if candidate_count > 10:
            # Too much noise: inhibit
            self.spike_threshold += 0.05
        elif candidate_count == 0:
            # Too quiet: excite (lower threshold)
            self.spike_threshold = max(0.1, self.spike_threshold - 0.05)
        else:
            # Return to base equilibrium slowly
            delta = (self.base_threshold - self.spike_threshold) * 0.1
            self.spike_threshold += delta

    def route_ast(self, input_grid: Grid, output_grid: Grid, com_metrics: dict, 
                  train_pairs: list[dict] = None, memory_boost: list[str] = None,
                  pedagogical_boost: list[str] = None) -> list[str]:
        """
        Execute tiered spreading activation via UPA Escalation.
        Moves from simple OR matching to Conjunction Gates (AND matching).
        
        memory_boost: Stimulus from DAW playback (Phase 15).
        pedagogical_boost: High-level reasoning tracks from Master Tapes (Phase 16).
        """
        from arc_upa import ARCUPALattice
        upa = ARCUPALattice()

        # 0. Feature Extraction (The 'Tokens' of structural geometry)
        from arc_bra import eigen_charge
        import json
        data = json.dumps(input_grid).encode('utf-8')
        charge = eigen_charge(data)
        
        # Synthetic token set from task features
        features = set()
        if com_metrics:
            dx, dy, dz = com_metrics.get("target_coords", (0, 0, 0))
            if dx > 15: features.add("geometric")
            if dy > 15: features.add("ontological")
            if dz > 15: features.add("spectral")
        
        if train_pairs:
            from arc_hard_heuristics import HardHeuristicsEngine as HHE
            if HHE.detect_recursive_tiling(train_pairs):
                features.add("recursive")
                features.add("tiling")
            
            import numpy as np
            if len(HHE.apply_nca_logic(np.array(input_grid))) > 0:
                features.add("nca")
                features.add("cellular")

        # 1. Tier Escalation Logic (Conjunction Gates)
        # Level 1: Immediate/Complex Attractors (Iterative)
        tier1_conjuncts = [{"nca", "cellular"}, {"local", "iterative"}]
        # Level 2: Urgent/Hard Heuristics (Tiling)
        tier2_conjuncts = [{"recursive", "tiling"}, {"relational"}]
        # Level 3: Monitor/Standard Primitives (Geometric)
        tier3_conjuncts = [{"geometric"}, {"ontological"}]

        activated_experts = []
        
        # Check Level 1 (High Complexity / Iterative)
        for group in tier1_conjuncts:
            if group.issubset(features):
                activated_experts.extend(["kernel_local_3x3", "kernel_local_5x5", "nca_flood", "nca_outline"])
                break
        
        # Check Level 2 (Relational / Tiling)
        for group in tier2_conjuncts:
            if group.issubset(features):
                activated_experts.extend(["kernel_period", "kernel_delta", "relational_graph"])
                break

        # Check Level 3 (Baseline / Standard)
        for group in tier3_conjuncts:
            if group.issubset(features):
                activated_experts.extend(["rotate", "flip", "recolor", "outline", "kernel_local_3x3"])
                break

        # 2. Resonate and Spike
        from arc_bra import _PRIM_COORDS
        self.membrane_potentials = {node: 0.0 for node in _PRIM_COORDS.keys()}
        if com_metrics:
            self._apply_stimulus(com_metrics.get("target_coords", (0, 0, 0)), strength=5.0, task_charge=charge)

        # Boost specifically activated experts
        for exp in activated_experts:
            if exp in self.membrane_potentials:
                self.membrane_potentials[exp] += 1.0

        # Phase 15: Memory Boost (Stimulus from DAW tape)
        if memory_boost:
            for exp in memory_boost:
                if exp in self.membrane_potentials:
                    self.membrane_potentials[exp] += 2.0  # Strong signal
                else:
                    self.membrane_potentials[exp] = 2.0   # Action potential trigger

        # Phase 16: Pedagogical Boost (Stimulus from Master Tapes)
        if pedagogical_boost:
            for exp in pedagogical_boost:
                if exp in self.membrane_potentials:
                    self.membrane_potentials[exp] += 4.0  # Extreme signal (Reasoning Priority)
                else:
                    self.membrane_potentials[exp] = 4.0

        spiked_nodes = []
        for prim, potential in self.membrane_potentials.items():
            if potential >= self.spike_threshold:
                spiked_nodes.append((prim, potential))

        spiked_nodes.sort(key=lambda x: x[1], reverse=True)
        self.neuromodulate(len(spiked_nodes))
        
        return [prim for prim, _ in spiked_nodes]
