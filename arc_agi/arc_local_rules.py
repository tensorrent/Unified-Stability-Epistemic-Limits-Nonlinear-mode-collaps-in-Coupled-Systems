import numpy as np
from typing import List, Dict, Any, Tuple
from arc_types import Grid, grid_shape
from arc_bra import charge_neighborhood

class LocalRuleLearner:
    """
    Learns deterministic mappings from 3x3 neighborhood charges to output colors.
    Targeting the 30% gap where global transforms fail.
    """
    def __init__(self):
        # Synaptic map: signature (u64) -> color_frequencies (Dict[u8, int])
        self.synapses: Dict[int, Dict[int, int]] = {}

    def learn_from_pair(self, inp: Grid, out: Grid):
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        
        # Local rules currently assume 1:1 shape mapping for neighborhood learning
        if (ih, iw) != (oh, ow):
            return

        for r in range(ih):
            for c in range(iw):
                charge = charge_neighborhood(inp, r, c)
                target_color = out[r][c]
                
                if charge not in self.synapses:
                    self.synapses[charge] = {}
                
                self.synapses[charge][target_color] = self.synapses[charge].get(target_color, 0) + 1

    def get_deterministic_rules(self) -> Dict[int, int]:
        """Returns map of charge -> most frequent color spike."""
        rules = {}
        for charge, frequencies in self.synapses.items():
            # Winner takes all for the local spike
            winner = max(frequencies.items(), key=lambda x: x[1])[0]
            rules[charge] = winner
        return rules

class LocalRulePropagator:
    """
    Applies learned local neighborhood rules to arbitrary grids.
    """
    @staticmethod
    def apply_layer(grid: Grid, rules: Dict[int, int]) -> Grid:
        h, w = grid_shape(grid)
        out = [row[:] for row in grid] # Deep copy
        
        changed = False
        for r in range(h):
            for c in range(w):
                charge = charge_neighborhood(grid, r, c)
                if charge in rules:
                    new_color = rules[charge]
                    if out[r][c] != new_color:
                        out[r][c] = new_color
                        changed = True
        return out

class AxonPropagator:
    """
    Iterative Neuromorphic Dynamics (NCA style).
    Propagates rules until the grid reaches a stable attractor state.
    """
    @staticmethod
    def propagate_until_stable(grid: Grid, rules: Dict[int, int], max_steps: int = 50) -> Grid:
        from arc_bra import bra_grids_exact
        current = grid
        for i in range(max_steps):
            next_grid = LocalRulePropagator.apply_layer(current, rules)
            if bra_grids_exact(current, next_grid):
                # Stability reached
                return next_grid
            current = next_grid
        return current

if __name__ == "__main__":
    print("Local Rule Engine Initialized.")
