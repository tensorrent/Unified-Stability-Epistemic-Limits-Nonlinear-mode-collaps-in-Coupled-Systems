# Copyright (c) 2026 Brad Wallace. All rights reserved.
# Subject to Sovereign Integrity Protocol License (SIP License v1.1).
# See SIP_LICENSE.md for full terms.
"""
generate_topologies.py
======================
Generates the definitive suite of graph topologies for testing the
Nonlinear Mode Collapse Scaling Law: 
    beta_c a_m^2 = (8 omega_m / 3 Gamma_m) Delta_omega_m

Categories:
    1. Symmetric & Regular Baseline (Ring, Grid)
    2. Complex Networks (Erdos-Renyi, Barabasi-Albert, Watts-Strogatz)
    3. Stress Tests (Star, Dumbbell)
"""

import networkx as nx
import numpy as np

def compute_theoretical_metrics(name, G, k_stiffness=1.0):
    print(f"\n--- Analyzing Topology: {name} (N={G.number_of_nodes()}) ---")
    
    # Ensure graph is connected (mostly for Dumbbell safety or random nets)
    if not nx.is_connected(G):
        print("  WARNING: Graph is disconnected!")
        
    L = nx.laplacian_matrix(G).toarray()
    
    # We use eigh since Laplacian is symmetric
    evals, evecs = np.linalg.eigh(L)
    
    # omega_j = sqrt(k + lambda_j)
    omegas = np.sqrt(k_stiffness + evals)
    N = G.number_of_nodes()
    
    # Let's inspect a few characteristic modes for each graph
    # Mode 1 is the Fiedler mode (lowest non-zero AC mode)
    # Mode N//2 is a mid-frequency mode
    # Mode N-1 is the highest frequency mode
    interesting_modes = [1, N // 2, N - 1]
    
    for m in interesting_modes:
        omega_m = omegas[m]
        phi_m = evecs[:, m]
        
        # Gamma_m = sum(phi_m(i)^4)
        Gamma_m = np.sum(phi_m**4)
        
        # Calculate nearest spectral gap Delta_omega_m
        gaps = np.abs(omegas - omega_m)
        gaps[m] = np.inf # exclude self
        Delta_omega_m = np.min(gaps)
        
        # Theoretical threshold prediction for beta_c * a_m^2
        if Gamma_m > 0:
            predicted = (8.0 * omega_m * Delta_omega_m) / (3.0 * Gamma_m)
        else:
            predicted = np.nan
            
        print(f"Mode {m:2d} | ω: {omega_m:.4f} | Δω: {Delta_omega_m:.6f} | Γ (Loc): {Gamma_m:.4f} | Pred β_c*a^2: {predicted:.5f}")

def main():
    # Base stiffness factor (k)
    k_stiffness = 1.0 
    
    # ==========================================
    # 1. The "Symmetric & Regular" Baseline
    # ==========================================
    compute_theoretical_metrics("Ring Graph (1D Lattice)", nx.cycle_graph(32), k_stiffness)
    compute_theoretical_metrics("2D Grid (8x4)", nx.grid_2d_graph(8, 4), k_stiffness)
    
    # ==========================================
    # 2. The "Complex Network" Suite
    # ==========================================
    compute_theoretical_metrics("Erdős-Rényi (p=0.2)", nx.erdos_renyi_graph(32, 0.2, seed=42), k_stiffness)
    compute_theoretical_metrics("Barabási-Albert (m=2)", nx.barabasi_albert_graph(32, 2, seed=42), k_stiffness)
    compute_theoretical_metrics("Watts-Strogatz (k=4, p=0.1)", nx.watts_strogatz_graph(32, 4, 0.1, seed=42), k_stiffness)
    
    # ==========================================
    # 3. The "Stress Tests" (Edge Cases)
    # ==========================================
    compute_theoretical_metrics("Star Graph (Extreme Localization)", nx.star_graph(31), k_stiffness) # 1 center, 31 leaves
    
    # Dumbbell: Two complete N=15 graphs connected by a single bridge
    dumbbell = nx.empty_graph(30)
    for i in range(15):
        for j in range(i+1, 15):
            dumbbell.add_edge(i, j)          # First cluster
            dumbbell.add_edge(i+15, j+15)    # Second cluster
    dumbbell.add_edge(0, 15) # Bridge edge
    compute_theoretical_metrics("Dumbbell Graph (Tiny Spectral Gap)", dumbbell, k_stiffness)

if __name__ == "__main__":
    main()
