# -----------------------------------------------------------------------------
# SOVEREIGN INTEGRITY PROTOCOL (SIP) LICENSE v1.1
# Copyright (c) 2026, Bradley Wallace (tensorrent). All rights reserved.
# See SIP_LICENSE.md for full terms.
# -----------------------------------------------------------------------------
"""
stress_test_gamma_overlap.py
============================
Validates the geometry correction factor G_m = Gamma_mmnn / Gamma_m
across multiple graph families, reproducing the GPT stress-test results
and confirming the corrected universal collapse law.
"""

import numpy as np

def ring_laplacian(n):
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 2
        L[i, (i+1) % n] = -1
        L[(i+1) % n, i] = -1
    return L

def grid_laplacian(rows, cols):
    n = rows * cols
    L = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            deg = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    j = nr * cols + nc
                    L[i, j] = -1
                    deg += 1
            L[i, i] = deg
    return L

def er_laplacian(n, p, seed=42):
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.rand() < p:
                A[i, j] = 1
                A[j, i] = 1
    D = np.diag(A.sum(axis=1))
    return D - A

def ba_laplacian(n, m=2, seed=42):
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    # Start with complete graph on m+1 nodes
    for i in range(m+1):
        for j in range(i+1, m+1):
            A[i, j] = 1
            A[j, i] = 1
    for new in range(m+1, n):
        degrees = A.sum(axis=1)[:new]
        probs = degrees / degrees.sum() if degrees.sum() > 0 else np.ones(new)/new
        targets = rng.choice(new, size=m, replace=False, p=probs)
        for t in targets:
            A[new, t] = 1
            A[t, new] = 1
    D = np.diag(A.sum(axis=1))
    return D - A

def star_laplacian(n):
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = 1
        A[i, 0] = 1
    D = np.diag(A.sum(axis=1))
    return D - A

def compute_gamma_stats(L, name, k=1.0):
    eigvals, eigvecs = np.linalg.eigh(L)
    # omega_m = sqrt(k + lambda_m)
    omegas = np.sqrt(k + eigvals)
    
    n = L.shape[0]
    # Skip the zero eigenvalue (mode 0)
    nonzero = [i for i in range(n) if eigvals[i] > 1e-10]
    
    ratios = []
    gammas_m = []
    
    for m in nonzero:
        phi_m = eigvecs[:, m]
        Gamma_m = np.sum(phi_m**4)
        gammas_m.append(Gamma_m)
        
        # Find nearest frequency neighbor
        best_n = None
        best_dist = float('inf')
        for nn in nonzero:
            if nn == m:
                continue
            d = abs(omegas[nn] - omegas[m])
            if d < best_dist:
                best_dist = d
                best_n = nn
        
        if best_n is not None:
            phi_n = eigvecs[:, best_n]
            Gamma_mmnn = np.sum(phi_m**2 * phi_n**2)
            r = Gamma_mmnn / Gamma_m if Gamma_m > 1e-15 else 0
            ratios.append(r)
    
    mean_r = np.mean(ratios) if ratios else 0
    min_r = np.min(ratios) if ratios else 0
    max_r = np.max(ratios) if ratios else 0
    mean_gamma = np.mean(gammas_m) if gammas_m else 0
    
    print(f"  {name:25s}  G_m mean={mean_r:.4f}  min={min_r:.5f}  max={max_r:.4f}  mean_Gamma_m={mean_gamma:.4f}")
    return mean_r

def main():
    print("=" * 75)
    print(" Stress Test: Geometry Correction Factor G_m = Gamma_mmnn / Gamma_m")
    print("=" * 75)
    print()
    
    results = {}
    
    L = ring_laplacian(16)
    results['Ring(16)'] = compute_gamma_stats(L, "Ring (N=16)")
    
    L = grid_laplacian(8, 4)
    results['Grid(8x4)'] = compute_gamma_stats(L, "Grid (8×4, N=32)")
    
    L = er_laplacian(32, 0.2)
    results['ER(32)'] = compute_gamma_stats(L, "Erdős-Rényi (N=32, p=0.2)")
    
    L = ba_laplacian(32, m=2)
    results['BA(32)'] = compute_gamma_stats(L, "Barabási-Albert (N=32, m=2)")
    
    L = star_laplacian(32)
    results['Star(32)'] = compute_gamma_stats(L, "Star (N=32)")
    
    print()
    print("-" * 75)
    print(" Validation Assertions")
    print("-" * 75)
    
    # Standard networks should have G_m ≈ 1/3
    standard = ['Ring(16)', 'Grid(8x4)', 'ER(32)', 'BA(32)']
    for name in standard:
        r = results[name]
        assert 0.2 < r < 0.5, f"{name}: G_m={r:.3f} outside expected range [0.2, 0.5]"
        print(f"  PASS: {name} G_m={r:.4f} ∈ [0.2, 0.5]")
    
    # Star graph should have much lower G_m
    star_r = results['Star(32)']
    assert star_r < 0.15, f"Star: G_m={star_r:.3f} not sufficiently low for inhomogeneous graph"
    print(f"  PASS: Star(32) G_m={star_r:.4f} < 0.15 (extreme localization confirmed)")
    
    # Overall mean for standard networks
    std_mean = np.mean([results[k] for k in standard])
    print(f"\n  Standard network mean G_m = {std_mean:.4f}")
    assert abs(std_mean - 1/3) < 0.05, f"Mean G_m={std_mean:.3f} deviates from 1/3 by more than 0.05"
    print(f"  PASS: Mean G_m ≈ 1/3 (deviation = {abs(std_mean - 1/3):.4f})")
    
    print(f"\n{'=' * 75}")
    print(" ALL STRESS TESTS PASSED.")
    print(f"{'=' * 75}")

if __name__ == "__main__":
    main()
