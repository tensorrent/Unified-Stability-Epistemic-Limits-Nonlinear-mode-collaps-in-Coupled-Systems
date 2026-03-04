"""
test_integer_constraints.py - Verification for Phase 19
======================================================
"""
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from arc_integer_constraints import ConstraintState, compute_theta_m, verify_mode_collapse_safety, rc1_audit_trace

def test_paper_example():
    """Verifies math against Section 11.3 of the paper."""
    print("--- Test: Paper Example (Section 11.3) ---")
    
    # Paper states: W=1224, G=1, D=266, B=1021, A=1000 -> Theta = 8,682,240,000
    state = ConstraintState(
        beta=1021,
        amplitude=1000,
        omega=1224,
        gamma=1,
        delta_omega=266
    )
    
    theta = compute_theta_m(state)
    expected_theta = 8682240000
    
    print(f"Calculated Theta: {theta}")
    print(f"Expected Theta:   {expected_theta}")
    
    # Tolerance due to ceiling math and integer truncation in paper's manual calculation
    # Paper says: (8 * 1224 * 10000 * 266) / 3 = 8,682,240,000 exactly since 1224 is div by 3.
    # 1224 / 3 = 408. 
    # 8 * 408 * 10000 * 266 = 3264 * 10000 * 266 = 32,640,000 * 266 = 8,682,240,000.
    
    assert theta == expected_theta, f"Theta mismatch! Got {theta}, expected {expected_theta}"
    
    # Check current B*A^2
    current = state.beta * (state.amplitude**2) # 1021 * 1,000,000 = 1,021,000,000
    print(f"B * A^2:          {current}")
    
    is_safe = verify_mode_collapse_safety(state)
    print(f"Is Safe:          {is_safe}")
    assert is_safe == True
    
    print("Audit Trace Preview:")
    print(rc1_audit_trace(state))
    print("PASSED\n")

def test_failure_case():
    """Verifies that the boundary correctly triggers on high beta."""
    print("--- Test: Failure Case (High Beta) ---")
    
    # Increase beta to exceed theta
    state = ConstraintState(
        beta=9000000, # 9M * 1M = 9,000,000,000 > 8,682,240,000
        amplitude=1000,
        omega=1224,
        gamma=1,
        delta_omega=266
    )
    
    is_safe = verify_mode_collapse_safety(state)
    print(f"Is Safe:          {is_safe}")
    assert is_safe == False
    
    print("Audit Trace Preview:")
    print(rc1_audit_trace(state))
    print("PASSED\n")

if __name__ == "__main__":
    try:
        test_paper_example()
        test_failure_case()
        print("All Phase 19 mathematical tests PASSED.")
    except Exception as e:
        print(f"Verification FAILED: {e}")
        sys.exit(1)
