#!/usr/bin/env python3
"""
Test script for grasp_model.py
Verifies all functionality of the GraspMatrix class.
"""

import numpy as np
import sys
sys.path.append('controllers')

from controllers.grasp_model import GraspMatrix


def test_skew_symmetric():
    """Test the skew-symmetric matrix implementation."""
    print("\n" + "="*60)
    print("TEST 1: Skew-Symmetric Matrix")
    print("="*60)
    
    v = np.array([1.0, 2.0, 3.0])
    skew_v = GraspMatrix._skew_symmetric(v)
    
    print(f"Vector v: {v}")
    print(f"Skew-symmetric matrix:")
    print(skew_v)
    
    # Test property: skew(v) @ u = v × u
    u = np.array([4.0, 5.0, 6.0])
    cross_product_skew = skew_v @ u
    cross_product_numpy = np.cross(v, u)
    
    print(f"\nTest: skew(v) @ u should equal v × u")
    print(f"  skew(v) @ u: {cross_product_skew}")
    print(f"  v × u:       {cross_product_numpy}")
    print(f"  Difference:  {np.linalg.norm(cross_product_skew - cross_product_numpy):.2e}")
    
    if np.allclose(cross_product_skew, cross_product_numpy):
        print("  ✅ PASS: Cross product matches!")
    else:
        print("  ❌ FAIL: Cross product doesn't match!")
        
    # Test anti-symmetry: skew(v)^T = -skew(v)
    is_antisymmetric = np.allclose(skew_v.T, -skew_v)
    print(f"\nTest: skew(v)^T = -skew(v) (anti-symmetry)")
    print(f"  Result: {is_antisymmetric}")
    if is_antisymmetric:
        print("  ✅ PASS: Matrix is anti-symmetric!")
    else:
        print("  ❌ FAIL: Matrix is not anti-symmetric!")


def test_basic_grasp_matrix():
    """Test basic grasp matrix construction."""
    print("\n" + "="*60)
    print("TEST 2: Basic Grasp Matrix Construction")
    print("="*60)
    
    # Simple 2-finger grasp
    grasp = GraspMatrix(n_contacts=2)
    
    # Object at origin
    object_pos = np.array([0.0, 0.0, 0.0])
    
    # Contacts symmetrically placed
    contact_positions = np.array([
        [0.05, 0.0, 0.0],   # Right finger: 5cm to the right
        [-0.05, 0.0, 0.0]   # Left finger: 5cm to the left
    ])
    
    print(f"Object position: {object_pos}")
    print(f"Contact 1 position: {contact_positions[0]}")
    print(f"Contact 2 position: {contact_positions[1]}")
    
    # Compute grasp matrix
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    print(f"\nGrasp Matrix G (6×6):")
    print(G)
    print(f"\nShape: {G.shape}")
    
    # Check dimensions
    expected_shape = (6, 6)  # 2 contacts × 3D = 6 force components
    if G.shape == expected_shape:
        print(f"✅ PASS: Shape is correct {expected_shape}")
    else:
        print(f"❌ FAIL: Expected {expected_shape}, got {G.shape}")
    
    # Check force closure
    has_force_closure = grasp.is_force_closure()
    print(f"\nForce closure: {has_force_closure}")
    rank = np.linalg.matrix_rank(G)
    print(f"Matrix rank: {rank}/6")
    
    if has_force_closure:
        print("✅ PASS: Grasp has force closure (can control all 6 DOF)!")
    else:
        print("⚠️  WARNING: Grasp does NOT have force closure")


def test_symmetric_squeeze():
    """Test symmetric squeeze forces (should produce no net wrench)."""
    print("\n" + "="*60)
    print("TEST 3: Symmetric Squeeze Forces")
    print("="*60)
    
    grasp = GraspMatrix(n_contacts=2)
    object_pos = np.array([0.0, 0.0, 0.0])
    contact_positions = np.array([
        [0.05, 0.0, 0.0],   # Right
        [-0.05, 0.0, 0.0]   # Left
    ])
    
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    # Symmetric inward squeeze: 10N each, pointing at object center
    contact_forces = np.array([
        -10.0, 0.0, 0.0,  # Contact 1: 10N to the left (toward object)
        10.0, 0.0, 0.0    # Contact 2: 10N to the right (toward object)
    ])
    
    print(f"Contact forces (symmetric squeeze):")
    print(f"  Contact 1: {contact_forces[:3]} N")
    print(f"  Contact 2: {contact_forces[3:]} N")
    
    # Compute resulting wrench
    wrench = grasp.contact_forces_to_object_wrench(contact_forces)
    
    print(f"\nResulting object wrench:")
    print(f"  Forces:  {wrench[:3]} N")
    print(f"  Torques: {wrench[3:]} Nm")
    print(f"  Total wrench magnitude: {np.linalg.norm(wrench):.2e}")
    
    # Symmetric squeeze should produce near-zero wrench
    if np.linalg.norm(wrench) < 1e-10:
        print("✅ PASS: Symmetric squeeze produces no net wrench!")
    else:
        print(f"⚠️  WARNING: Expected near-zero wrench, got {np.linalg.norm(wrench):.2e}")


def test_lift_force():
    """Test upward forces (should lift object)."""
    print("\n" + "="*60)
    print("TEST 4: Lifting Forces")
    print("="*60)
    
    grasp = GraspMatrix(n_contacts=2)
    object_pos = np.array([0.0, 0.0, 0.05])
    contact_positions = np.array([
        [0.03, 0.0, 0.05],   # Right
        [-0.03, 0.0, 0.05]   # Left
    ])
    
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    # Both fingers push upward with 5N
    contact_forces = np.array([
        0.0, 0.0, 5.0,   # Contact 1: 5N up
        0.0, 0.0, 5.0    # Contact 2: 5N up
    ])
    
    print(f"Contact forces (both upward):")
    print(f"  Contact 1: {contact_forces[:3]} N")
    print(f"  Contact 2: {contact_forces[3:]} N")
    
    wrench = grasp.contact_forces_to_object_wrench(contact_forces)
    
    print(f"\nResulting object wrench:")
    print(f"  Forces:  {wrench[:3]} N")
    print(f"  Torques: {wrench[3:]} Nm")
    
    # Should produce 10N upward force, no torque
    expected_force = np.array([0.0, 0.0, 10.0])
    expected_torque = np.array([0.0, 0.0, 0.0])
    
    force_error = np.linalg.norm(wrench[:3] - expected_force)
    torque_error = np.linalg.norm(wrench[3:] - expected_torque)
    
    print(f"\nExpected force:  {expected_force} N")
    print(f"Force error:     {force_error:.2e}")
    print(f"Expected torque: {expected_torque} Nm")
    print(f"Torque error:    {torque_error:.2e}")
    
    if force_error < 1e-10 and torque_error < 1e-10:
        print("✅ PASS: Lifting forces produce correct wrench!")
    else:
        print("❌ FAIL: Wrench doesn't match expected values")


def test_rotation_torque():
    """Test forces that create pure rotation."""
    print("\n" + "="*60)
    print("TEST 5: Rotational Torque")
    print("="*60)
    
    grasp = GraspMatrix(n_contacts=2)
    object_pos = np.array([0.0, 0.0, 0.0])
    contact_positions = np.array([
        [0.05, 0.0, 0.0],   # Right, at y=0
        [-0.05, 0.0, 0.0]   # Left, at y=0
    ])
    
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    # Forces in opposite y-directions (create rotation about z-axis)
    contact_forces = np.array([
        0.0, 5.0, 0.0,    # Contact 1: 5N in +y
        0.0, -5.0, 0.0    # Contact 2: 5N in -y
    ])
    
    print(f"Contact forces (opposite tangential):")
    print(f"  Contact 1: {contact_forces[:3]} N")
    print(f"  Contact 2: {contact_forces[3:]} N")
    
    wrench = grasp.contact_forces_to_object_wrench(contact_forces)
    
    print(f"\nResulting object wrench:")
    print(f"  Forces:  {wrench[:3]} N")
    print(f"  Torques: {wrench[3:]} Nm")
    
    # Should produce no net force, but torque about z-axis
    # Torque = r × F for each contact
    # Contact 1: [0.05, 0, 0] × [0, 5, 0] = [0, 0, 0.25]
    # Contact 2: [-0.05, 0, 0] × [0, -5, 0] = [0, 0, 0.25]
    # Total: [0, 0, 0.5] Nm
    
    expected_force = np.array([0.0, 0.0, 0.0])
    expected_torque = np.array([0.0, 0.0, 0.5])
    
    force_error = np.linalg.norm(wrench[:3] - expected_force)
    torque_error = np.linalg.norm(wrench[3:] - expected_torque)
    
    print(f"\nExpected force:  {expected_force} N")
    print(f"Force error:     {force_error:.2e}")
    print(f"Expected torque: {expected_torque} Nm")
    print(f"Torque error:    {torque_error:.2e}")
    
    if force_error < 1e-10 and torque_error < 1e-10:
        print("✅ PASS: Rotational forces produce correct torque!")
    else:
        print("❌ FAIL: Torque doesn't match expected value")


def test_object_twist_to_contacts():
    """Test mapping object motion to contact velocities (Equation 1)."""
    print("\n" + "="*60)
    print("TEST 6: Object Twist to Contact Velocities (Eq. 1)")
    print("="*60)
    
    grasp = GraspMatrix(n_contacts=2)
    object_pos = np.array([0.0, 0.0, 0.0])
    contact_positions = np.array([
        [0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0]
    ])
    
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    # Object moving upward at 0.01 m/s, no rotation
    object_twist = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    
    print(f"Object twist:")
    print(f"  Linear velocity:  {object_twist[:3]} m/s")
    print(f"  Angular velocity: {object_twist[3:]} rad/s")
    
    contact_velocities = grasp.object_twist_to_contact_velocities(object_twist)
    
    print(f"\nContact velocities (ċ = G^T ν):")
    print(f"  Contact 1: {contact_velocities[:3]} m/s")
    print(f"  Contact 2: {contact_velocities[3:]} m/s")
    
    # Both contacts should move up at same speed
    expected_vel = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.01])
    error = np.linalg.norm(contact_velocities - expected_vel)
    
    print(f"\nExpected: both contacts move up at 0.01 m/s")
    print(f"Error: {error:.2e}")
    
    if error < 1e-10:
        print("✅ PASS: Contact velocities correct!")
    else:
        print("❌ FAIL: Contact velocities don't match expected")


def test_internal_forces():
    """Test internal force computation (nullspace)."""
    print("\n" + "="*60)
    print("TEST 7: Internal Forces (Nullspace)")
    print("="*60)
    
    grasp = GraspMatrix(n_contacts=2)
    object_pos = np.array([0.0, 0.0, 0.0])
    contact_positions = np.array([
        [0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0]
    ])
    
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    # Pure squeeze forces (internal)
    contact_forces = np.array([
        -10.0, 0.0, 0.0,  # Squeeze left
        10.0, 0.0, 0.0    # Squeeze right
    ])
    
    print(f"Contact forces (symmetric squeeze):")
    print(f"  Contact 1: {contact_forces[:3]} N")
    print(f"  Contact 2: {contact_forces[3:]} N")
    
    # Compute internal component
    f_internal = grasp.compute_internal_forces(contact_forces)
    
    print(f"\nInternal force component:")
    print(f"  Contact 1: {f_internal[:3]} N")
    print(f"  Contact 2: {f_internal[3:]} N")
    print(f"  Magnitude: {np.linalg.norm(f_internal):.3f} N")
    
    # Check that internal forces produce no wrench
    wrench = grasp.contact_forces_to_object_wrench(f_internal)
    wrench_magnitude = np.linalg.norm(wrench)
    
    print(f"\nWrench from internal forces: {wrench}")
    print(f"Wrench magnitude: {wrench_magnitude:.2e}")
    
    if wrench_magnitude < 1e-10:
        print("✅ PASS: Internal forces produce no object wrench!")
    else:
        print("⚠️  WARNING: Internal forces should produce zero wrench")


def test_3_contacts():
    """Test with 3 contacts (more common realistic scenario)."""
    print("\n" + "="*60)
    print("TEST 8: Three-Contact Grasp")
    print("="*60)
    
    grasp = GraspMatrix(n_contacts=3)
    object_pos = np.array([0.0, 0.0, 0.05])
    
    # 3 contacts in triangular formation
    contact_positions = np.array([
        [0.03, 0.0, 0.05],      # Contact 1
        [-0.015, 0.026, 0.05],  # Contact 2 (120° apart)
        [-0.015, -0.026, 0.05]  # Contact 3 (120° apart)
    ])
    
    print(f"Object position: {object_pos}")
    print(f"Contact positions:")
    for i, pos in enumerate(contact_positions):
        print(f"  Contact {i+1}: {pos}")
    
    G = grasp.compute_grasp_matrix(contact_positions, object_pos)
    
    print(f"\nGrasp Matrix shape: {G.shape}")
    print(f"Expected shape: (6, 9)  [6 wrench DOF × 9 force components]")
    
    # Check force closure
    has_force_closure = grasp.is_force_closure()
    rank = np.linalg.matrix_rank(G)
    
    print(f"\nForce closure: {has_force_closure}")
    print(f"Matrix rank: {rank}/6")
    
    if has_force_closure:
        print("✅ PASS: 3-contact grasp has force closure!")
    else:
        print("⚠️  WARNING: Grasp lacks force closure")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*60)
    print("GRASP MATRIX TEST SUITE")
    print("Testing: grasp_model.py (Section II.A)")
    print("="*60)
    
    tests = [
        test_skew_symmetric,
        test_basic_grasp_matrix,
        test_symmetric_squeeze,
        test_lift_force,
        test_rotation_torque,
        test_object_twist_to_contacts,
        test_internal_forces,
        test_3_contacts
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()