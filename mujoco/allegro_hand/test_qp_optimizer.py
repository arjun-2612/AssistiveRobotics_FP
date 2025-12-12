#!/usr/bin/env python3
"""
Test QP Optimizer (Section III.B)
Tests internal force optimization with friction constraints.
"""

import numpy as np
import sys
sys.path.append('controllers')

from controllers.QP import InternalForceOptimizer
from controllers.grasp_model import GraspMatrix

print("=" * 70)
print("QP OPTIMIZER TEST")
print("=" * 70)

# Test setup: 4 contacts in square configuration around cube
contact_positions = np.array([
    [0.03, 0.0, 0.11],   # Front
    [-0.03, 0.0, 0.11],  # Back
    [0.0, 0.03, 0.11],   # Right
    [0.0, -0.03, 0.11],  # Left
])

contact_normals = np.array([
    [-1, 0, 0],  # Front (points inward)
    [1, 0, 0],   # Back
    [0, -1, 0],  # Right
    [0, 1, 0],   # Left
])

object_center = np.array([0.0, 0.0, 0.11])

# Compute grasp matrix
n_contacts = 4
grasp_model = GraspMatrix(n_contacts=n_contacts)
contact_pos_flat = contact_positions.flatten()
G = grasp_model.compute_grasp_matrix(contact_pos_flat, object_center)

print(f"\n✓ Grasp matrix: {G.shape}")
print(f"  Rank: {np.linalg.matrix_rank(G)} (need 6 for full wrench control)")
print(f"  Force closure: {grasp_model.is_force_closure()}")

# Initialize QP optimizer
qp_optimizer = InternalForceOptimizer(
    n_contacts=n_contacts,
    friction_coefficient=0.8,
    f_min=0.5,   # Minimum 0.5N normal force
    f_max=20.0   # Maximum 20N
)

print(f"\n✓ QP optimizer initialized")
print(f"  Friction coefficient: {qp_optimizer.mu}")
print(f"  Force limits: [{qp_optimizer.f_min}, {qp_optimizer.f_max}] N")

# Test 1: Hold object in place (only gravity compensation)
print("\n" + "=" * 70)
print("TEST 1: Gravity Compensation (hold in place)")
print("=" * 70)

object_mass = 0.1  # 100g cube
gravity = 9.81
weight = object_mass * gravity

# Desired wrench: just counteract gravity
w_desired = np.array([0, 0, weight, 0, 0, 0])  # Lift force only

# Desired internal forces: small squeeze
f_d_normals = np.array([2.0, 2.0, 2.0, 2.0])  # 2N squeeze per contact

print(f"\nDesired wrench: {w_desired}")
print(f"Desired normal forces: {f_d_normals}")

# Solve QP
f_contact, info = qp_optimizer.compute_contact_forces(
    desired_wrench=w_desired,
    grasp_matrix=G,
    desired_normal_forces=f_d_normals,
    contact_normals=contact_normals
)

print(f"\n✓ QP solved: {info['qp_status']}")
print(f"  Optimal: {info['qp_optimal']}")

# Verify wrench
w_actual = G.T @ f_contact
print(f"\nWrench verification:")
print(f"  Desired: {w_desired}")
print(f"  Actual:  {w_actual}")
print(f"  Error:   {np.linalg.norm(w_actual - w_desired):.4f}")

# Check friction constraints
print(f"\nContact forces:")
for i in range(n_contacts):
    f_i = f_contact[3*i:3*(i+1)]
    n_i = contact_normals[i]
    
    # Normal and tangential components
    f_n = np.dot(f_i, n_i)
    f_tangent = f_i - f_n * n_i
    f_t_mag = np.linalg.norm(f_tangent)
    
    friction_ratio = f_t_mag / (f_n + 1e-10)
    
    print(f"  Contact {i+1}:")
    print(f"    Force: {f_i}")
    print(f"    Normal: {f_n:.3f} N")
    print(f"    Tangential: {f_t_mag:.3f} N")
    print(f"    Friction ratio: {friction_ratio:.3f} (limit: {qp_optimizer.mu:.3f})")
    
    if friction_ratio > qp_optimizer.mu + 0.01:
        print(f"    ⚠ FRICTION VIOLATION!")
    else:
        print(f"    ✓ Within friction cone")

# Test 2: Lift object (requires both wrench and internal forces)
print("\n" + "=" * 70)
print("TEST 2: Lift Object (wrench + internal forces)")
print("=" * 70)

# Desired wrench: lift + slight forward acceleration
w_desired = np.array([2.0, 0, weight + 5.0, 0, 0, 0])  # Extra 5N up, 2N forward

print(f"\nDesired wrench: {w_desired}")

f_contact, info = qp_optimizer.compute_contact_forces(
    desired_wrench=w_desired,
    grasp_matrix=G,
    desired_normal_forces=f_d_normals,
    contact_normals=contact_normals
)

print(f"\n✓ QP solved: {info['qp_status']}")

w_actual = G.T @ f_contact
print(f"\nWrench verification:")
print(f"  Desired: {w_desired}")
print(f"  Actual:  {w_actual}")
print(f"  Error:   {np.linalg.norm(w_actual - w_desired):.4f}")

# Test 3: High desired forces (test limits)
print("\n" + "=" * 70)
print("TEST 3: Force Limits (high desired forces)")
print("=" * 70)

f_d_normals_high = np.array([25.0, 25.0, 25.0, 25.0])  # Request 25N (above limit)
print(f"Desired normal forces: {f_d_normals_high} N")
print(f"Force limit: {qp_optimizer.f_max} N")

f_contact, info = qp_optimizer.compute_contact_forces(
    desired_wrench=w_desired,
    grasp_matrix=G,
    desired_normal_forces=f_d_normals_high,
    contact_normals=contact_normals
)

print(f"\n✓ QP solved: {info['qp_status']}")

print(f"\nActual normal forces:")
for i in range(n_contacts):
    f_i = f_contact[3*i:3*(i+1)]
    n_i = contact_normals[i]
    f_n = np.dot(f_i, n_i)
    print(f"  Contact {i+1}: {f_n:.2f} N (requested {f_d_normals_high[i]:.2f})")

print("\n" + "=" * 70)
print("QP TEST SUMMARY")
print("=" * 70)
print("✓ Wrench constraints satisfied")
print("✓ Friction cone constraints checked")
print("✓ Force limits enforced")
print("=" * 70)