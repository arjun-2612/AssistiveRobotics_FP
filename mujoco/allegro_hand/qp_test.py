import numpy as np
from controllers.QP import QPGraspOptimizer

def test_gravity_compensation():
    print("=== Test: Gravity Compensation ===")
    
    # 1. Scene Setup
    # Assume a 0.1kg object floating in the air
    obj_pos = np.array([0.0, 0.0, 0.5])
    obj_mass = 0.1
    g = 9.81
    
    # Two fingertips grasping from the left and right sides of the object (X-axis direction)
    p1 = obj_pos + np.array([-0.05, 0, 0.02]) # Left contact point
    p2 = obj_pos + np.array([0.05, 0, -0.02])    # Right contact point

    # n1 = np.array([1.0, 0, 0])             # Normal pointing right (towards object)
    # n2 = np.array([-1.0, 0, 0])            # Normal pointing left (towards object)

    # --- Case B: Correct normals (Spherical geometry) ---
    # Normal = (Object Center - Contact Point) / Normalized
    # This lets the QP know it implies a curved surface, rather than just squeezing along the X-axis
    n1 = (obj_pos - p1)
    n1 /= np.linalg.norm(n1) # Normalize
    
    n2 = (obj_pos - p2)
    n2 /= np.linalg.norm(n2)
    
    contacts = [p1, p2]
    normals = [n1, n2]
    
    # 2. Simulate Impedance Controller Output
    # Goal: Keep object stationary -> Counteract downward gravity -> Provide upward force +mg
    # Desired Wrench = [Fx, Fy, Fz, Tx, Ty, Tz]
    target_wrench = np.array([0, 0, obj_mass * g, 0, 0, 0])
    print(f"Target Wrench (Counteract Gravity): {target_wrench}")
    
    # 3. Run QP
    qp = QPGraspOptimizer(friction_coef=0.5, n_fingers=2)
    forces = qp.solve(target_wrench, obj_pos, contacts, normals)
    
    print("\n--- QP Calculation Results ---")
    print(f"Finger 1 Force: {forces[0]}")
    print(f"Finger 2 Force: {forces[1]}")
    
    # 4. Verification
    # Verification A: Does the total force equal the target?
    total_force = np.sum(forces, axis=0)
    print(f"\nVerification A (Net Force): Calculated {total_force} vs Target {target_wrench[:3]}")
    
    # Verification B: Is the friction cone satisfied?
    # Theoretically, to generate upward friction to counteract gravity, each finger must apply sufficient normal squeezing force
    # F_friction <= mu * F_normal
    # F_z = 0.5 * mg = 0.49N
    # F_normal >= F_z / mu = 0.49 / 0.5 = 0.98N
    f1_normal = np.dot(forces[0], n1)
    f1_tangent = forces[0] - f1_normal * n1
    f1_tan_mag = np.linalg.norm(f1_tangent)
    
    print(f"Verification B (Friction Cone): Finger 1 Normal Force={f1_normal:.3f}N, Tangential Force={f1_tan_mag:.3f}N")
    print(f"                Friction Coef mu={qp.mu}, Max Allowed Tangential Force={f1_normal * qp.mu:.3f}N")
    
    net_torque = np.zeros(3)
    for i in range(2):
        # 力臂 r = p - COM
        r = contacts[i] - obj_pos
        # 力矩 = r x f
        torque_i = np.cross(r, forces[i])
        net_torque += torque_i
    torque_err = np.linalg.norm(net_torque - target_wrench[3:])
    print(f"                torque err: {torque_err:.6f} Nm")

    if f1_tan_mag <= f1_normal * qp.mu + 1e-4 and torque_err < 1e-2:
        print(">>> Test Passed: Friction cone constraint satisfied! (No slipping)")
    else:
        print(">>> Test Failed: Slippage occurred!")

if __name__ == "__main__":
    test_gravity_compensation()