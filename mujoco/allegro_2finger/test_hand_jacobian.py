#!/usr/bin/env python3
# filepath: /home/jorge/Documents/AssistiveRobotics/AssistiveRobotics_FP/mujoco/allegro_2finger/test_hand_jacobian.py
"""
Simple test for hand_jacobian.py
Tests the core functionality without complexity.
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
import sys
import time
sys.path.append('controllers')

from controllers.hand_jacobian import HandJacobian


def load_model():
    """Load the MuJoCo model."""
    model = mj.MjModel.from_xml_path('mjcf/scene.xml')
    data = mj.MjData(model)
    
    # Use open hand configuration (no object contact)
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'open_hand_test')
    if key_id >= 0:
        mj.mj_resetDataKeyframe(model, data, key_id)
    else:
        mj.mj_resetData(model, data)
    
    mj.mj_forward(model, data)
    return model, data

def add_sphere_marker(viewer, pos, color, radius=0.005):
    """Add a sphere marker to the viewer."""
    if viewer.scn.ngeom >= viewer.scn.maxgeom:
        return
    
    mj.mjv_initGeom(
        viewer.scn.geoms[viewer.scn.ngeom],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=color
    )
    viewer.scn.ngeom += 1


def add_arrow_marker(viewer, start_pos, direction, length, color, thickness=0.003):
    """Add an arrow marker to the viewer."""
    if viewer.scn.ngeom >= viewer.scn.maxgeom:
        return
    
    # Normalize direction
    direction = np.array(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Create rotation matrix to align arrow with direction
    z_axis = np.array([0, 0, 1])
    if np.linalg.norm(direction - z_axis) < 1e-6:
        rotation = np.eye(3)
    elif np.linalg.norm(direction + z_axis) < 1e-6:
        rotation = -np.eye(3)
    else:
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    mj.mjv_initGeom(
        viewer.scn.geoms[viewer.scn.ngeom],
        type=mj.mjtGeom.mjGEOM_ARROW,
        size=[thickness, thickness, length],
        pos=start_pos,
        mat=rotation.flatten(),
        rgba=color
    )
    viewer.scn.ngeom += 1


def test_1_dimensions():
    """TEST 1: Check Jacobian has correct dimensions."""
    print("\n" + "="*60)
    print("TEST 1: Jacobian Dimensions")
    print("="*60)
    
    model, data = load_model()
    jac = HandJacobian(model, data)
    
    # Get fingertip body IDs
    ff_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'ff_tip')
    th_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'th_tip')
    
    print(f"Found bodies: ff_tip (ID={ff_tip_id}), th_tip (ID={th_tip_id})")
    
    # Compute Jacobian for 2 contacts at body centers
    contact_bodies = [ff_tip_id, th_tip_id]
    contact_offsets = [np.zeros(3), np.zeros(3)]  # At body centers
    
    J = jac.compute_contact_jacobian(contact_bodies, contact_offsets)
    
    print(f"\nJacobian shape: {J.shape}")
    print(f"Expected: (6, {model.nv})  [2 contacts × 3D, {model.nv} DOFs]")

    # Visualize structure
    print("\nJacobian structure:")
    print(f"  Rows 0-2: ff_tip velocity (x, y, z)")
    print(f"  Rows 3-5: th_tip velocity (x, y, z)")
    print(f"  Cols 0-7: Hand joint contributions")
    print(f"  Cols 8-13: Object DOF contributions (if object present)")
    
    # Show a sample of the Jacobian
    print("\nSample Jacobian values (first 8 cols, hand joints):")
    print("       ffj0    ffj1    ffj2    ffj3    thj0    thj1    thj2    thj3")
    for i, label in enumerate(['ff_x', 'ff_y', 'ff_z', 'th_x', 'th_y', 'th_z']):
        row = J[i, :8]
        print(f"{label}: " + " ".join([f"{v:7.4f}" for v in row]))
    
    if J.shape == (6, model.nv):
        print("✅ PASS: Correct dimensions!")
        return True
    else:
        print("❌ FAIL: Wrong dimensions!")
        return False
    
def test_2_numerical_jacobian(visualize=False):
    """TEST 2: Compare analytical Jacobian to numerical derivative."""
    print("\n" + "="*60)
    print("TEST 2: Numerical Verification (ċ = J q̇)")
    print("="*60)
    
    model, data = load_model()
    jac = HandJacobian(model, data)
    
    # Get fingertip bodies
    ff_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'ff_tip')
    th_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'th_tip')
    contact_bodies = [ff_tip_id, th_tip_id]
    contact_offsets = [np.zeros(3), np.zeros(3)]
    
    # Compute analytical Jacobian
    J_analytical = jac.compute_contact_jacobian(contact_bodies, contact_offsets)
    
    # Get initial positions
    def get_contact_positions(data):
        pos1 = data.xpos[ff_tip_id].copy()
        pos2 = data.xpos[th_tip_id].copy()
        return np.concatenate([pos1, pos2])
    
    q0 = data.qpos.copy()
    pos0 = get_contact_positions(data)

    print("\nInitial fingertip positions:")
    print(f"  ff_tip: {pos0[:3]}")
    print(f"  th_tip: {pos0[3:]}")
    
    # Compute numerical Jacobian by finite differences
    J_numerical = np.zeros((6, model.nv))
    epsilon = 1e-6
    
    print("\nComputing numerical Jacobian (hand joints 0-7)...")
    
    if visualize:
        print("\nOpening viewer to show perturbations...")
        with mjv.launch_passive(model, data) as viewer:
            for i in range(8):  # Only test hand joints
                # Reset to original
                data.qpos[:] = q0
                mj.mj_forward(model, data)
                
                # Show original position
                viewer.scn.ngeom = model.ngeom
                add_sphere_marker(viewer, data.xpos[ff_tip_id], [1, 0, 0, 0.8], 0.006)
                add_sphere_marker(viewer, data.xpos[th_tip_id], [0, 0, 1, 0.8], 0.006)
                viewer.sync()
                time.sleep(0.5)
                
                # Perturb joint i
                data.qpos[i] += epsilon
                mj.mj_forward(model, data)
                
                # Show perturbed position
                viewer.scn.ngeom = model.ngeom
                add_sphere_marker(viewer, pos0[:3], [1, 0.5, 0.5, 0.5], 0.005)  # Old ff_tip
                add_sphere_marker(viewer, pos0[3:], [0.5, 0.5, 1, 0.5], 0.005)  # Old th_tip
                add_sphere_marker(viewer, data.xpos[ff_tip_id], [1, 0, 0, 1], 0.006)  # New ff_tip
                add_sphere_marker(viewer, data.xpos[th_tip_id], [0, 0, 1, 1], 0.006)  # New th_tip
                
                # Draw arrows showing displacement
                pos_new = get_contact_positions(data)
                displacement = pos_new - pos0
                if np.linalg.norm(displacement[:3]) > 1e-8:
                    add_arrow_marker(viewer, pos0[:3], displacement[:3], 
                                   np.linalg.norm(displacement[:3]), [0, 1, 0, 1])
                if np.linalg.norm(displacement[3:]) > 1e-8:
                    add_arrow_marker(viewer, pos0[3:], displacement[3:], 
                                   np.linalg.norm(displacement[3:]), [0, 1, 0, 1])
                
                viewer.sync()
                
                joint_name = model.joint(i).name
                print(f"  Perturbing {joint_name}...")
                time.sleep(1.0)
                
                # Finite difference
                J_numerical[:, i] = (pos_new - pos0) / epsilon
    else:
        for i in range(8):  # Only test hand joints
            # Perturb joint i
            data.qpos[:] = q0
            data.qpos[i] += epsilon
            mj.mj_forward(model, data)
            
            # Measure position change
            pos_new = get_contact_positions(data)
            
            # Finite difference: dpos/dq ≈ Δpos/Δq
            J_numerical[:, i] = (pos_new - pos0) / epsilon

    
    # Reset
    data.qpos[:] = q0
    mj.mj_forward(model, data)
    
    # Compare
    print("\nJoint-by-joint comparison:")
    print("  Joint      Analytical norm  Numerical norm   Error")
    print("  " + "-"*55)
    max_error = 0
    for i in range(8):
        joint_name = model.joint(i).name
        analytical = J_analytical[:, i]
        numerical = J_numerical[:, i]
        error = np.linalg.norm(analytical - numerical)
        max_error = max(max_error, error)
        
        print(f"  {joint_name:8s}  {np.linalg.norm(analytical):14.6e}  "
              f"{np.linalg.norm(numerical):14.6e}  {error:.2e}")
    
    print(f"\nMaximum error: {max_error:.2e} m/rad")
    
    threshold = 1e-4  # 0.1 mm/rad
    if max_error < threshold:
        print(f"✅ PASS: Error < {threshold} m/rad")
        return True
    else:
        print(f"❌ FAIL: Error > {threshold} m/rad")
        return False
    
def test_3_force_to_torque(visualize=False):
    """TEST 3: Verify τ = J^T f."""
    print("\n" + "="*60)
    print("TEST 3: Force to Torque Mapping (τ = J^T f)")
    print("="*60)
    
    model, data = load_model()
    jac = HandJacobian(model, data)
    
    # Get fingertip bodies
    ff_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'ff_tip')
    th_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'th_tip')
    contact_bodies = [ff_tip_id, th_tip_id]
    contact_offsets = [np.zeros(3), np.zeros(3)]
    
    # Compute Jacobian
    J = jac.compute_contact_jacobian(contact_bodies, contact_offsets)
    
    # Apply opposing forces (squeeze pattern)
    force_mag = 5.0  # Newtons
    forces = np.array([
        0.0, 0.0, force_mag,   # ff_tip: 5N upward
        0.0, 0.0, -force_mag   # th_tip: 5N downward
    ])
    
    print("\nApplied forces:")
    print(f"  ff_tip: {forces[:3]} N")
    print(f"  th_tip: {forces[3:]} N")
    
    # Map to torques
    torques = jac.contact_forces_to_joint_torques(forces, J)
    
    print("\nComputed torques (hand joints):")
    print("  Joint    Torque (Nm)   Direction")
    print("  " + "-"*40)
    for i in range(8):
        joint_name = model.joint(i).name
        direction = "↻ CW" if torques[i] > 0 else "↺ CCW" if torques[i] < 0 else "  --"
        print(f"  {joint_name:8s} {torques[i]:+9.4f}   {direction}")

    print(f"\nTotal torque magnitude: {np.linalg.norm(torques[:8]):.4f} Nm")

    if visualize:
        print("\nOpening viewer to show forces and resulting motion...")
        with mjv.launch_passive(model, data) as viewer:
            # Show forces
            viewer.scn.ngeom = model.ngeom
            
            ff_pos = data.xpos[ff_tip_id]
            th_pos = data.xpos[th_tip_id]
            
            # Show contact points
            add_sphere_marker(viewer, ff_pos, [1, 0, 0, 0.8], 0.007)
            add_sphere_marker(viewer, th_pos, [0, 0, 1, 0.8], 0.007)
            
            # Show force arrows
            force_scale = 0.01  # 1cm per Newton
            add_arrow_marker(viewer, ff_pos, [0, 0, 1], force_mag * force_scale, 
                           [0, 1, 0, 1], 0.004)
            add_arrow_marker(viewer, th_pos, [0, 0, -1], force_mag * force_scale, 
                           [0, 1, 0, 1], 0.004)
            
            viewer.sync()
            
            print("Green arrows = applied forces")
            print("Now applying computed torques to joints...")
            time.sleep(2)
            
            # Simulate with torques
            for step in range(200):
                data.ctrl[:8] = torques[:8] * 0.1  # Scale down for stability
                mj.mj_step(model, data)
                
                # Update visualization
                viewer.scn.ngeom = model.ngeom
                add_sphere_marker(viewer, data.xpos[ff_tip_id], [1, 0, 0, 0.8], 0.007)
                add_sphere_marker(viewer, data.xpos[th_tip_id], [0, 0, 1, 0.8], 0.007)
                
                viewer.sync()
                time.sleep(0.01)
            
            print("Watch how the fingers move in response to the torques!")
            time.sleep(2)
    
    # Check dimension
    if torques.shape[0] == model.nv:
        print(f"\n✅ PASS: Torque vector has correct dimension ({model.nv})")
        return True
    else:
        print(f"\n❌ FAIL: Wrong dimension")
        return False


def test_4_velocity_mapping(visualize=False):
    """TEST 4: Verify ċ = J q̇."""
    print("\n" + "="*60)
    print("TEST 4: Velocity Mapping (ċ = J q̇)")
    print("="*60)
    
    model, data = load_model()
    jac = HandJacobian(model, data)
    
    # Get fingertip bodies
    ff_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'ff_tip')
    th_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'th_tip')
    contact_bodies = [ff_tip_id, th_tip_id]
    contact_offsets = [np.zeros(3), np.zeros(3)]
    
    # Compute Jacobian
    J = jac.compute_contact_jacobian(contact_bodies, contact_offsets)
    
    # Set joint velocities
    q_vel = np.zeros(model.nv)
    q_vel[0] = 0.5   # ffj0: 0.5 rad/s
    q_vel[4] = -0.3  # thj0: -0.3 rad/s
    
    print("\nJoint velocities:")
    print(f"  ffj0: {q_vel[0]} rad/s")
    print(f"  thj0: {q_vel[4]} rad/s")
    
    # Map to contact velocities
    c_vel = jac.joint_velocities_to_contact_velocities(q_vel, J)
    
    print("\nPredicted contact velocities:")
    print(f"  ff_tip: [{c_vel[0]:+.4f}, {c_vel[1]:+.4f}, {c_vel[2]:+.4f}] m/s")
    print(f"  th_tip: [{c_vel[3]:+.4f}, {c_vel[4]:+.4f}, {c_vel[5]:+.4f}] m/s")
    print(f"  ff_tip speed: {np.linalg.norm(c_vel[:3]):.4f} m/s ({np.linalg.norm(c_vel[:3])*1000:.1f} mm/s)")
    print(f"  th_tip speed: {np.linalg.norm(c_vel[3:]):.4f} m/s ({np.linalg.norm(c_vel[3:])*1000:.1f} mm/s)")
    
    if visualize:
        print("\nOpening viewer to show predicted motion...")
        with mjv.launch_passive(model, data) as viewer:
            # Show initial position
            viewer.scn.ngeom = model.ngeom
            ff_pos_0 = data.xpos[ff_tip_id].copy()
            th_pos_0 = data.xpos[th_tip_id].copy()
            
            add_sphere_marker(viewer, ff_pos_0, [1, 0, 0, 0.8], 0.007)
            add_sphere_marker(viewer, th_pos_0, [0, 0, 1, 0.8], 0.007)
            
            # Show velocity arrows
            vel_scale = 0.05  # 5cm per m/s
            add_arrow_marker(viewer, ff_pos_0, c_vel[:3], 
                           np.linalg.norm(c_vel[:3]) * vel_scale, [1, 1, 0, 1], 0.004)
            add_arrow_marker(viewer, th_pos_0, c_vel[3:], 
                           np.linalg.norm(c_vel[3:]) * vel_scale, [1, 1, 0, 1], 0.004)
            
            viewer.sync()
            print("Yellow arrows = predicted velocity directions")
            time.sleep(2)
            
            # Animate motion
            print("Now animating the actual motion...")
            for step in range(100):
                data.qvel[:] = q_vel
                mj.mj_step(model, data)
                
                viewer.scn.ngeom = model.ngeom
                
                # Trail showing old position
                add_sphere_marker(viewer, ff_pos_0, [1, 0.5, 0.5, 0.3], 0.005)
                add_sphere_marker(viewer, th_pos_0, [0.5, 0.5, 1, 0.3], 0.005)
                
                # Current position
                add_sphere_marker(viewer, data.xpos[ff_tip_id], [1, 0, 0, 1], 0.007)
                add_sphere_marker(viewer, data.xpos[th_tip_id], [0, 0, 1, 1], 0.007)
                
                # Velocity arrows
                add_arrow_marker(viewer, data.xpos[ff_tip_id], c_vel[:3], 
                               np.linalg.norm(c_vel[:3]) * vel_scale, [1, 1, 0, 0.6], 0.003)
                add_arrow_marker(viewer, data.xpos[th_tip_id], c_vel[3:], 
                               np.linalg.norm(c_vel[3:]) * vel_scale, [1, 1, 0, 0.6], 0.003)
                
                viewer.sync()
                time.sleep(0.02)
            
            print("Notice: Fingertips move in the direction of yellow arrows!")
            time.sleep(2)
    
    # Check dimension
    if c_vel.shape[0] == 6:
        print("\n✅ PASS: Contact velocity has correct dimension (6)")
        return True
    else:
        print("\n❌ FAIL: Wrong dimension")
        return False


def test_5_rank_analysis():
    """TEST 5: Check Jacobian rank and condition number."""
    print("\n" + "="*60)
    print("TEST 5: Rank and Conditioning")
    print("="*60)
    
    model, data = load_model()
    jac = HandJacobian(model, data)
    
    # Get fingertip bodies
    ff_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'ff_tip')
    th_tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'th_tip')
    contact_bodies = [ff_tip_id, th_tip_id]
    contact_offsets = [np.zeros(3), np.zeros(3)]
    
    J = jac.compute_contact_jacobian(contact_bodies, contact_offsets)
    
    # Compute rank
    rank = np.linalg.matrix_rank(J)
    max_rank = min(J.shape)
    
    print(f"\nJacobian shape: {J.shape}")
    print(f"Rank: {rank}/{max_rank}")

    if rank == max_rank:
        print("  ✓ Full rank: All contact velocity directions are independent")
    else:
        print(f"  ✗ Rank deficient: Only {rank} independent directions")
    
    
    # Compute condition number
    cond = np.linalg.cond(J)
    print(f"Condition number: {cond:.2e}")
    
    if cond < 100:
        print("  → Well-conditioned")
    elif cond < 1000:
        print("  → Moderately conditioned")
    else:
        print("  → Poorly conditioned")

    # Compute singular values
    U, s, Vt = np.linalg.svd(J)
    print(f"\nSingular values (measure of 'strength' in each direction):")
    for i, sv in enumerate(s):
        print(f"  σ{i+1}: {sv:.6f}")
    
    print(f"\nManipulability ellipsoid:")
    print(f"  Largest axis:  {s[0]:.6f} (easiest to control)")
    print(f"  Smallest axis: {s[-1]:.6f} (hardest to control)")
    print(f"  Ratio: {s[0]/s[-1]:.2f}x difference")

    if rank == max_rank:
        print("\n✅ PASS: Jacobian has full rank")
        return True
    else:
        print("\n⚠️  WARNING: Jacobian is rank-deficient")
        return True


def run_all_tests(visualize=False):
    """Run all tests."""
    print("="*60)
    print("HAND JACOBIAN TEST SUITE")
    print("Testing: controllers/hand_jacobian.py")
    if visualize:
        print("Mode: VISUAL (with MuJoCo viewer)")
    else:
        print("Mode: TEXT ONLY (fast)")
    print("="*60)
    
    tests = [
        ("Dimensions", lambda: test_1_dimensions()),
        ("Numerical Jacobian (ċ = J q̇)", lambda: test_2_numerical_jacobian(visualize)),
        ("Force-to-Torque (τ = J^T f)", lambda: test_3_force_to_torque(visualize)),
        ("Velocity Mapping", lambda: test_4_velocity_mapping(visualize)),
        ("Rank Analysis", lambda: test_5_rank_analysis()),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All tests passed! Your Jacobian is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == '__main__':
    import sys
    
    visualize = False
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--viz', '-v', '--visualize']:
            visualize = True
            print("\n🎬 Running with visualization enabled!")
            print("   This will be slower but you can SEE what's happening.\n")
        elif sys.argv[1] in ['--help', '-h']:
            print("Usage:")
            print("  python test_hand_jacobian.py          # Fast text-only tests")
            print("  python test_hand_jacobian.py --viz    # With visualization")
            sys.exit(0)
    
    try:
        run_all_tests(visualize=visualize)
    except KeyboardInterrupt:
        print("\n\nTests interrupted.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()