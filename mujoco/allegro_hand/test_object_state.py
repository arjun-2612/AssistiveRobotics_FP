#!/usr/bin/env python3
"""
Test the ObjectPoseEstimator.
Verify we can correctly read object pose and velocities.
"""

import time
import numpy as np
import mujoco as mj
import sys
sys.path.append('controllers')

from controllers import ObjectPoseEstimator


def reset_to_initial_grasp(model: mj.MjModel, data: mj.MjData):
    """Reset using keyframe."""
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
    if key_id >= 0:
        mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)


def test_object_state_estimator():
    """Test object state estimation."""
    
    # Load model
    model = mj.MjModel.from_xml_path('mjcf/scene.xml')
    data = mj.MjData(model)
    
    # Initialize estimator
    try:
        estimator = ObjectPoseEstimator(model, 'object')
        print("✓ ObjectPoseEstimator initialized successfully")
    except ValueError as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Reset to grasp pose
    reset_to_initial_grasp(model, data)
    
    print("\n" + "="*60)
    print("TESTING OBJECT STATE ESTIMATION")
    print("="*60)
    
    # Test 1: Get initial state
    print("\n--- Test 1: Initial State ---")
    state = estimator.get_object_state(data)
    
    print(f"Position: {state['position']}")
    print(f"Orientation (quat): {state['orientation']}")
    print(f"Linear velocity: {state['linear_velocity']}")
    print(f"Angular velocity: {state['angular_velocity']}")
    print(f"Rotation matrix shape: {state['rotation_matrix'].shape}")
    
    # Verify rotation matrix is valid (orthonormal)
    R = state['rotation_matrix']
    identity_check = R @ R.T
    is_orthonormal = np.allclose(identity_check, np.eye(3), atol=1e-6)
    print(f"Rotation matrix orthonormal: {is_orthonormal}")
    
    if is_orthonormal:
        print("✓ Rotation matrix is valid")
    else:
        print("✗ Rotation matrix is invalid")
    
    # Test 2: Get 6D pose vector
    print("\n--- Test 2: 6D Pose Vector ---")
    pose_6d = estimator.get_6d_pose_vector(data)
    print(f"6D pose (pos + euler): {pose_6d}")
    print(f"  Position (xyz): {pose_6d[:3]}")
    print(f"  Orientation (rpy): {pose_6d[3:]} rad")
    print(f"  Orientation (rpy): {np.rad2deg(pose_6d[3:])} deg")
    
    # Test 3: Run simulation and track motion
    print("\n--- Test 3: Tracking During Simulation ---")
    print("Opening interactive viewer...")
    print("You can interact with the simulation using mouse/keyboard")
    print("\nSimulating for 6 seconds, printing state every 0.2s...")
    
    # Use interactive viewer (blocking call)
    def controller(model, data):
        """Simulation controller callback."""
        # This runs every step
        pass
    
    # Launch interactive viewer with callback
    print("\nStarting simulation...")
    reset_to_initial_grasp(model, data)
    
    sim_time = 0
    dt = model.opt.timestep
    print_interval = 0.2
    next_print = print_interval
    
    # Manual simulation loop with viewer
    import mujoco.viewer as mjv
    
    viewer = mjv.launch_passive(model, data)
    
    try:
        while viewer.is_running():
            step_start = time.time()
            
            # Simulation step
            mj.mj_step(model, data)
            sim_time += dt
            
            # Print state periodically
            if sim_time >= next_print and sim_time < 6.0:
                state = estimator.get_object_state(data)

                # Count object contacts
                num_contacts = 0
                total_force = 0
                for i in range(data.ncon):
                    contact = data.contact[i]
                    body1 = model.body(model.geom_bodyid[contact.geom1]).name
                    body2 = model.body(model.geom_bodyid[contact.geom2]).name
                    if 'object' in (body1 + body2):
                        num_contacts += 1
                        total_force += abs(contact.dist) * 1000
                
                print(f"\nTime: {sim_time:.2f}s")
                print(f"  Position: {state['position']}")
                print(f"  Height (z): {state['position'][2]:.4f}")
                print(f"  Linear vel: {state['linear_velocity']}")
                print(f"  Speed: {np.linalg.norm(state['linear_velocity']):.4f} m/s")
                print(f"  Contacts with object: {num_contacts}")
                print(f"  Estimated contact force: {total_force:.2f} N")
                next_print += print_interval
            
            # Apply perturbation after 6 seconds
            elif 6.0 <= sim_time < 6.1 and sim_time - dt < 6.0:
                print("\n--- Test 4: Applying Perturbation ---")
                print("Applying small upward force on object...")
            
            if 6.0 <= sim_time < 6.5:
                # Apply upward force
                data.xfrc_applied[estimator.object_id][2] = 0.5  # 0.5N upward
            
            elif sim_time >= 6.5 and sim_time - dt < 6.5:
                state_after = estimator.get_object_state(data)
                print(f"\nAfter perturbation:")
                print(f"  Position: {state_after['position']}")
                print(f"  Linear vel: {state_after['linear_velocity']}")
                print(f"  Speed: {np.linalg.norm(state_after['linear_velocity']):.4f} m/s")
                print("\nLetting physics settle...")
            
            elif sim_time >= 8.0 and sim_time - dt < 8.0:
                state_final = estimator.get_object_state(data)
                print(f"\nFinal state:")
                print(f"  Position: {state_final['position']}")
                print(f"  Height: {state_final['position'][2]:.4f}")
                
                print("\n" + "="*60)
                print("TEST COMPLETE")
                print("="*60)
                print("\n🎮 Viewer controls:")
                print("  - Right-click + drag: Rotate view")
                print("  - Scroll: Zoom")
                print("  - Double-click body: Select")
                print("  - Ctrl + Right-click: Apply force")
                print("  - Backspace: Reset to keyframe")
                print("  - Close window to exit")
                print("="*60)
            
            # Update viewer
            viewer.sync()
            
            # Time regulation (run at roughly real-time)
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        viewer.close()
        print("Viewer closed")


if __name__ == '__main__':
    test_object_state_estimator()