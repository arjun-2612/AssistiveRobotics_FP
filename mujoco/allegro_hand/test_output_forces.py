#!/usr/bin/env python3
"""
Complete Pipeline Visualization (Without QP)
Demonstrates all working components:
1. Object spawned between fingers
2. Contact detection and visualization
3. Grasp matrix computation
4. Impedance controller forces
5. Force visualization as arrows
6. Real-time monitoring
"""

import numpy as np
import sys
sys.path.append('controllers')

import mujoco as mj
import mujoco.viewer as viewer

from controllers.impedance_controller import ImpedanceController
from controllers.grasp_model import GraspMatrix
from controllers.object_state import ObjectPoseEstimator
from controllers.contact_detector import ContactDetector
from controllers.hand_jacobian import HandJacobian

print("=" * 70)
print("COMPLETE PIPELINE VISUALIZATION")
print("=" * 70)

# ============================================================================
# Load Model and Initialize
# ============================================================================
print("\n[1] Loading MuJoCo Model")
print("-" * 70)

model_path = 'mjcf/scene.xml'
model = mj.MjModel.from_xml_path(model_path)
data = mj.MjData(model)

print(f"✓ Model loaded: {model_path}")
print(f"  - Bodies: {model.nbody}")
print(f"  - Joints: {model.njnt}")
print(f"  - DOFs: {model.nv}")

# Reset to initial grasp keyframe
keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
if keyframe_id >= 0:
    mj.mj_resetDataKeyframe(model, data, keyframe_id)
    print(f"✓ Reset to 'initial_grasp' keyframe")
else:
    print("⚠ No 'initial_grasp' keyframe found")

mj.mj_forward(model, data)

# ============================================================================
# Initialize Components
# ============================================================================
print("\n[2] Initializing Components")
print("-" * 70)

pose_estimator = ObjectPoseEstimator(model, 'object')
contact_detector = ContactDetector(model, 'object')
hand_jacobian = HandJacobian(model, data)

print("✓ Object pose estimator initialized")
print("✓ Contact detector initialized")
print("✓ Hand Jacobian computer initialized")

# ============================================================================
# Analyze Initial State
# ============================================================================
print("\n[3] Initial State Analysis")
print("-" * 70)

# Get object pose
x_obj = pose_estimator.get_6d_pose_vector(data)
print(f"Object pose: {x_obj}")
print(f"  Position: {x_obj[:3]}")
print(f"  Orientation (Euler): {x_obj[3:]}")

# Detect contacts
contacts = contact_detector.get_object_contacts(data)
n_contacts = len(contacts)
print(f"\n✓ Detected {n_contacts} contacts")

if n_contacts > 0:
    for i, contact in enumerate(contacts):
        print(f"\nContact {i+1}:")
        print(f"  Finger: {contact['finger']}")
        print(f"  Position: {contact['position']}")
        print(f"  Normal: {contact['normal']}")
        print(f"  Distance: {contact['distance']:.4f} m")
else:
    print("⚠ No contacts detected! Check keyframe configuration.")
    sys.exit(1)

# Compute grasp matrix
contact_pos = np.array([c['position'] for c in contacts]).flatten()
contact_normals = np.array([c['normal'] for c in contacts])

grasp_matrix = GraspMatrix(n_contacts=n_contacts)
G = grasp_matrix.compute_grasp_matrix(contact_pos, x_obj[:3])
print(f"\n✓ Grasp matrix computed: {G.shape}")
print(f"  Rank: {np.linalg.matrix_rank(G)}")
print(f"  Force closure: {grasp_matrix.is_force_closure()}")

# Initialize impedance controller
impedance_controller = ImpedanceController(
    contact_stiffness=50.0,  # Lower for visualization
    contact_damping=5.0,
    n_contacts=n_contacts
)
impedance_controller.set_initial_contacts(contact_pos)
print(f"\n✓ Impedance controller initialized")

# ============================================================================
# Define Desired Pose
# ============================================================================
print("\n[4] Setting Desired Pose")
print("-" * 70)

x_desired = x_obj.copy()

# Choose motion type:
motion_type = input("\nChoose motion type:\n"
                   "  1. Lift object (2cm up)\n"
                   "  2. Rotate object (30° around z)\n"
                   "  3. Move sideways (1cm in x)\n"
                   "  4. Stay static (no motion)\n"
                   "Enter choice (1-4): ")

if motion_type == '1':
    x_desired[2] += 0.02
    print("→ Target: Lift 2cm")
elif motion_type == '2':
    x_desired[5] += np.pi/6
    print("→ Target: Rotate 30°")
elif motion_type == '3':
    x_desired[0] += 0.01
    print("→ Target: Move 1cm in x")
else:
    print("→ Target: Stay static")

print(f"Initial pose: {x_obj}")
print(f"Desired pose: {x_desired}")

# ============================================================================
# Visualization Loop
# ============================================================================
print("\n[5] Starting Visualization")
print("-" * 70)
print("Viewer controls:")
print("  - Mouse: Rotate/pan camera")
print("  - Scroll: Zoom")
print("  - ESC: Exit")
print("\nPress ENTER to launch viewer...")
input()

with viewer.launch_passive(model, data) as v:
    # Enable visualizations
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    v.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False
    
    # Adjust visualization scale
    model.vis.scale.contactwidth = 0.1
    model.vis.scale.contactheight = 0.1
    model.vis.scale.forcewidth = 0.05
    
    step_count = 0
    
    print("\nViewer launched. Monitoring state...\n")
    
    while v.is_running():
        # Get current state
        x_current = pose_estimator.get_6d_pose_vector(data)
        contacts = contact_detector.get_object_contacts(data)
        n_current_contacts = len(contacts)
        
        # Update contact positions
        if n_current_contacts > 0 and n_current_contacts == n_contacts:
            contact_pos = np.array([c['position'] for c in contacts]).flatten()
            contact_normals = np.array([c['normal'] for c in contacts])
            
            # Compute grasp matrix
            G = grasp_matrix.compute_grasp_matrix(contact_pos, x_current[:3])
            
            # Compute W matrix
            W = ImpedanceController.compute_W_matrix(x_current[3:])
            
            # Compute impedance forces
            contact_vel = np.zeros(len(contact_pos))
            f_impedance = impedance_controller.compute_contact_forces(
                x_current=x_current,
                x_desired=x_desired,
                contact_positions=contact_pos,
                contact_velocities=contact_vel,
                grasp_matrix=G,
                W_matrix=W
            )
            
            # Apply forces to object (visualization)
            object_id = pose_estimator.object_id
            data.xfrc_applied[object_id, :] = 0
            
            # Sum forces and torques
            total_force = np.zeros(3)
            total_torque = np.zeros(3)
            
            for i in range(n_current_contacts):
                force_3d = f_impedance[i*3:(i+1)*3]
                contact_position = contact_pos[i*3:(i+1)*3]
                
                total_force += force_3d
                r = contact_position - x_current[:3]
                total_torque += np.cross(r, force_3d)
            
            data.xfrc_applied[object_id, :3] = total_force
            data.xfrc_applied[object_id, 3:] = total_torque
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                pose_error = x_desired - x_current
                error_norm = np.linalg.norm(pose_error)
                
                print(f"Step {step_count}:")
                print(f"  Contacts: {n_current_contacts}")
                print(f"  Pose error: {error_norm:.4f}")
                print(f"  Total force: {total_force} N")
                print(f"  Total torque: {total_torque} Nm")
                
                # Show per-contact forces
                print("  Contact forces:")
                for i in range(n_current_contacts):
                    f_i = f_impedance[i*3:(i+1)*3]
                    f_mag = np.linalg.norm(f_i)
                    print(f"    Contact {i+1}: {f_i} (mag: {f_mag:.3f} N)")
                
                # Compute object wrench
                w_obj = G @ f_impedance
                print(f"  Object wrench: {w_obj}")
                print()
        
        # Step simulation
        mj.mj_step(model, data)
        v.sync()
        step_count += 1

print("\n✓ Visualization complete")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATION SUMMARY")
print("=" * 70)
print("✓ Object spawned between fingers")
print("✓ Contacts detected and visualized")
print("✓ Grasp matrix computed")
print("✓ Impedance forces computed and applied")
print("✓ Forces visualized as arrows in viewer")
print("\nComponents demonstrated:")
print("  - Object pose estimation")
print("  - Contact detection")
print("  - Grasp matrix (Section II.A)")
print("  - Impedance controller (Section III.A)")
print("  - Force visualization")
print("\nNote: QP optimizer (Section III.B) not included in this demo")
print("=" * 70)