#!/usr/bin/env python3
"""
QP Grasp Maintenance Test
Visualize how QP optimizer maintains grasp without letting cube fall.
"""

import numpy as np
import sys
sys.path.append('controllers')

import mujoco as mj
import mujoco.viewer as viewer

from controllers.contact_detector import ContactDetector
from controllers.hand_jacobian import HandJacobian
from controllers.grasp_model import GraspMatrix
from controllers.QP import InternalForceOptimizer
from controllers.object_state import ObjectPoseEstimator
from controllers.nullspace_controller import NullspaceController


print("=" * 70)
print("QP GRASP MAINTENANCE TEST")
print("=" * 70)

# Load model
model = mj.MjModel.from_xml_path('mjcf/scene.xml')
data = mj.MjData(model)

# Reset to initial grasp
keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
mj.mj_resetDataKeyframe(model, data, keyframe_id)
mj.mj_forward(model, data)

# Initialize components
contact_detector = ContactDetector(model, 'object')
hand_jacobian = HandJacobian(model, data)
object_estimator = ObjectPoseEstimator(model, 'object')

# Get object info
object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'object')
object_mass = model.body_mass[object_body_id]
gravity = model.opt.gravity[2]
weight = object_mass * abs(gravity)

print(f"✓ Components initialized")
print(f"✓ Object mass: {object_mass:.3f} kg")
print(f"✓ Weight: {weight:.3f} N")

# Get initial contacts
contacts = contact_detector.get_object_contacts(data)
print(f"✓ Initial contacts: {len(contacts)}")

if len(contacts) < 2:
    print("⚠ Need at least 2 contacts! Adjust keyframe.")
    sys.exit(1)

# Initialize grasp matrix and QP optimizer
n_contacts = len(contacts)
grasp_model = GraspMatrix(n_contacts=n_contacts)
qp_optimizer = InternalForceOptimizer(
    n_contacts=n_contacts,
    friction_coefficient=0.8,
    f_min=0.1,
    f_max=2.0
)
# Initialize nullspace controller
nullspace_controller = NullspaceController(
    n_joints_per_finger=4,
    n_fingers=4,  # Allegro hand has 4 fingers
    K_null=5.0,   # Stiffness 
    D_null=0.5    # Damping
)

print(f"✓ QP optimizer initialized with {n_contacts} contacts\n")
print(f"✓ Nullspace controller initialized\n")

print("Control strategy:")
print("  1. Estimate object state (position, orientation)")
print("  2. Compute desired wrench (gravity compensation)")
print("  3. Build grasp matrix from contacts")
print("  4. Solve QP for optimal contact forces")
print("  5. Map forces to joint torques via Jacobian")
print("  6. Add nullspace torques to maintain finger configuration")
print("\nPress ENTER to start...")
input()

# Control parameters
desired_squeeze = 0.25  # Desired internal squeeze force (N)
initial_object_height = data.xpos[object_body_id][2]

# Get target positions from keyframe for position holding
key = model.key(keyframe_id)
q_target = key.qpos[:16].copy()  # Target joint positions

with viewer.launch_passive(model, data) as v:
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    step = 0
    
    while v.is_running():
        # 1. Get current contacts
        contacts = contact_detector.get_object_contacts(data)
        
        if len(contacts) >= 2:
            # 2. Estimate object state
            object_state = object_estimator.get_object_state(data)
            object_position = object_state['position']
            
            # Add additional object properties for dynamic wrench computation
            object_state['mass'] = object_mass
            object_state['inertia'] = np.diag(model.body_inertia[object_body_id])
            object_state['velocity'] = data.qvel[16:19]
            object_state['angular_velocity'] = data.qvel[19:22]

            # 3. Build grasp matrix
            contact_positions = np.array([c['position'] for c in contacts])
            contact_normals = np.array([c['normal'] for c in contacts])
            
            n_contacts = len(contacts)
            grasp_model.n_contacts = n_contacts
            qp_optimizer.n_contacts = n_contacts
            
            contact_pos_flat = contact_positions.flatten()
            G = grasp_model.compute_grasp_matrix(contact_pos_flat, object_position)
            
            # 4. Compute desired wrench (just gravity compensation)
            target_pos = np.array([-0.02, 0.00, 0.22])  # Stay at current position (static holding)
            target_vel = np.zeros(3)  # Zero velocity
            target_angular_vel = np.zeros(3)
            target_orientation = np.array([1, 0, 0, 0])
            desired_acc = qp_optimizer.compute_desired_acceleration(object_state, target_pos, target_vel, target_orientation=target_orientation, target_angular_vel=target_angular_vel)
            w_desired = qp_optimizer.compute_dynamic_wrench(object_state, desired_acc, gravity=gravity)
            # w_desired = np.array([0, 0, weight, 0, 0, 0])

            
            # 5. Desired internal forces (squeeze to maintain grasp)
            f_d_normals = np.ones(n_contacts) * desired_squeeze
            
            # 6. Solve QP
            f_computed, info = qp_optimizer.compute_contact_forces(
                desired_wrench=w_desired,
                grasp_matrix=G,
                desired_normal_forces=f_d_normals,
                contact_normals=contact_normals
            )
            # Print optimal forces
            if step % 10 == 0:
                print(f"\n{'='*60}")
                print(f"QP Optimal Forces (step {step}):")
                print(f"  Status: {info['qp_status']}, Optimal: {info['qp_optimal']}")
                print(f"\nContact forces breakdown:")
                for i in range(n_contacts):
                    f_i = f_computed[3*i:3*(i+1)]
                    n_i = contact_normals[i]
                    
                    # Normal component
                    f_n = np.dot(f_i, n_i)
                    # Tangential component
                    f_tangent = f_i - f_n * n_i
                    f_t_mag = np.linalg.norm(f_tangent)
                    
                    print(f"  Contact {i+1}:")
                    print(f"    Position: {contact_positions[i]}")
                    print(f"    Force vector: [{f_i[0]:6.3f}, {f_i[1]:6.3f}, {f_i[2]:6.3f}] N")
                    print(f"    Normal force: {f_n:.3f} N (desired: {f_d_normals[i]:.3f} N)")
                    print(f"    Tangential: {f_t_mag:.3f} N (friction ratio: {f_t_mag/(f_n+1e-10):.3f})")
            
            # 7. Compute joint torques via Jacobian
            J_list = []
            for contact in contacts:
                body_id = contact['body_id']
                contact_pos = contact['position']
                
                jacp = np.zeros((3, model.nv))  # Position (translation) Jacobian
                jacr = np.zeros((3, model.nv))  # Rotation Jacobian

                # Compute Jacobian at contact point
                mj.mj_jac(model, data, jacp, jacr, contact_pos, body_id)     
                J_list.append(jacp)       

            # Stack all contact Jacobians
            J = np.vstack(J_list)
            J = J[:, :16]  # Only hand joints
            

            # DEBUG: Print Jacobian info
            if step % 10 == 0:
                print(f"\nJacobian Debug:")
                print(f"  J shape: {J.shape}")
                print(f"  J max magnitude: {np.abs(J).max():.6f}")
                print(f"  f_contact magnitude: {np.linalg.norm(f_computed):.3f}")
            
            # τ = J^T f
            tau_qp = J.T @ -f_computed

            # 8. Compute nullspace torque to maintain finger configuration
            tau_null_raw = nullspace_controller.compute_nullspace_torque(
                joint_positions=data.qpos[:16],
                joint_velocities=data.qvel[:16],
                jacobian=J
            )
            
            # # Project into nullspace (so it doesn't affect object forces)
            J_T = J.T
            J_T_pinv = np.linalg.pinv(J_T)
            # print(f"  J shape: {J.shape}")
            # print(f"  J_T shape: {J_T.shape}")
            # print(f"  J_T_pinv shape: {J_T_pinv.shape}")
            N = np.eye(16) - J_T @ J_T_pinv  # Nullspace projector
            tau_null = N @ tau_null_raw
            # print(f"\nDebug shapes:")
            # print(f"  J_T @ J_T_pinv shape: {(J_T @ J_T_pinv).shape}")
            # print(f"  tau_null_raw shape: {tau_null_raw.shape}")
            
            # 9. Combine: τ_des = τ_contact + τ_null (Equation 30)
            tau_total = tau_qp           

            # Apply torques (with safety limits)
            data.ctrl[:16] = np.clip(tau_total, -5, 5)
            
            # Print status
            if step % 10 == 0:
                obj_height = object_position[2]
                height_error = obj_height - initial_object_height
                obj_vel = np.linalg.norm(data.qvel[16:19])
                
                # Verify wrench
                w_actual = G.T @ f_computed
                wrench_error = np.linalg.norm(w_actual - w_desired)
                
                print(f"\n{'='*60}")
                print(f"Step {step}:")
                print(f"  Contacts: {len(contacts)}")
                print(f"  QP status: {info['qp_status']}, optimal: {info['qp_optimal']}")
                print(f"\nObject State:")
                print(f"  Height: {obj_height:.4f} m (error: {height_error*1000:.2f} mm)")
                print(f"  Velocity: {obj_vel:.4f} m/s")
                print(f"\nWrench:")
                print(f"  Desired: {w_desired}")
                print(f"  Actual:  {w_actual}")
                print(f"  Error: {wrench_error:.4f}")
                print(f"  QP torque range: [{tau_qp.min():.2f}, {tau_qp.max():.2f}] Nm")
                
                if abs(obj_vel) < 0.001 and abs(height_error) < 0.002:
                    print(f"  ✓ Object stable!")
        
        mj.mj_step(model, data)
        v.sync()
        step += 1

print("\n✓ Visualization complete")