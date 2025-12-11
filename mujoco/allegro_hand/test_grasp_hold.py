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
from controllers.impedance_controller import ImpedanceController


print("=" * 70)
print("QP GRASP MAINTENANCE TEST")
print("=" * 70)

# Load model
model = mj.MjModel.from_xml_path('mjcf/scene.xml')
data = mj.MjData(model)

# Reset to initial grasp
keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp_cube')
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
init_contacts = contact_detector.get_object_contacts(data)
print(f"✓ Initial contacts: {len(init_contacts)}")

if len(init_contacts) < 2:
    print("⚠ Need at least 2 contacts! Adjust keyframe.")
    sys.exit(1)

# Initialize grasp matrix and QP optimizer
n_contacts = len(init_contacts)
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

impedance_controller = ImpedanceController(
    n_contacts=n_contacts,
    contact_damping=0.1,
    contact_stiffness=10.0,
)
c0 = np.array([c['position'] for c in init_contacts]).flatten()
impedance_controller.set_initial_contacts(c0)

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
desired_squeeze = 1.0  # Desired internal squeeze force (N)
initial_object_height = data.xpos[object_body_id][2]

# Get target positions from keyframe for position holding
key = model.key(keyframe_id)
q_target = key.qpos[:16].copy()  # Target joint positions

def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to ZYX Euler angles: roll (x), pitch (y), yaw (z).

    q is [w, x, y, z] as in MuJoCo.
    """
    w, x, y, z = q

    # Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])

    # ZYX convention
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.array([roll, pitch, yaw])

# Logging 
desired_pos = np.empty((0, 3))
actual_pos = np.empty((0, 3))
desired_lin_vel = np.empty((0, 3))
actual_lin_vel = np.empty((0, 3))
desired_rpy = np.empty((0, 3))
actual_rpy = np.empty((0, 3))
desired_ang_vel = np.empty((0, 3))
actual_ang_vel = np.empty((0, 3))

qp_forces = np.empty((0, 12))
impedance_controller_forces = np.empty((0, 12))
nullspace_controller_forces = np.empty((0, 12))
total_forces = np.empty((0, 12))
applied_forces = np.empty((0, 12))

try:
    with viewer.launch_passive(model, data) as v:
        v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        step = 0
        
        while v.is_running():
            # 1. Get current contacts
            contacts = contact_detector.get_object_contacts(data)
            fingers_in_contact = [c['body_id'] for c in contacts]
            fingers_in_c0 = [c['body_id'] for c in init_contacts]
            fingers_lost = set(fingers_in_c0) - set(fingers_in_contact)
            # print(f"  Contacts: {fingers_in_contact}, lost: {fingers_lost}")
            # print([(c['body_id'], c['body_name']) for c in contacts])
            
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

                # FIX: Ensure all normals point from object toward fingers (outward)
                for i in range(len(contact_normals)):
                    # Vector from object center to contact point
                    to_contact = contact_positions[i] - object_position
                    
                    # If normal points toward object center, flip it
                    if np.dot(contact_normals[i], to_contact) < 0:
                        contact_normals[i] = -contact_normals[i]
                
                n_contacts = len(contacts)
                grasp_model.n_contacts = n_contacts
                qp_optimizer.n_contacts = n_contacts
                impedance_controller.n_contacts = n_contacts

                active_ids = {c['body_id'] for c in contacts}
                c0_new_list = []

                for i, c_init in enumerate(init_contacts):
                    body_id = c_init['body_id']
                    
                    if body_id in active_ids:
                        # Extract the original 3D position from c0
                        xyz = c0[3*i : 3*i+3]
                        c0_new_list.append(xyz)

                if c0_new_list:
                    c0_new = np.concatenate(c0_new_list)
                else:
                    c0_new = np.zeros(0)   # no active contacts

                impedance_controller.set_initial_contacts(c0_new)
                
                contact_pos_flat = contact_positions.flatten()
                G = grasp_model.compute_grasp_matrix(contact_pos_flat, object_position)
                
                # 4. Compute desired wrench (just gravity compensation)
                target_pos = np.array([-0.02, 0.00, 0.11])  # Stay at current position (static holding)
                target_vel = np.zeros(3)  # Zero velocity
                target_angular_vel = np.zeros(3)
                target_orientation = np.array([1, 0, 0, 0])

                desired_acc = qp_optimizer.compute_desired_acceleration(
                    object_state, 
                    target_pos, 
                    target_vel, 
                    target_orientation=target_orientation, 
                    target_angular_vel=target_angular_vel,
                    Kp_pos=20.0,  # Increased for stronger position holding
                    Kd_pos=10.0,   # Increased for better damping
                    Kp_rot=5.0, 
                    Kd_rot=2.0,
                )
                w_desired = qp_optimizer.compute_dynamic_wrench(object_state, desired_acc, gravity=gravity)
                
                # 5. Desired internal forces (squeeze to maintain grasp)
                f_d_normals = np.ones(n_contacts) * desired_squeeze
                
                # 6. Solve QP
                f_int, info = qp_optimizer.compute_contact_forces(
                    desired_wrench=w_desired,
                    grasp_matrix=G,
                    desired_normal_forces=f_d_normals,
                    contact_normals=contact_normals
                )

                # 6b. Compute impedance controller forces
                contact_vels = grasp_model.object_twist_to_contact_velocities(
                    np.concatenate([object_state['linear_velocity'], object_state['angular_velocity']]),
                )
                eul = quat_to_euler_zyx(object_state['orientation'])
                f_impedance = impedance_controller.compute_contact_forces(
                    np.concatenate([object_state['position'], quat_to_euler_zyx(object_state['orientation'])]), 
                    np.concatenate([target_pos, quat_to_euler_zyx(target_orientation)]),
                    contact_pos_flat,
                    contact_vels,
                    G,
                    impedance_controller.compute_W_matrix(eul),
                )
                
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

                f_c = f_impedance + f_int

                # τ = J^T f
                tau_c = J.T @ f_c

                # 8. Compute nullspace torque to maintain finger configuration
                tau_total, info = nullspace_controller.compute_full_pipeline(
                    tau_contact=tau_c,
                    joint_positions=data.qpos[:16],
                    joint_velocities=data.qvel[:16],
                    jacobian=J
                )

                # 9. Combine: τ_des = τ_contact + τ_null (Equation 30)
                tau_total = tau_c     

                # Apply torques (with safety limits)
                data.ctrl[:16] = np.clip(tau_total, -0.5, 0.5)

                # Log data
                JT_pinv = np.linalg.pinv(J.T)
                f_ns = JT_pinv @ info['tau_null_projected']
                f_tot = JT_pinv @ tau_total
                f_app = JT_pinv @ data.ctrl[:16]

                desired_pos = np.vstack((desired_pos, target_pos))
                desired_lin_vel = np.vstack((desired_lin_vel, target_vel))
                desired_rpy = np.vstack((desired_rpy, quat_to_euler_zyx(target_orientation)))
                desired_ang_vel = np.vstack((desired_ang_vel, target_angular_vel))
                actual_pos = np.vstack((actual_pos, object_state['position']))
                actual_lin_vel = np.vstack((actual_lin_vel, object_state['linear_velocity']))
                actual_rpy = np.vstack((actual_rpy, quat_to_euler_zyx(object_state['orientation'])))
                actual_ang_vel = np.vstack((actual_ang_vel, object_state['angular_velocity']))

                f_impedance_full = np.zeros(12)
                f_int_full = np.zeros(12)
                f_ns_full = np.zeros(12)
                f_tot_full = np.zeros(12)
                f_app_full = np.zeros(12)
                j = 0
                finger_ids = [16, 11, 6, 21] # [ff_tip, mf_tip, rf_tip, th_tip]
                for i in range(4):
                    if finger_ids[i] in fingers_lost:
                        f_impedance_full[3*i:3*(i+1)] = np.zeros(3)
                        f_int_full[3*i:3*(i+1)] = np.zeros(3)
                        f_ns_full[3*i:3*(i+1)] = np.zeros(3)
                        f_tot_full[3*i:3*(i+1)] = np.zeros(3)
                        f_app_full[3*i:3*(i+1)] = np.zeros(3)
                    else:
                        f_impedance_full[3*i:3*(i+1)] = f_impedance[3*j:3*(j+1)]
                        f_int_full[3*i:3*(i+1)] = f_int[3*j:3*(j+1)]
                        f_ns_full[3*i:3*(i+1)] = f_ns[3*j:3*(j+1)]
                        f_tot_full[3*i:3*(i+1)] = f_tot[3*j:3*(j+1)]
                        f_app_full[3*i:3*(i+1)] = f_app[3*j:3*(j+1)]
                        j += 1

                applied_forces = np.vstack((applied_forces, f_app_full))
                qp_forces = np.vstack((qp_forces, f_int_full))
                impedance_controller_forces = np.vstack((impedance_controller_forces, f_impedance_full))
                nullspace_controller_forces = np.vstack((nullspace_controller_forces, f_ns_full))
                total_forces = np.vstack((total_forces, f_tot_full))
                
                # Print status
                if step % 10 == 0:
                    obj_height = object_position[2]
                    height_error = obj_height - initial_object_height
                    obj_vel = np.linalg.norm(data.qvel[16:19])
                    
                    # Verify wrench
                    w_actual = G @ f_c
                    wrench_error = np.linalg.norm(w_actual - w_desired)
                    
                    # print(f"\n{'='*60}")
                    # print(f"QP Optimal Forces (step {step}):")
                    # print(f"  Status: {info['qp_status']}, Optimal: {info['qp_optimal']}")
                    # print(f"\nContact forces breakdown:")
                    for i in range(n_contacts):
                        f_i = f_c[3*i:3*(i+1)]
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

                    # print(f"\nJacobian Debug:")
                    # print(f"  J shape: {J.shape}")
                    # print(f"  J max magnitude: {np.abs(J).max():.6f}")
                    print(f"  f_contact magnitude: {np.linalg.norm(f_c.reshape(-1, 3), axis=1)}")

                    print(f"\n{'='*60}")
                    print(f"Step {step}:")
                    print(f"  Contacts: {len(contacts)}")
                    # print(f"  QP status: {info['qp_status']}, optimal: {info['qp_optimal']}")
                    print(f"\nObject State:")
                    print(f"  Height: {obj_height:.4f} m (error: {height_error*1000:.2f} mm)")
                    print(f"  Velocity: {obj_vel:.4f} m/s")
                    print(f"\nWrench:")
                    print(f"  Desired: {w_desired}")
                    print(f"  Error: {wrench_error:.4f}")
                    print(f"  QP torque range: [{tau_total.min():.2f}, {tau_total.max():.2f}] Nm")
                    
                    if abs(obj_vel) < 0.001 and abs(height_error) < 0.002:
                        print(f"  ✓ Object stable!")

            mj.mj_step(model, data)
            v.sync()
            step += 1
except KeyboardInterrupt:
    print("\n\n✋ Control interrupted by user. Saving data...")
    print(desired_pos.shape)
    print(actual_pos.shape)
    np.savez(
        'logs/grasp_hold_log.npz',
        desired_pos=desired_pos,
        actual_pos=actual_pos,
        desired_lin_vel=desired_lin_vel,
        actual_lin_vel=actual_lin_vel,
        desired_rpy=desired_rpy,
        actual_rpy=actual_rpy,
        desired_ang_vel=desired_ang_vel,
        actual_ang_vel=actual_ang_vel,
        qp_forces=qp_forces,
        impedance_controller_forces=impedance_controller_forces,
        nullspace_controller_forces=nullspace_controller_forces,
        total_forces=total_forces,
        applied_forces=applied_forces
    )
    print("✓ Data saved to 'grasp_hold_log.npz'.")

print("\n✓ Visualization complete")