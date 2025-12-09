#!/usr/bin/env python3
"""
Full Pipeline Test with Joint Control
Demonstrates fingers actively holding the cube using joint torques.
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
print("FULL PIPELINE: ACTIVE GRASP WITH JOINT CONTROL")
print("=" * 70)

# Load model
model = mj.MjModel.from_xml_path('mjcf/scene.xml')
data = mj.MjData(model)

# Reset to initial grasp
keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
mj.mj_resetDataKeyframe(model, data, keyframe_id)
mj.mj_forward(model, data)

# Initialize components
pose_estimator = ObjectPoseEstimator(model, 'object')
contact_detector = ContactDetector(model, 'object')
hand_jacobian = HandJacobian(model, data)

print("✓ Components initialized\n")

# Get initial state
x_obj = pose_estimator.get_6d_pose_vector(data)
contacts = contact_detector.get_object_contacts(data)
n_contacts = len(contacts)

print(f"Initial state:")
print(f"  Object position: {x_obj[:3]}")
print(f"  Contacts: {n_contacts}")

if n_contacts == 0:
    print("⚠ No contacts! Adjust keyframe.")
    sys.exit(1)

# Initialize controllers
contact_pos = np.array([c['position'] for c in contacts]).flatten()
grasp_matrix = GraspMatrix(n_contacts=n_contacts)
G = grasp_matrix.compute_grasp_matrix(contact_pos, x_obj[:3])

impedance_controller = ImpedanceController(
    contact_stiffness=100.0,  # Higher stiffness for firm grip
    contact_damping=10.0,
    n_contacts=n_contacts
)
impedance_controller.set_initial_contacts(contact_pos)

# Desired pose (hold in place)
x_desired = x_obj.copy()
x_desired[2] += 0.05  # Try to lift 5cm

print(f"\nTarget: Lift to z = {x_desired[2]:.3f} m")
print("\nPress ENTER to start...")
input()

with viewer.launch_passive(model, data) as v:
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    step = 0
    
    while v.is_running():
        # Get current state
        x_current = pose_estimator.get_6d_pose_vector(data)
        contacts = contact_detector.get_object_contacts(data)
        
        if len(contacts) == n_contacts:
            contact_pos = np.array([c['position'] for c in contacts]).flatten()
            
            # Compute grasp matrix
            G = grasp_matrix.compute_grasp_matrix(contact_pos, x_current[:3])
            W = ImpedanceController.compute_W_matrix(x_current[3:])
            
            # Compute desired contact forces (impedance control)
            contact_vel = np.zeros(len(contact_pos))
            f_impedance = impedance_controller.compute_contact_forces(
                x_current=x_current,
                x_desired=x_desired,
                contact_positions=contact_pos,
                contact_velocities=contact_vel,
                grasp_matrix=G,
                W_matrix=W
            )
            
            # Build contact body list and offsets for Jacobian
            contact_body_ids = []
            contact_offsets = []
            for contact in contacts:
                contact_body_ids.append(contact['body_id'])
                body_pos = data.xpos[contact['body_id']]
                offset = contact['position'] - body_pos
                contact_offsets.append(offset)
            
            # Compute Jacobian
            J = hand_jacobian.compute_contact_jacobian(contact_body_ids, contact_offsets)
            
            # Convert forces to joint torques: τ = J^T @ f
            tau = J.T @ f_impedance
            
            # Apply to hand joints (16 DOFs)
            data.ctrl[:16] = tau[:16] * 0.5  # Scale down for stability
            
            # Print status
            if step % 100 == 0:
                pose_error = x_desired - x_current
                print(f"\nStep {step}:")
                print(f"  Object z: {x_current[2]:.4f} m (target: {x_desired[2]:.4f})")
                print(f"  Contacts: {len(contacts)}")
                print(f"  Pose error: {np.linalg.norm(pose_error):.4f}")
                print(f"  Max joint torque: {np.abs(tau[:16]).max():.2f} Nm")
        
        # Step simulation
        mj.mj_step(model, data)
        v.sync()
        step += 1

print("\n✓ Test complete")