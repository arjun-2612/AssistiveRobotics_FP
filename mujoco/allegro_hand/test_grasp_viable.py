#!/usr/bin/env python3
"""
QP Grasp Maintenance Test - Simplified version
Test if position control alone can hold the object first.
"""

import numpy as np
import sys
sys.path.append('controllers')

import mujoco as mj
import mujoco.viewer as viewer

from controllers.contact_detector import ContactDetector
from controllers.object_state import ObjectPoseEstimator

print("=" * 70)
print("SIMPLE POSITION CONTROL TEST")
print("=" * 70)

# Load model
model = mj.MjModel.from_xml_path('mjcf/scene.xml')
data = mj.MjData(model)

# Reset to initial grasp
keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
mj.mj_resetDataKeyframe(model, data, keyframe_id)
mj.mj_forward(model, data)
key = model.key(keyframe_id)

# Initialize components
contact_detector = ContactDetector(model, 'object')
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

print("\nTest: Can position control alone hold the object?")
print("Press ENTER to start...")
input()

# Control parameters
initial_object_height = data.xpos[object_body_id][2]
q_target = key.ctrl[:16].copy()  # Save initial joint configuration

with viewer.launch_passive(model, data) as v:
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    step = 0
    
    while v.is_running():
        contacts = contact_detector.get_object_contacts(data)
        
        if len(contacts) >= 2:
            data.ctrl[:16] = q_target
            
            # Print status
            if step % 200 == 0:
                obj_height = data.xpos[object_body_id][2]
                height_error = obj_height - initial_object_height
                obj_vel = np.linalg.norm(data.qvel[16:19])
                
                print(f"\nStep {step}:")
                print(f"  Contacts: {len(contacts)}")
                print(f"  Object height: {obj_height:.4f} m (error: {height_error*1000:.2f} mm)")
                print(f"  Object velocity: {obj_vel:.4f} m/s")
                print(f"  Target positions: {q_target[:4]}...")
                print(f"  Actual positions: {data.qpos[:4]}...")
                
                if height_error < -0.01:
                    print(f"  ⚠ Object falling! Height loss: {-height_error*100:.1f} cm")
                elif abs(obj_vel) < 0.001 and abs(height_error) < 0.002:
                    print(f"  ✓ Object stable!")
        else:
            data.ctrl[:16] = 0
            if step % 200 == 0:
                print(f"\n⚠ Lost contact! Only {len(contacts)} contacts")
        
        # Step simulation
        mj.mj_step(model, data)
        v.sync()
        step += 1

print("\n✓ Test complete")
print("\nResult: If object stayed stable, keyframe is good and QP can work.")
print("If object fell, keyframe needs adjustment first.")