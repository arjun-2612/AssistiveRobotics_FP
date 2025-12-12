#!/usr/bin/env python3
"""
Test contact detection with the 2-finger Allegro hand.
Starts directly from the 'initial_grasp' keyframe so the object is already
pinched in the air.
"""
import time
import argparse
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv


def print_contact_info(model: mj.MjModel, data: mj.MjData):
    """Print detailed information about all active contacts."""
    print(f"\n{'='*60}")
    print(f"Time: {data.time:.3f}s | Active contacts: {data.ncon}")
    print(f"{'='*60}")
    if data.ncon == 0:
        print("No contacts detected")
        return

    # Group contacts by body pairs
    contact_groups = {}
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = model.geom(contact.geom1).name or f"geom_{contact.geom1}"
        geom2_name = model.geom(contact.geom2).name or f"geom_{contact.geom2}"
        body1_name = model.body(model.geom_bodyid[contact.geom1]).name
        body2_name = model.body(model.geom_bodyid[contact.geom2]).name
        
        # Create unique key for this body pair
        key = tuple(sorted([body1_name, body2_name])) 
        if key not in contact_groups:
            contact_groups[key] = []
        contact_groups[key].append(i)
    
    # Print grouped contacts
    for (body1, body2), contact_indices in contact_groups.items():
        print(f"\n{body1} <-> {body2}: {len(contact_indices)} contact points")
        for idx in contact_indices:
            contact = data.contact[idx]
            print(f"  Contact {idx}:")
            print(f"    Position: [{contact.pos[0]:.4f}, {contact.pos[1]:.4f}, {contact.pos[2]:.4f}]")
            print(f"    Normal:   [{contact.frame[0]:.3f}, {contact.frame[1]:.3f}, {contact.frame[2]:.3f}]")
            print(f"    Distance: {contact.dist:.6f}")
            
        # Calculate spread of contact points
        if len(contact_indices) > 1:
            positions = np.array([data.contact[idx].pos for idx in contact_indices])
            center = positions.mean(axis=0)
            spread = np.linalg.norm(positions - center, axis=1).max()
            print(f"  Contact spread: {spread*1000:.2f} mm from center {center}")


def get_object_position(model: mj.MjModel, data: mj.MjData) -> np.ndarray | None:
    """Return world position of the object body."""
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'object')
    if obj_id < 0:
        return None
    return data.xpos[obj_id].copy()


def reset_to_initial_grasp(model: mj.MjModel, data: mj.MjData):
    """Reset MuJoCo state using the 'initial_grasp_cube' keyframe."""
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp_cube')
    if key_id < 0:
        raise RuntimeError("Keyframe 'initial_grasp_cube' not found in scene.xml")
    mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)

    obj_pos = get_object_position(model, data)
    obj_pos = get_object_position(model, data)
    print("\nInitialized from keyframe 'initial_grasp_cube':")
    print(f"  qpos[:16] (hand joints) = {data.qpos[:16]}")  
    print(f"    Index  (ffj0-3): {data.qpos[0:4]}")   
    print(f"    Middle (mfj0-3): {data.qpos[4:8]}") 
    print(f"    Ring   (rfj0-3): {data.qpos[8:12]}")  
    print(f"    Thumb  (thj0-3): {data.qpos[12:16]}") 
    print(f"  Object position   = {obj_pos}")
    


def hold_grasp(model: mj.MjModel, data: mj.MjData, viewer):
    """Hold initial grasp and monitor contacts."""
    reset_to_initial_grasp(model, data)
    hold_steps = 5000
    for step in range(hold_steps):
        mj.mj_step(model, data)
        viewer.sync()
        if step % 200 == 0:
            obj_pos = get_object_position(model, data)
            print(f"\nHold step {step}: object position {obj_pos}")
            print_contact_info(model, data)


def main():
    parser = argparse.ArgumentParser(description="Start from grasped configuration and monitor contacts.")
    parser.add_argument('--model', default='mjcf/scene.xml', help='Path to scene XML')
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(args.model)
    data = mj.MjData(model)

     # Reset immediately so the viewer opens already in the grasp state
    reset_to_initial_grasp(model, data)

    with mjv.launch_passive(model, data) as viewer:
        input("Viewer ready. Press Enter to start simulation...")
        hold_grasp(model, data, viewer)
        print("\nHolding grasp. Close viewer to exit.")
        while viewer.is_running():
            mj.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == '__main__':
    main()