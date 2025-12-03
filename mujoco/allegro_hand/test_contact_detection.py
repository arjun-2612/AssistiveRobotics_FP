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

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = model.geom(contact.geom1).name or f"geom_{contact.geom1}"
        geom2_name = model.geom(contact.geom2).name or f"geom_{contact.geom2}"
        body1_name = model.body(model.geom_bodyid[contact.geom1]).name
        body2_name = model.body(model.geom_bodyid[contact.geom2]).name
        print(f"\nContact {i}:")
        print(f"  Geoms: {geom1_name} <-> {geom2_name}")
        print(f"  Bodies: {body1_name} <-> {body2_name}")
        print(f"  Position: {contact.pos}")
        print(f"  Normal:   {contact.frame[:3]}")
        print(f"  Distance: {contact.dist:.6f}")
        if 'object' in (body1_name + body2_name):
            print("  *** OBJECT CONTACT ***")


def get_object_position(model: mj.MjModel, data: mj.MjData) -> np.ndarray | None:
    """Return world position of the object body."""
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'object')
    if obj_id < 0:
        return None
    return data.xpos[obj_id].copy()


def reset_to_initial_grasp(model: mj.MjModel, data: mj.MjData):
    """Reset MuJoCo state using the 'initial_grasp' keyframe."""
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
    if key_id < 0:
        raise RuntimeError("Keyframe 'initial_grasp' not found in scene.xml")
    mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)

    obj_pos = get_object_position(model, data)
    obj_pos = get_object_position(model, data)
    print("\nInitialized from keyframe 'initial_grasp':")
    print(f"  qpos[:16] (hand joints) = {data.qpos[:16]}")  
    print(f"    Index  (ffj0-3): {data.qpos[0:4]}")   
    print(f"    Middle (mfj0-3): {data.qpos[4:8]}") 
    print(f"    Ring   (rfj0-3): {data.qpos[8:12]}")  
    print(f"    Thumb  (thj0-3): {data.qpos[12:16]}") 
    print(f"  Object position   = {obj_pos}")
    


def hold_grasp(model: mj.MjModel, data: mj.MjData, viewer):
    """Hold initial grasp and monitor contacts."""
    reset_to_initial_grasp(model, data)
    hold_steps = 24000
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