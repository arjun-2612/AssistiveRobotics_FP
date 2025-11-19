#!/usr/bin/env python3
"""
Test contact detection with the 2-finger Allegro hand.
This is Step 1: Verify we can detect contacts before building the full controller.
"""

import argparse
import time
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
        
        # Get geom names
        geom1_name = model.geom(contact.geom1).name or f"geom_{contact.geom1}"
        geom2_name = model.geom(contact.geom2).name or f"geom_{contact.geom2}"
        
        # Get body names
        body1_id = model.geom_bodyid[contact.geom1]
        body2_id = model.geom_bodyid[contact.geom2]
        body1_name = model.body(body1_id).name or f"body_{body1_id}"
        body2_name = model.body(body2_id).name or f"body_{body2_id}"
        
        print(f"\nContact {i}:")
        print(f"  Geoms: {geom1_name} <-> {geom2_name}")
        print(f"  Bodies: {body1_name} <-> {body2_name}")
        print(f"  Position: {contact.pos}")
        print(f"  Normal: {contact.frame[:3]}")
        print(f"  Distance: {contact.dist:.6f}")
        
        # Check if object is involved
        is_object_contact = ('object' in body1_name or 'object' in body2_name or
                            'object' in geom1_name or 'object' in geom2_name)
        if is_object_contact:
            print(f"  *** OBJECT CONTACT DETECTED ***")


def get_object_contacts(model: mj.MjModel, data: mj.MjData):
    """Get contact points specifically involving the object."""
    object_contacts = []
    
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Get body names
        body1_id = model.geom_bodyid[contact.geom1]
        body2_id = model.geom_bodyid[contact.geom2]
        body1_name = model.body(body1_id).name
        body2_name = model.body(body2_id).name
        
        # Check if object is involved
        if 'object' in body1_name or 'object' in body2_name:
            object_contacts.append({
                'position': contact.pos.copy(),
                'normal': contact.frame[:3].copy(),
                'distance': contact.dist,
                'geom1': contact.geom1,
                'geom2': contact.geom2,
                'body1': body1_name,
                'body2': body2_name
            })
    
    return object_contacts


def get_object_position(model: mj.MjModel, data: mj.MjData):
    """Get the object's current position."""
    object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'object')
    if object_body_id < 0:
        return None
    
    # Get the position from xpos (Cartesian positions of bodies)
    return data.xpos[object_body_id].copy()


def test_grasp_sequence(model: mj.MjModel, data: mj.MjData, viewer):
    """Test a simple grasp sequence to create contacts."""
    print("\n" + "="*60)
    print("STARTING GRASP SEQUENCE TEST")
    print("="*60)
    
    # Joint names for 2-finger hand
    # Index finger: ffj0, ffj1, ffj2, ffj3
    # Thumb: thj0, thj1, thj2, thj3
    
    # Phase 1: Open hand wide
    print("\nPhase 1: Opening hand wide...")
    target_open = np.array([
        -0.2,   # ffj0 (base, abduction) - spread out
        0.0,    # ffj1 (proximal) - straight
        0.0,    # ffj2 (medial)
        0.0,    # ffj3 (distal)
        1.0,    # thj0 (thumb base) - oppose more
        -0.1,   # thj1 (thumb proximal) - extend back
        0.0,    # thj2 (thumb medial)
        0.0     # thj3 (thumb distal)
    ])
    
    for _ in range(1000):  # ~1 second
        data.ctrl[:] = target_open
        mj.mj_step(model, data)
        viewer.sync()
    
    obj_pos = get_object_position(model, data)
    if obj_pos is not None:
        print(f"Object position after opening: {obj_pos}")
    print_contact_info(model, data)
    
    # Phase 2: Close gradually to grasp
    print("\nPhase 2: Closing gradually to grasp object...")
    
    # Start position (open)
    start = target_open.copy()
    
    # End position (closed for grasp)
    target_grasp = np.array([
        0.0,    # ffj0 - neutral abduction
        1.4,    # ffj1 - curl index finger strongly
        1.3,    # ffj2 - curl more
        1.2,    # ffj3 - curl distal
        1.0,    # thj0 - maintain opposition
        0.8,    # thj1 - curl thumb
        1.2,    # thj2 - curl thumb medial
        1.0     # thj3 - curl thumb distal
    ])
    
    # Interpolate gradually over 2 seconds
    steps = 2000
    for step in range(steps):
        alpha = step / steps  # 0 to 1
        data.ctrl[:] = (1 - alpha) * start + alpha * target_grasp
        mj.mj_step(model, data)
        viewer.sync()
        
        # Print contacts every 200 steps
        if step % 200 == 0:
            obj_pos = get_object_position(model, data)
            if obj_pos is not None:
                print(f"\nStep {step}, Object position: {obj_pos}")
            print_contact_info(model, data)
    
    # Phase 3: Hold and monitor
    print("\nPhase 3: Holding grasp and monitoring...")
    for step in range(1000):  # 1 second hold
        data.ctrl[:] = target_grasp
        mj.mj_step(model, data)
        viewer.sync()
        
        if step % 200 == 0:
            contacts = get_object_contacts(model, data)
            obj_pos = get_object_position(model, data)
            
            print(f"\nHold step {step}:")
            if obj_pos is not None:
                print(f"  Object position: {obj_pos}")
            print(f"  Object contacts: {len(contacts)}")
            
            for j, c in enumerate(contacts):
                print(f"  Contact {j}: {c['body1']} <-> {c['body2']}")
                print(f"    Position: {c['position']}")
                print(f"    Normal: {c['normal']}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL GRASP STATUS")
    print("="*60)
    obj_pos = get_object_position(model, data)
    if obj_pos is not None:
        print(f"Final object position: {obj_pos}")
    
    contacts = get_object_contacts(model, data)
    print(f"Total object contacts: {len(contacts)}")
    
    if len(contacts) >= 2:
        print("✓ SUCCESS: Object is in contact with multiple fingers!")
    elif len(contacts) == 1:
        print("⚠ PARTIAL: Object has only one contact point")
    else:
        print("✗ FAILED: No contacts with object")


def main():
    ap = argparse.ArgumentParser(description="Test contact detection with 2-finger Allegro hand")
    ap.add_argument('--model', default='mjcf/scene.xml', help='Path to scene XML')
    ap.add_argument('--interactive', action='store_true', 
                   help='Interactive mode (manual control)')
    args = ap.parse_args()
    
    print(f"Loading model from: {args.model}")
    model = mj.MjModel.from_xml_path(args.model)
    data = mj.MjData(model)
    
    print(f"\nModel info:")
    print(f"  Number of bodies: {model.nbody}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Number of actuators: {model.nu}")
    print(f"  Number of geoms: {model.ngeom}")
    
    # Print body names
    print(f"\nBodies in model:")
    for i in range(model.nbody):
        print(f"  {i}: {model.body(i).name}")
    
    # Print actuator/joint info
    print(f"\nActuators/Joints:")
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_name = model.joint(joint_id).name if joint_id >= 0 else "N/A"
        print(f"  {i}: {model.actuator(i).name} -> joint: {joint_name}")
    
    with mjv.launch_passive(model, data) as viewer:
        if args.interactive:
            print("\n" + "="*60)
            print("INTERACTIVE MODE")
            print("Use the viewer to manually move joints and observe contacts")
            print("Press Ctrl+C to exit")
            print("="*60)
            
            try:
                while viewer.is_running():
                    mj.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.001)
            except KeyboardInterrupt:
                print("\nExiting...")
        else:
            # Run automated test sequence
            test_grasp_sequence(model, data, viewer)
            
            print("\n" + "="*60)
            print("TEST COMPLETE")
            print("Press Ctrl+C to exit")
            print("="*60)
            
            # Keep viewer open
            try:
                while viewer.is_running():
                    mj.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.001)
            except KeyboardInterrupt:
                print("\nExiting...")


if __name__ == '__main__':
    main()