#!/usr/bin/env python3
"""
Full Integration Test - Complete Control Pipeline
Tests all components from the paper working together:

Section II.A - Modeling:
  - Object state estimation
  - Contact detection
  - Grasp matrix computation
  - Hand Jacobian computation

Section III.A - Object Controller (Impedance):
  - Desired contact motion from object error
  - Spring-damper contact forces

Section III.B - Internal Forces (QP):
  - Optimal internal force distribution
  - Friction cone constraints

Section III.C - Force Mapping:
  - Combine impedance + internal forces
  - Map to joint torques

Section III.D - Nullspace Controller:
  - Joint coupling (q3-q4)
  - Prevent drift/limits

Section III.E - Contact Management:
  - Smooth contact transitions
  - Activation functions
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
import sys
sys.path.append('controllers')

from controllers import (
    ObjectPoseEstimator,
    ImpedanceController,
    GraspMatrix,
    HandJacobian,
    ContactDetector,
    InternalForceOptimizer,
    ForceMapper,
    NullspaceController,
    ContactManager
)


class FullPipelineController:
    """
    Complete controller integrating all components from the paper.
    """
    
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        """
        Initialize full pipeline controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Section II.A - Modeling components
        print("Initializing modeling components (Section II.A)...")
        self.object_estimator = ObjectPoseEstimator(model, 'object')
        self.contact_detector = ContactDetector(model, 'object')
        self.grasp_matrix = GraspMatrix(n_contacts=2)
        self.hand_jacobian = HandJacobian(model, data)
        
        # Section III.A - Object controller (Impedance)
        print("Initializing impedance controller (Section III.A)...")
        self.impedance_controller = ImpedanceController(
            contact_stiffness=100.0,
            contact_damping=10.0,
            n_contacts=2
        )
        
        # Section III.B - Internal forces (QP)
        print("Initializing QP optimizer (Section III.B)...")
        self.qp_optimizer = InternalForceOptimizer(
            n_contacts=2,
            friction_coefficient=0.8,
            f_min=0.5,
            f_max=30.0
        )
        
        # Section III.C - Force mapping
        print("Initializing force mapper (Section III.C)...")
        self.force_mapper = ForceMapper()
        
        # Section III.D - Nullspace controller
        print("Initializing nullspace controller (Section III.D)...")
        self.nullspace_controller = NullspaceController(
            n_joints_per_finger=4,
            n_fingers=2,
            K_null=2.0,
            D_null=0.2
        )
        
        # Section III.E - Contact manager
        print("Initializing contact manager (Section III.E)...")
        self.contact_manager = ContactManager(transition_time=0.1)
        
        # Control parameters
        self.dt = model.opt.timestep
        self.initialized = False
        
        print("✓ Full pipeline controller initialized!\n")
        
    def initialize_contacts(self):
        """Initialize contacts from current configuration."""
        contacts = self.contact_detector.get_object_contacts(self.data)
        
        if len(contacts) == 0:
            print("⚠ Warning: No contacts detected!")
            return False
            
        print(f"Detected {len(contacts)} contacts:")
        for i, contact in enumerate(contacts):
            print(f"  Contact {i}: pos={contact['position']}, "
                  f"body={contact['body_name']}")
            
            # Add to contact manager
            self.contact_manager.add_contact(
                position=contact['position'],
                normal=contact['normal'],
                contact_id=i
            )
        
        # Store initial contact positions for impedance controller
        c_initial = self.contact_detector.get_contact_positions_vector(self.data)
        if c_initial is not None:
            self.impedance_controller.set_initial_contacts(c_initial)
            
        self.initialized = True
        return True
        
    def compute_control(self, x_desired: np.ndarray) -> np.ndarray:
        """
        Compute control torques for desired object pose.
        
        This implements the FULL PIPELINE from the paper.
        
        Args:
            x_desired: Desired object pose (6D)
            
        Returns:
            tau_des: Desired joint torques (8D for 2-finger hand)
        """
        if not self.initialized:
            print("⚠ Controller not initialized! Call initialize_contacts() first.")
            return np.zeros(8)
        
        # ============================================================
        # SECTION II.A - Get current state
        # ============================================================
        
        # Get object state
        object_state = self.object_estimator.get_object_state(self.data)
        x_current = self.object_estimator.get_6d_pose_vector(self.data)
        v_current = np.concatenate([
            object_state['linear_velocity'],
            object_state['angular_velocity']
        ])
        
        # Update contact manager activations (Eq. 34, 36)
        self.contact_manager.update_activations()
        
        # Get current contacts
        contacts = self.contact_detector.get_object_contacts(self.data)
        if len(contacts) == 0:
            return np.zeros(8)
        
        # Get contact positions and normals
        c_positions = np.array([c['position'] for c in contacts])
        c_normals = np.array([c['normal'] for c in contacts])
        c_flat = c_positions.flatten()
        
        # Estimate contact velocities (simplified - should use Jacobian)
        c_vel = np.zeros(6)  # Placeholder
        
        # Compute grasp matrix G (Eq. 1, 3)
        G = self.grasp_matrix.compute_grasp_matrix(
            c_positions,
            object_state['position']
        )
        
        # Compute W matrix for Euler angles
        euler_angles = x_current[3:]
        W = ImpedanceController.compute_W_matrix(euler_angles)
        
        # Compute hand Jacobian J (Eq. 2, 4)
        contact_body_ids = [c['body_id'] for c in contacts]
        contact_local_pos = [np.zeros(3) for _ in contacts]  # Simplified
        J = self.hand_jacobian.compute_contact_jacobian(
            contact_body_ids,
            contact_local_pos
        )
        
        # Get joint states
        q = self.data.qpos[:8].copy()  # Assuming first 8 are finger joints
        qd = self.data.qvel[:8].copy()
        
        # ============================================================
        # SECTION III.A - Impedance Controller (Eq. 10-14)
        # ============================================================
        
        print("\n--- Section III.A: Impedance Controller ---")
        
        # Compute desired contact forces (Eq. 14)
        f_x = self.impedance_controller.compute_contact_forces(
            x_current,
            x_desired,
            c_flat,
            c_vel,
            G,
            W
        )
        
        print(f"Impedance forces f_x: {f_x}")
        print(f"  Magnitude: {np.linalg.norm(f_x):.3f} N")
        
        # ============================================================
        # SECTION III.B - Internal Forces QP (Eq. 15-27)
        # ============================================================
        
        print("\n--- Section III.B: Internal Forces (QP) ---")
        
        # Compute desired wrench (simplified - just gravity compensation)
        object_mass = 0.05  # kg
        g = 9.81
        w_desired = np.array([0, 0, object_mass * g, 0, 0, 0])
        
        print(f"Desired wrench: {w_desired}")
        
        # Get activation values (Eq. 35)
        activation_values = self.contact_manager.get_activation_vector()
        print(f"Contact activations: {activation_values}")
        
        # Compute internal forces with activation
        desired_normal_forces = np.array([5.0, 5.0])  # 5N squeeze per finger
        
        f_int_total, qp_info = self.qp_optimizer.compute_contact_forces_with_activation(
            w_desired,
            G,
            desired_normal_forces,
            c_normals,
            activation_values
        )
        
        print(f"Internal forces f_int: {qp_info['f_internal']}")
        print(f"  QP status: {qp_info['qp_status']}")
        print(f"  Optimal: {qp_info['qp_optimal']}")
        
        # ============================================================
        # SECTION III.C - Force Mapping (Eq. 28-29)
        # ============================================================
        
        print("\n--- Section III.C: Force Mapping ---")
        
        # Combine forces (Eq. 28)
        f_c = self.force_mapper.combine_forces(f_x, qp_info['f_internal'])
        print(f"Combined contact forces f_c: {f_c}")
        print(f"  Magnitude: {np.linalg.norm(f_c):.3f} N")
        
        # Map to joint torques (Eq. 29)
        tau_c = self.force_mapper.map_to_joint_torques(f_c, J)
        print(f"Contact torques τ_c: {tau_c}")
        print(f"  Magnitude: {np.linalg.norm(tau_c):.3f} Nm")
        
        # ============================================================
        # SECTION III.D - Nullspace Controller (Eq. 30-33)
        # ============================================================
        
        print("\n--- Section III.D: Nullspace Controller ---")
        
        # Compute nullspace torques
        tau_des, ns_info = self.nullspace_controller.compute_full_pipeline(
            tau_c,
            q,
            qd,
            J
        )
        
        print(f"Nullspace torques τ_null: {ns_info['tau_null_projected']}")
        print(f"  Magnitude: {ns_info['nullspace_magnitude']:.3f} Nm")
        
        # Analyze joint coupling
        coupling_info = self.nullspace_controller.analyze_joint_coupling(q)
        for finger_info in coupling_info['fingers']:
            if 'coupling_error' in finger_info:
                print(f"  Finger {finger_info['finger_id']}: "
                      f"q3={finger_info['q3']:.3f}, "
                      f"q4={finger_info['q4']:.3f}, "
                      f"error={finger_info['coupling_error']:.3f}")
        
        print(f"\nFinal desired torques τ_des: {tau_des}")
        print(f"  Magnitude: {np.linalg.norm(tau_des):.3f} Nm")
        
        return tau_des
        
    def print_summary(self):
        """Print summary of pipeline status."""
        print("\n" + "="*60)
        print("FULL PIPELINE STATUS")
        print("="*60)
        
        # Object state
        state = self.object_estimator.get_object_state(self.data)
        print(f"\nObject State:")
        print(f"  Position: {state['position']}")
        print(f"  Orientation (euler): {state['orientation']}")
        
        # Contacts
        contact_info = self.contact_manager.get_contact_info()
        print(f"\nContacts (Section III.E):")
        print(f"  Active: {contact_info['n_contacts']}")
        print(f"  Transitioning: {contact_info['n_transitioning']}")
        print(f"  Stable: {self.contact_manager.is_stable()}")
        
        for contact in contact_info['contacts']:
            print(f"  Contact {contact['id']}: a={contact['activation']:.3f}, "
                  f"mode={contact['mode']}, transitioning={contact['is_transitioning']}")
        
        # Grasp matrix
        if self.grasp_matrix.G is not None:
            print(f"\nGrasp Matrix:")
            print(f"  Shape: {self.grasp_matrix.G.shape}")
            print(f"  Force closure: {self.grasp_matrix.is_force_closure()}")
        
        print("="*60 + "\n")


def reset_to_initial_grasp(model: mj.MjModel, data: mj.MjData):
    """Reset to initial grasp keyframe."""
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'initial_grasp')
    if key_id >= 0:
        mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)


def test_full_pipeline():
    """
    Test the complete control pipeline from the paper.
    """
    print("\n" + "="*60)
    print("FULL INTEGRATION TEST")
    print("Object-Level Impedance Control for Dexterous In-Hand Manipulation")
    print("="*60 + "\n")
    
    # Load model
    print("Loading MuJoCo model...")
    model = mj.MjModel.from_xml_path('mjcf/scene.xml')
    data = mj.MjData(model)
    reset_to_initial_grasp(model, data)
    print("✓ Model loaded\n")
    
    # Initialize controller
    print("="*60)
    print("INITIALIZING FULL PIPELINE CONTROLLER")
    print("="*60)
    controller = FullPipelineController(model, data)
    
    # Initialize contacts
    print("\n" + "="*60)
    print("DETECTING AND INITIALIZING CONTACTS")
    print("="*60)
    if not controller.initialize_contacts():
        print("❌ Failed to initialize contacts!")
        return
    
    print("✓ Contacts initialized\n")
    
    # Get initial object state
    initial_state = controller.object_estimator.get_object_state(data)
    x_initial = controller.object_estimator.get_6d_pose_vector(data)
    
    print(f"Initial object state:")
    print(f"  Position: {initial_state['position']}")
    print(f"  Orientation: {initial_state['orientation']}")
    
    # Define desired pose: lift object 2cm up
    x_desired = x_initial.copy()
    x_desired[2] += 0.02  # 2cm up in z
    
    print(f"\nDesired object state:")
    print(f"  Position: {x_desired[:3]}")
    print(f"  Orientation: {x_desired[3:]}")
    
    # Test 1: Static test (compute control once)
    print("\n" + "="*60)
    print("TEST 1: SINGLE CONTROL COMPUTATION")
    print("="*60)
    
    tau_des = controller.compute_control(x_desired)
    controller.print_summary()
    
    print(f"\n✓ Control computation successful!")
    print(f"  Final torques: {tau_des}")
    
    # Test 2: Closed-loop simulation
    print("\n" + "="*60)
    print("TEST 2: CLOSED-LOOP SIMULATION")
    print("="*60)
    
    with mjv.launch_passive(model, data) as viewer:
        reset_to_initial_grasp(model, data)
        controller.initialize_contacts()
        
        print("\nRunning closed-loop control...")
        print("Watch the object move upward smoothly!\n")
        
        n_steps = 3000  # 3 seconds at 1kHz
        
        for step in range(n_steps):
            # Compute control
            tau_des = controller.compute_control(x_desired)
            
            # Apply torques to actuators
            data.ctrl[:8] = tau_des
            
            # Step simulation
            mj.mj_step(model, data)
            viewer.sync()
            
            # Print progress
            if step % 300 == 0:
                state = controller.object_estimator.get_object_state(data)
                x_current = controller.object_estimator.get_6d_pose_vector(data)
                error = np.linalg.norm(x_desired[:3] - x_current[:3])
                
                print(f"\nStep {step}/3000:")
                print(f"  Current z: {x_current[2]:.4f} m")
                print(f"  Target z: {x_desired[2]:.4f} m")
                print(f"  Position error: {error:.4f} m")
                print(f"  Torque norm: {np.linalg.norm(tau_des):.3f} Nm")
        
        # Final results
        final_state = controller.object_estimator.get_object_state(data)
        x_final = controller.object_estimator.get_6d_pose_vector(data)
        final_error = np.linalg.norm(x_desired[:3] - x_final[:3])
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Initial position: {x_initial[:3]}")
        print(f"Target position: {x_desired[:3]}")
        print(f"Final position: {x_final[:3]}")
        print(f"Position error: {final_error:.4f} m ({final_error*1000:.2f} mm)")
        
        if final_error < 0.005:  # 5mm tolerance
            print("\n✅ SUCCESS! Object reached target within 5mm tolerance!")
        else:
            print(f"\n⚠️  Object did not fully reach target (error: {final_error*1000:.1f}mm)")
            print("   This may require tuning controller gains.")
        
        controller.print_summary()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETE")
        print("="*60)
        print("\nPress Enter to exit...")
        input()


if __name__ == '__main__':
    try:
        test_full_pipeline()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()