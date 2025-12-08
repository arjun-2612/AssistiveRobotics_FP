"""
MuJoCo simulation for Allegro Hand cube reorientation using a trained ONNX policy.

This script deploys the trained policy in MuJoCo:
- Loads the Allegro hand and a cube
- Constructs observations matching the Isaac Lab environment
- Runs the ONNX policy to get actions
- Applies actions with EMA smoothing (alpha=0.95)

Usage:
    python rl_cube_reorient.py --policy path/to/policy.onnx
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import os
from scipy.spatial.transform import Rotation


class AllegroHandEnv:
    """MuJoCo environment for Allegro Hand cube reorientation deployment."""
    
    # Joint names in order matching Isaac Lab alphabetical ordering:
    # Isaac Lab uses regex ".*" which matches joints alphabetically:
    #   index_joint_0, index_joint_1, index_joint_2, index_joint_3,
    #   middle_joint_0, middle_joint_1, middle_joint_2, middle_joint_3,
    #   ring_joint_0, ring_joint_1, ring_joint_2, ring_joint_3,
    #   thumb_joint_0, thumb_joint_1, thumb_joint_2, thumb_joint_3
    # MuJoCo mapping: ff=index, mf=middle, rf=ring, th=thumb
    JOINT_NAMES = [
        "ffj0", "ffj1", "ffj2", "ffj3",  # Index finger (0-3)
        "mfj0", "mfj1", "mfj2", "mfj3",  # Middle finger (0-3)
        "rfj0", "rfj1", "rfj2", "rfj3",  # Ring finger (0-3)
        "thj0", "thj1", "thj2", "thj3",  # Thumb (0-3)
    ]
    
    # Actuator names in same order
    ACTUATOR_NAMES = [
        "ffa0", "ffa1", "ffa2", "ffa3",  # Index finger
        "mfa0", "mfa1", "mfa2", "mfa3",  # Middle finger
        "rfa0", "rfa1", "rfa2", "rfa3",  # Ring finger
        "tha0", "tha1", "tha2", "tha3",  # Thumb
    ]
    
    def __init__(self, model_path: str, policy_path: str, ema_alpha: float = 0.95):
        """Initialize the environment.
        
        Args:
            model_path: Path to the MJCF scene file
            policy_path: Path to the ONNX policy file
            ema_alpha: EMA smoothing factor for actions (default 0.95)
        """
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set simulation timestep to match Isaac Lab (1/120s)
        self.model.opt.timestep = 1.0 / 120.0
        self.decimation = 4  # Policy runs every 4 sim steps
        
        # EMA smoothing for actions
        self.ema_alpha = ema_alpha
        self.last_action = np.zeros(16)
        self.current_targets = np.zeros(16)
        
        # Get joint and actuator indices
        self._setup_joint_mapping()
        
        # Load ONNX policy
        self._load_policy(policy_path)
        
        # Goal orientation (quaternion w, x, y, z)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.goal_pos_offset = np.array([0.0, 0.0, -0.04])  # From Isaac Lab config
        
        # Orientation success threshold (rad)
        self.orientation_success_threshold = 0.1
        
        # Initial state
        self.reset()
    
    def _setup_joint_mapping(self):
        """Setup mapping from joint names to indices."""
        self.joint_ids = []
        self.joint_limits = []
        
        for name in self.JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                raise ValueError(f"Joint {name} not found in model")
            self.joint_ids.append(jnt_id)
            
            # Get joint limits
            jnt_range = self.model.jnt_range[jnt_id]
            self.joint_limits.append((jnt_range[0], jnt_range[1]))
        
        self.joint_limits = np.array(self.joint_limits)
        
        # Actuator indices
        self.actuator_ids = []
        for name in self.ACTUATOR_NAMES:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id == -1:
                raise ValueError(f"Actuator {name} not found in model")
            self.actuator_ids.append(act_id)
        
        # Object body index
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if self.object_body_id == -1:
            print("Warning: Object body not found. Make sure scene has 'object' body.")
    
    def _load_policy(self, policy_path: str):
        """Load ONNX policy."""
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        self.policy = ort.InferenceSession(
            policy_path,
            providers=['CPUExecutionProvider']
        )
        
        self.policy_input_name = self.policy.get_inputs()[0].name
        self.policy_output_name = self.policy.get_outputs()[0].name
        
        print(f"Loaded policy from {policy_path}")
        print(f"  Input: {self.policy_input_name}, shape: {self.policy.get_inputs()[0].shape}")
        print(f"  Output: {self.policy_output_name}, shape: {self.policy.get_outputs()[0].shape}")
    
    def reset(self):
        """Reset the environment.
        
        Hand starts in default configuration (from XML).
        Cube spawns just above the palm at a fixed position.
        A random goal orientation is sampled for the cube.
        """
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions to match Isaac Lab
        # All joints at 0 except thumb_joint_0 (thj0) at 0.28
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            if self.JOINT_NAMES[i] == "thj0":
                self.data.qpos[qpos_adr] = 0.28
            else:
                self.data.qpos[qpos_adr] = 0.0
        
        # Set cube position relative to palm
        if self.object_body_id != -1:
            # Get palm body id and its position/orientation
            palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
            
            # Run forward kinematics to get palm world position
            mujoco.mj_forward(self.model, self.data)
            
            palm_pos = self.data.xpos[palm_body_id].copy()
            palm_quat = self.data.xquat[palm_body_id].copy()  # w, x, y, z
            
            # Convert palm quaternion to rotation matrix
            palm_rot = Rotation.from_quat([palm_quat[1], palm_quat[2], palm_quat[3], palm_quat[0]])
            palm_rot_matrix = palm_rot.as_matrix()
            
            # Cube offset in palm's local frame (in front of palm)
            # Adjust this offset based on where the cube should be
            local_offset = np.array([0.0, 0.0, 0.15])  # 15cm in front of palm
            
            # Transform to world frame
            cube_pos = palm_pos + palm_rot_matrix @ local_offset
            
            # Find the freejoint for the object and set its position
            for j in range(self.model.njnt):
                if (self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and
                    self.model.jnt_bodyid[j] == self.object_body_id):
                    qpos_adr = self.model.jnt_qposadr[j]
                    # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
                    self.data.qpos[qpos_adr:qpos_adr+3] = cube_pos
                    # Identity orientation
                    self.data.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
                    # Zero velocities
                    dof_adr = self.model.jnt_dofadr[j]
                    self.data.qvel[dof_adr:dof_adr+6] = 0.0
                    break
        
        # Reset action buffers
        self.last_action = np.zeros(16)
        self.current_targets = self._get_joint_positions()
        
        # Generate random goal orientation for the cube
        self._sample_goal_orientation()
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def _sample_goal_orientation(self):
        """Sample a random goal orientation."""
        random_rot = Rotation.random()
        q = random_rot.as_quat()  # x, y, z, w format from scipy
        # Convert to w, x, y, z format (Isaac Lab convention)
        self.goal_quat = np.array([q[3], q[0], q[1], q[2]])
    
    def _get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        positions = np.zeros(16)
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            positions[i] = self.data.qpos[qpos_adr]
        return positions
    
    def _get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        velocities = np.zeros(16)
        for i, jnt_id in enumerate(self.joint_ids):
            dof_adr = self.model.jnt_dofadr[jnt_id]
            velocities[i] = self.data.qvel[dof_adr]
        return velocities
    
    def _get_object_state(self):
        """Get object position, orientation, and velocities."""
        if self.object_body_id == -1:
            return (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 
                    np.zeros(3), np.zeros(3))
        
        pos = self.data.xpos[self.object_body_id].copy()
        quat = self.data.xquat[self.object_body_id].copy()

        print(f"Cube pos: {pos}, quat: {quat}")
        
        # Get velocities from freejoint
        lin_vel = np.zeros(3)
        ang_vel = np.zeros(3)
        for j in range(self.model.njnt):
            if (self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and 
                self.model.jnt_bodyid[j] == self.object_body_id):
                dof_adr = self.model.jnt_dofadr[j]
                ang_vel = self.data.qvel[dof_adr:dof_adr+3].copy()
                lin_vel = self.data.qvel[dof_adr+3:dof_adr+6].copy()
                break
        
        return pos, quat, lin_vel, ang_vel
    
    def _normalize_joint_pos(self, positions: np.ndarray) -> np.ndarray:
        """Normalize joint positions to [-1, 1] based on limits."""
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return 2.0 * (positions - low) / (high - low) - 1.0
    
    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate (w, x, y, z format)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def get_observation(self) -> np.ndarray:
        """Construct observation vector matching Isaac Lab environment.
        
        Observation structure (72 dimensions):
        - joint_pos: 16 (normalized to [-1, 1])
        - joint_vel: 16 (scaled by 0.2)
        - object_pos: 3
        - object_quat: 4
        - object_lin_vel: 3
        - object_ang_vel: 3 (scaled by 0.2)
        - goal_pose: 7 (pos + quat from command)
        - goal_quat_diff: 4
        - last_action: 16
        """
        obs = []
        
        # Joint positions (normalized)
        joint_pos = self._get_joint_positions()
        obs.append(self._normalize_joint_pos(joint_pos))
        
        # Joint velocities (scaled)
        obs.append(self._get_joint_velocities() * 0.2)
        
        # Object state
        obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = self._get_object_state()
        obs.append(obj_pos)
        obs.append(obj_quat)
        obs.append(obj_lin_vel)
        obs.append(obj_ang_vel * 0.2)
        
        # Goal pose
        obs.append(obj_pos + self.goal_pos_offset)
        obs.append(self.goal_quat)
        
        # Goal quaternion difference
        goal_quat_conj = self._quat_conjugate(self.goal_quat)
        quat_diff = self._quat_multiply(obj_quat, goal_quat_conj)
        obs.append(quat_diff)
        
        # Last action
        obs.append(self.last_action)
        
        return np.concatenate(obs).astype(np.float32)
    
    def _rescale_action_to_limits(self, action: np.ndarray) -> np.ndarray:
        """Rescale action from [-1, 1] to joint limits."""
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        scaled = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(scaled, low, high)
    
    def get_orientation_error(self) -> float:
        """Get current orientation error in radians."""
        _, obj_quat, _, _ = self._get_object_state()
        goal_quat_conj = self._quat_conjugate(self.goal_quat)
        quat_diff = self._quat_multiply(obj_quat, goal_quat_conj)
        return 2.0 * np.arccos(np.clip(np.abs(quat_diff[0]), 0.0, 1.0))
    
    def step(self, action: np.ndarray) -> np.ndarray:
        """Execute action and return new observation.
        
        Args:
            action: 16D action vector (normalized to [-1, 1])
        
        Returns:
            observation: New observation after stepping
        """
        # Store action
        self.last_action = action.copy()
        
        # Apply EMA smoothing
        new_targets = self._rescale_action_to_limits(action)
        self.current_targets = (self.ema_alpha * self.current_targets + 
                                (1 - self.ema_alpha) * new_targets)
        
        # Set actuator controls
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = self.current_targets[i]
        
        # Step simulation (decimation steps)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        
        return self.get_observation()
    
    def check_success_and_resample(self) -> bool:
        """Check if goal reached, resample if so. Returns True if goal was reached."""
        if self.get_orientation_error() < self.orientation_success_threshold:
            self._sample_goal_orientation()
            return True
        return False


def run_deployment(env: AllegroHandEnv):
    """Run the policy deployment indefinitely."""
    print("\nStarting deployment...")
    print("Close the viewer window to stop.\n")
    
    env.reset()
    step = 0
    goals_reached = 0
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # Get observation
            obs = env.get_observation()
            
            # Get action from policy
            obs_input = obs.reshape(1, -1).astype(np.float32)
            action = env.policy.run(
                [env.policy_output_name],
                {env.policy_input_name: obs_input}
            )[0][0]
            
            # Clip and execute action
            action = np.clip(action, -1.0, 1.0)
            env.step(action)
            
            # Check if goal reached, resample if so
            if env.check_success_and_resample():
                goals_reached += 1
                # print(f"Goal {goals_reached} reached! New goal sampled.")
            
            step += 1
            viewer.sync()
            
            # Periodic status update
            if step % 500 == 0:
                error = env.get_orientation_error()
                # print(f"Step {step}: orientation_error={error:.4f} rad, goals_reached={goals_reached}")


def main():
    parser = argparse.ArgumentParser(description="Allegro Hand Cube Reorientation - Policy Deployment")
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to ONNX policy file")
    parser.add_argument("--scene", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "mjcf", "scene.xml"),
                        help="Path to MJCF scene file")
    parser.add_argument("--ema-alpha", type=float, default=0.95,
                        help="EMA smoothing factor for actions")
    args = parser.parse_args()
    
    # Create environment
    env = AllegroHandEnv(
        model_path=args.scene,
        policy_path=args.policy,
        ema_alpha=args.ema_alpha
    )
    
    print("\nEnvironment ready")
    print(f"  Observation dim: {env.get_observation().shape[0]}")
    print(f"  Action dim: 16")
    print(f"  EMA alpha: {args.ema_alpha}")
    
    run_deployment(env)


if __name__ == "__main__":
    main()
