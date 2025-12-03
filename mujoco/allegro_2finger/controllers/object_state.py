import numpy as np
import mujoco as mj
from typing import Dict, Optional


class ObjectPoseEstimator:
    """Estimates object pose and velocity from MuJoCo simulation."""
    
    def __init__(self, model: mj.MjModel, object_body_name: str = 'object'):
        """
        Args:
            model: MuJoCo model
            object_body_name: Name of the object body in the XML
        """
        self.model = model
        self.object_body_name = object_body_name
        
        # Get object body ID
        self.object_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name)
        if self.object_id < 0:
            raise ValueError(f"Object body '{object_body_name}' not found in model")
        
        # History for numerical differentiation if needed
        self.pose_history = []
        self.max_history = 10
        
    def get_object_state(self, data: mj.MjData) -> Dict:
        """
        Get complete object state: position, orientation, velocities.
        
        Args:
            data: MuJoCo data structure
            
        Returns:
            Dictionary with position, orientation, rotation_matrix, velocities
        """
        # Position (3D)
        position = data.xpos[self.object_id].copy()
        
        # Orientation as quaternion (w, x, y, z)
        orientation = data.xquat[self.object_id].copy()
        
        # Rotation matrix from quaternion
        rotation_matrix = self._quat_to_matrix(orientation)
        
        # Velocities (linear and angular)
        # For freejoint bodies, velocities are in qvel
        linear_velocity, angular_velocity = self._get_velocities(data)
        
        state = {
            'position': position,
            'orientation': orientation,  # (w, x, y, z)
            'rotation_matrix': rotation_matrix,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity
        }
        
        # Store in history
        self.pose_history.append({
            'time': data.time,
            'position': position.copy(),
            'orientation': orientation.copy()
        })
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        return state
    
    def _get_velocities(self, data: mj.MjData):
        """Get linear and angular velocities of the object."""
        # Find the freejoint for this body
        # Freejoint has 6 velocity DOFs: linear (3) + angular (3)
        
        # Get DOF address for this body
        jnt_adr = self.model.body_jntadr[self.object_id]
        if jnt_adr < 0:
            return np.zeros(3), np.zeros(3)
        
        jnt_type = self.model.jnt_type[jnt_adr]
        
        if jnt_type == mj.mjtJoint.mjJNT_FREE:
            # Freejoint: first 3 are linear velocity, next 3 are angular
            dof_adr = self.model.jnt_dofadr[jnt_adr]
            linear_vel = data.qvel[dof_adr:dof_adr+3].copy()
            angular_vel = data.qvel[dof_adr+3:dof_adr+6].copy()
            return linear_vel, angular_vel
        else:
            # Not a freejoint, compute numerically
            return self._compute_velocity_numerically()
    
    def _compute_velocity_numerically(self):
        """Fallback: compute velocity from position history."""
        if len(self.pose_history) < 2:
            return np.zeros(3), np.zeros(3)
        
        dt = self.pose_history[-1]['time'] - self.pose_history[-2]['time']
        if dt <= 0:
            return np.zeros(3), np.zeros(3)
        
        # Linear velocity
        dp = self.pose_history[-1]['position'] - self.pose_history[-2]['position']
        linear_vel = dp / dt
        
        # Angular velocity (simplified)
        # For accurate angular velocity, need quaternion derivative
        angular_vel = np.zeros(3)
        
        return linear_vel, angular_vel
    
    @staticmethod
    def _quat_to_matrix(quat):
        """
        Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
        
        Args:
            quat: Quaternion as [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quat
        
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def get_6d_pose_vector(self, data: mj.MjData) -> np.ndarray:
        """
        Get object pose as 6D vector (position + euler angles or axis-angle).
        Useful for impedance control.
        
        Returns:
            6D vector: [x, y, z, rx, ry, rz]
        """
        state = self.get_object_state(data)
        position = state['position']
        
        # Convert quaternion to Euler angles (ZYX convention)
        euler = self._quat_to_euler(state['orientation'])
        
        return np.concatenate([position, euler])
    
    @staticmethod
    def _quat_to_euler(quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])