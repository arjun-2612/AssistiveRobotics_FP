"""
Hand Jacobian - Section II.A
Implements the hand Jacobian J that relates joint velocities to contact velocities.

Key equations:
- ċ = J q̇       (Equation 2) - Contact velocities from joint velocities
- τ = J^T f     (Equation 4) - Joint torques from contact forces
"""

import numpy as np
import mujoco as mj
from typing import List


class HandJacobian:
    """
    Computes the hand Jacobian J ∈ R^(3n x m).
    
    The Jacobian relates:
    - Joint velocities q̇ to contact velocities: ċ = J q̇
    - Contact forces f to joint torques: τ = J^T f
    """
    
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        """
        Initialize hand Jacobian computer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
    def compute_contact_jacobian(self, contact_body_ids: List[int], contact_local_positions: List[np.ndarray]) -> np.ndarray:
        """
        Compute Jacobian J for multiple contact points.
        
        For each contact i on body b at local position p_local:
        J_i = J_linear(body=b, point=p_local)
        
        Args:
            contact_body_ids: List of body IDs for each contact
            contact_local_positions: Local positions of contacts on their bodies
            
        Returns:
            J: Hand Jacobian (3n × m)
        """
        n_contacts = len(contact_body_ids)
        n_joints = self.model.nv  # Total DOFs
        
        # Initialize Jacobian
        J = np.zeros((3 * n_contacts, n_joints))
        
        for i, (body_id, local_pos) in enumerate(zip(contact_body_ids, contact_local_positions)):
            # Compute Jacobian for this contact point
            J_i = self._compute_point_jacobian(body_id, local_pos)
            
            # Place in full Jacobian (only linear part, 3 rows)
            J[3*i:3*i+3, :] = J_i[:3, :]  # Only linear velocity part
            
        return J
    
    def _compute_point_jacobian(self, body_id: int, local_position: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian for a single point on a body.
        
        Args:
            body_id: MuJoCo body ID
            local_position: Position in body frame (3D)
            
        Returns:
            J_point: 6 x nv Jacobian (linear + angular)
        """
        # Allocate Jacobian matrices
        jacp = np.zeros((3, self.model.nv))  # Linear velocity Jacobian
        jacr = np.zeros((3, self.model.nv))  # Angular velocity Jacobian
        
        # Compute Jacobian using MuJoCo
        mj.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        
        # If we need Jacobian at a different point (not body center),
        # we need to adjust using the angular velocity Jacobian
        if np.linalg.norm(local_position) > 1e-10:
            # Get current body rotation matrix
            body_rot = self.data.xmat[body_id].reshape(3, 3)
            
            # Transform local offset to world frame
            world_offset = body_rot @ local_position
            
            # Adjust linear velocity Jacobian:
            # v_point = v_body_center + ω × r
            # J_point = J_center + [r]× * J_rotation
            # where [r]× is the skew-symmetric matrix of r
            
            # Skew-symmetric matrix of world offset
            r_skew = np.array([
                [0, -world_offset[2], world_offset[1]],
                [world_offset[2], 0, -world_offset[0]],
                [-world_offset[1], world_offset[0], 0]
            ])
            
            # Adjust linear Jacobian
            jacp = jacp + r_skew @ jacr
        
        # Stack linear and angular parts
        J_point = np.vstack([jacp, jacr])
        
        return J_point
    
    def contact_forces_to_joint_torques(self, contact_forces: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """
        Map contact forces to joint torques (Equation 4).
        
        τ = J^T f
        
        Args:
            contact_forces: f ∈ R^(3n) - forces at all contacts
            jacobian: J ∈ R^(3n×m) - hand Jacobian
            
        Returns:
            joint_torques: τ ∈ R^m
        """
        # τ = J^T f
        joint_torques = jacobian.T @ contact_forces
        
        return joint_torques
    
    def joint_velocities_to_contact_velocities(self, joint_velocities: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """
        Map joint velocities to contact velocities (Equation 2).
        
        ċ = J q̇
        
        Args:
            joint_velocities: q̇ ∈ R^m
            jacobian: J ∈ R^(3n×m)
            
        Returns:
            contact_velocities: ċ ∈ R^(3n)
        """
        # ċ = J q̇
        contact_velocities = jacobian @ joint_velocities
        
        return contact_velocities