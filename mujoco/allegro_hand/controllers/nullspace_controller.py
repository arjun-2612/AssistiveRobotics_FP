"""
Nullspace Controller - Section III.D
Adds torques in the nullspace of the Jacobian to:
1. Prevent joint drift and avoid joint limits
2. Couple joints (e.g., q3 and q4 in anthropomorphic fingers)

Key equations:
- τ_des = τ_c + (I - J^T(J^T)^+) τ_null  (Equation 30-31)
- τ_null^[j] = K_null Δq^[j] + D_null Δq̇^[j]  (Equation 32)
- Δq^[j] = (1/2)[0, 0, q3-q4, q3-q4]^T  (Equation 33) for coupling
"""

import numpy as np
from typing import Optional, List, Tuple


class NullspaceController:
    """
    Nullspace controller for anthropomorphic fingers.
    
    Adds torques that lie in the nullspace of J^T, meaning they:
    - Do NOT affect contact forces on the object
    - DO affect joint configuration (prevent drift, avoid limits)
    
    Particularly useful for fingers with >3 DOF like the Allegro hand.
    """
    
    def __init__(
        self,
        n_joints_per_finger: int = 4,
        n_fingers: int = 2,
        K_null: float = 1.0,
        D_null: float = 0.1
    ):
        """
        Initialize nullspace controller.
        
        Args:
            n_joints_per_finger: Number of joints per finger (e.g., 4 for Allegro)
            n_fingers: Number of fingers
            K_null: Stiffness coefficient for nullspace controller
            D_null: Damping coefficient for nullspace controller
        """
        self.n_joints_per_finger = n_joints_per_finger
        self.n_fingers = n_fingers
        self.n_joints_total = n_joints_per_finger * n_fingers
        
        self.K_null = K_null
        self.D_null = D_null
        
    def compute_nullspace_torque(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        jacobian: np.ndarray,
        finger_configs: Optional[List[dict]] = None
    ) -> np.ndarray:
        """
        Compute nullspace torque τ_null (Equation 32).
        
        For anthropomorphic fingers, couples joints 3 and 4 using Equation 33.
        
        Args:
            joint_positions: q ∈ R^m - current joint angles
            joint_velocities: q̇ ∈ R^m - current joint velocities
            jacobian: J ∈ R^(3n×m) - hand Jacobian
            finger_configs: List of finger configuration dicts (optional)
            
        Returns:
            tau_null: τ_null ∈ R^m - nullspace torque
        """
        assert joint_positions.shape[0] == self.n_joints_total
        assert joint_velocities.shape[0] == self.n_joints_total
        
        # Initialize nullspace torque
        tau_null = np.zeros(self.n_joints_total)
        
        # For each finger, compute coupling torque (Equation 33)
        for finger_idx in range(self.n_fingers):
            # Joint indices for this finger
            start_idx = finger_idx * self.n_joints_per_finger
            q_finger = joint_positions[start_idx:start_idx + self.n_joints_per_finger]
            qd_finger = joint_velocities[start_idx:start_idx + self.n_joints_per_finger]
            
            # Compute Δq^[j] for coupling joints 3 and 4 (Equation 33)
            delta_q = self._compute_joint_coupling(q_finger)
            delta_qd = self._compute_joint_coupling_velocity(qd_finger)
            
            # Joint impedance torque (Equation 32)
            # τ_null^[j] = K_null * Δq^[j] + D_null * Δq̇^[j]
            tau_null_finger = self.K_null * delta_q + self.D_null * delta_qd
            
            # Place in full torque vector
            tau_null[start_idx:start_idx + self.n_joints_per_finger] = tau_null_finger
            
        return tau_null
    
    def _compute_joint_coupling(self, q_finger: np.ndarray) -> np.ndarray:
        """
        Compute joint coupling Δq for a single finger (Equation 33).
        
        For anthropomorphic fingers (4 DOF):
        Δq^[j] = (1/2) * [0, 0, q3 - q4, q3 - q4]^T
        
        This couples joints 3 and 4 to move together.
        
        Args:
            q_finger: Joint angles for one finger (4,)
            
        Returns:
            delta_q: Coupling error (4,)
        """
        assert q_finger.shape[0] == self.n_joints_per_finger, \
            f"Expected {self.n_joints_per_finger} joints per finger"
        
        if self.n_joints_per_finger == 4:
            # Equation (33): Δq^[j] = (1/2) * [0, 0, q3 - q4, q3 - q4]^T
            q3 = q_finger[2]  # Third joint (index 2)
            q4 = q_finger[3]  # Fourth joint (index 3)
            
            delta_q = np.array([
                0.0,
                0.0,
                0.5 * (q3 - q4),
                0.5 * (q3 - q4)
            ])
        else:
            # For other finger types, no coupling (or custom logic)
            delta_q = np.zeros(self.n_joints_per_finger)
            
        return delta_q
    
    def _compute_joint_coupling_velocity(self, qd_finger: np.ndarray) -> np.ndarray:
        """
        Compute joint coupling velocity Δq̇ for damping term.
        
        Similar to Equation 33 but for velocities.
        
        Args:
            qd_finger: Joint velocities for one finger (4,)
            
        Returns:
            delta_qd: Coupling velocity (4,)
        """
        assert qd_finger.shape[0] == self.n_joints_per_finger
        
        if self.n_joints_per_finger == 4:
            qd3 = qd_finger[2]
            qd4 = qd_finger[3]
            
            delta_qd = np.array([
                0.0,
                0.0,
                0.5 * (qd3 - qd4),
                0.5 * (qd3 - qd4)
            ])
        else:
            delta_qd = np.zeros(self.n_joints_per_finger)
            
        return delta_qd
    
    def project_to_nullspace(
        self,
        tau_null: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        Project torque into nullspace of J^T (Equation 30).
        
        Projects τ_null into nullspace so it doesn't affect contact forces:
        τ_null_projected = (I - J^T(J^T)^+) τ_null
        
        Args:
            tau_null: Raw nullspace torque τ_null ∈ R^m
            jacobian: Hand Jacobian J ∈ R^(3n×m)
            
        Returns:
            tau_null_projected: Nullspace-projected torque ∈ R^m
        """
        # Compute nullspace projector: (I - J^T(J^T)^+)
        J_T = jacobian.T
        J_T_pinv = np.linalg.pinv(J_T)
        
        I = np.eye(self.n_joints_total)
        nullspace_projector = I - J_T @ J_T_pinv
        
        # Project torque into nullspace
        tau_null_projected = nullspace_projector @ tau_null
        
        return tau_null_projected
    
    def compute_final_torque(
        self,
        tau_contact: np.ndarray,
        tau_null: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        Compute final desired torque (Equation 30-31).
        
        τ_des = τ_c + (I - J^T(J^T)^+) τ_null
        
        Args:
            tau_contact: τ_c ∈ R^m - torques from contact forces (Eq. 29)
            tau_null: τ_null ∈ R^m - nullspace torque (Eq. 32)
            jacobian: J ∈ R^(3n×m) - hand Jacobian
            
        Returns:
            tau_des: τ_des ∈ R^m - final desired torque
        """
        # Project nullspace torque
        tau_null_projected = self.project_to_nullspace(tau_null, jacobian)
        
        # Combine (Equation 31)
        tau_des = tau_contact + tau_null_projected
        
        return tau_des
    
    def compute_full_pipeline(
        self,
        tau_contact: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        jacobian: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Full nullspace control pipeline.
        
        1. Compute nullspace torque (Eq. 32-33)
        2. Project to nullspace
        3. Add to contact torques (Eq. 30-31)
        
        Args:
            tau_contact: Contact torques τ_c ∈ R^m
            joint_positions: q ∈ R^m
            joint_velocities: q̇ ∈ R^m
            jacobian: J ∈ R^(3n×m)
            
        Returns:
            tau_des: Final desired torques
            info: Dictionary with intermediate values
        """
        # Step 1: Compute raw nullspace torque
        tau_null = self.compute_nullspace_torque(
            joint_positions,
            joint_velocities,
            jacobian
        )
        
        # Step 2: Project to nullspace
        tau_null_projected = self.project_to_nullspace(tau_null, jacobian)
        
        # Step 3: Combine
        tau_des = tau_contact + tau_null_projected
        
        # Prepare info
        info = {
            'tau_contact': tau_contact,
            'tau_null_raw': tau_null,
            'tau_null_projected': tau_null_projected,
            'tau_des': tau_des,
            'nullspace_magnitude': np.linalg.norm(tau_null_projected)
        }
        
        return tau_des, info
    
    def set_gains(self, K_null: float, D_null: float):
        """Update nullspace controller gains."""
        self.K_null = K_null
        self.D_null = D_null
        
    def get_gains(self) -> Tuple[float, float]:
        """Get current gains."""
        return self.K_null, self.D_null
    
    def analyze_joint_coupling(
        self,
        joint_positions: np.ndarray
    ) -> dict:
        """
        Analyze joint coupling for all fingers.
        
        Args:
            joint_positions: q ∈ R^m
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'fingers': []
        }
        
        for finger_idx in range(self.n_fingers):
            start_idx = finger_idx * self.n_joints_per_finger
            q_finger = joint_positions[start_idx:start_idx + self.n_joints_per_finger]
            
            if self.n_joints_per_finger == 4:
                q3 = q_finger[2]
                q4 = q_finger[3]
                coupling_error = q3 - q4
                
                finger_info = {
                    'finger_id': finger_idx,
                    'q3': q3,
                    'q4': q4,
                    'coupling_error': coupling_error,
                    'joints': q_finger
                }
            else:
                finger_info = {
                    'finger_id': finger_idx,
                    'joints': q_finger
                }
                
            analysis['fingers'].append(finger_info)
            
        return analysis