"""
Force Mapping - Section III.C
Combines impedance and internal forces, then maps to joint torques.

Key equations:
- f_c = f_x + f_int  (Equation 28) - Combine force components
- τ_c = J^T f_c      (Equation 29) - Map to joint torques
"""

import numpy as np
from typing import Tuple


class ForceMapper:
    """
    Maps contact forces to joint torques.
    
    Implements Section III.C:
    1. Combines impedance forces (f_x) and internal forces (f_int)
    2. Maps combined forces to joint torques using hand Jacobian
    """
    
    def __init__(self):
        """Initialize force mapper."""
        pass
    
    def combine_forces(
        self,
        impedance_forces: np.ndarray,
        internal_forces: np.ndarray
    ) -> np.ndarray:
        """
        Combine impedance and internal forces (Equation 28).
        
        f_c = f_x + f_int
        
        Args:
            impedance_forces: f_x ∈ R^(3n) - from impedance controller (Eq. 14)
            internal_forces: f_int ∈ R^(3n) - from QP optimizer (Eq. 17)
            
        Returns:
            contact_forces: f_c ∈ R^(3n) - combined contact forces
        """
        assert impedance_forces.shape == internal_forces.shape, \
            "Impedance and internal forces must have same dimensions"
        
        # Equation (28): f_c = f_x + f_int
        contact_forces = impedance_forces + internal_forces
        
        return contact_forces
    
    def map_to_joint_torques(
        self,
        contact_forces: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        Map contact forces to joint torques (Equation 29).
        
        τ_c = J^T f_c
        
        Args:
            contact_forces: f_c ∈ R^(3n) - combined contact forces
            jacobian: J ∈ R^(3n×m) - hand Jacobian
            
        Returns:
            joint_torques: τ_c ∈ R^m - actuator torques
        """
        # Validate dimensions
        n_contacts = contact_forces.shape[0] // 3
        expected_rows = 3 * n_contacts
        
        assert jacobian.shape[0] == expected_rows, \
            f"Jacobian rows ({jacobian.shape[0]}) must match contact force dimension ({expected_rows})"
        
        # Equation (29): τ_c = J^T f_c
        joint_torques = jacobian.T @ contact_forces
        
        return joint_torques
    
    def compute_full_pipeline(
        self,
        impedance_forces: np.ndarray,
        internal_forces: np.ndarray,
        jacobian: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full force mapping pipeline (Equations 28-29).
        
        1. Combine forces: f_c = f_x + f_int
        2. Map to torques: τ_c = J^T f_c
        
        Args:
            impedance_forces: f_x ∈ R^(3n)
            internal_forces: f_int ∈ R^(3n)
            jacobian: J ∈ R^(3n×m)
            
        Returns:
            contact_forces: f_c ∈ R^(3n)
            joint_torques: τ_c ∈ R^m
        """
        # Step 1: Combine forces (Eq. 28)
        contact_forces = self.combine_forces(impedance_forces, internal_forces)
        
        # Step 2: Map to joint torques (Eq. 29)
        joint_torques = self.map_to_joint_torques(contact_forces, jacobian)
        
        return contact_forces, joint_torques
    
    def decompose_torques(
        self,
        joint_torques: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        Inverse operation: map joint torques back to contact forces.
        
        f_c = (J^T)^+ τ_c
        
        Useful for analysis/debugging.
        
        Args:
            joint_torques: τ_c ∈ R^m
            jacobian: J ∈ R^(3n×m)
            
        Returns:
            contact_forces: f_c ∈ R^(3n)
        """
        # Pseudoinverse of J^T
        J_T = jacobian.T
        J_T_pinv = np.linalg.pinv(J_T)
        
        # Reconstruct forces
        contact_forces = J_T_pinv @ joint_torques
        
        return contact_forces
    
    @staticmethod
    def analyze_force_decomposition(
        impedance_forces: np.ndarray,
        internal_forces: np.ndarray,
        n_contacts: int = 2
    ) -> dict:
        """
        Analyze the contribution of each force component.
        
        Args:
            impedance_forces: f_x ∈ R^(3n)
            internal_forces: f_int ∈ R^(3n)
            n_contacts: Number of contacts
            
        Returns:
            Dictionary with force analysis per contact
        """
        analysis = {
            'total': impedance_forces + internal_forces,
            'impedance_magnitude': np.linalg.norm(impedance_forces),
            'internal_magnitude': np.linalg.norm(internal_forces),
            'contacts': []
        }
        
        for i in range(n_contacts):
            f_x_i = impedance_forces[3*i:3*i+3]
            f_int_i = internal_forces[3*i:3*i+3]
            f_total_i = f_x_i + f_int_i
            
            contact_info = {
                'contact_id': i,
                'impedance_force': f_x_i,
                'internal_force': f_int_i,
                'total_force': f_total_i,
                'impedance_norm': np.linalg.norm(f_x_i),
                'internal_norm': np.linalg.norm(f_int_i),
                'total_norm': np.linalg.norm(f_total_i)
            }
            
            analysis['contacts'].append(contact_info)
        
        return analysis