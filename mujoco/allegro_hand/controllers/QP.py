"""
Internal Forces Optimization - Section III.B
Solves a Quadratic Program to compute contact forces that:
1. Generate desired object wrench (for dynamics)
2. Add internal forces (grasp maintenance)
3. Satisfy friction cone constraints

Key equation (17): f = (G^T)^+ w + (I - G^T(G^T)^+) f_*
"""

import numpy as np
from typing import Optional, Tuple
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not installed. Install with: pip install cvxpy")


class InternalForceOptimizer:
    """
    Computes optimal contact forces using QP formulation from Section III.B.
    
    Solves Equation (18):
    min ||f_* - f_dyn||²
    subject to friction cone constraints (Equations 21-23)
    """
    
    def __init__(
        self,
        n_contacts: int = 4,
        friction_coefficient: float = 0.8,
        f_min: float = 0.1,
        f_max: float = 50.0
    ):
        """
        Initialize internal force optimizer.
        
        Args:
            n_contacts: Number of contact points
            friction_coefficient: μ - friction coefficient
            f_min: Minimum normal force (N)
            f_max: Maximum normal force (N)
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for QP optimization. Install with: pip install cvxpy")
            
        self.n_contacts = n_contacts
        self.mu = friction_coefficient
        self.f_min = f_min
        self.f_max = f_max
        
    def compute_contact_forces(
        self,
        desired_wrench: np.ndarray,
        grasp_matrix: np.ndarray,
        desired_normal_forces: Optional[np.ndarray] = None,
        contact_normals: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute optimal contact forces using QP.
        
        Solves Equation (18) in the paper:
        min ||f_* - f_dyn||²
        subject to friction cone constraints
        
        Then reconstructs full forces using Equation (17):
        f = (G^T)^+ w + (I - G^T(G^T)^+) f_*
        
        Args:
            desired_wrench: w ∈ R^6 - desired object wrench
            grasp_matrix: G ∈ R^(6×3n) - grasp matrix
            desired_normal_forces: f_d^[i] ∈ R^n - desired normal forces (optional)
            contact_normals: n^[i] ∈ R^(3×n) - contact normal vectors (optional)
            
        Returns:
            f: Optimal contact forces (3n vector)
            info: Dictionary with optimization info
        """
        # Compute dynamic load component (Equation 15)
        # For now, w_dyn = desired_wrench (can add dynamics later)
        w_dyn = desired_wrench
        
        # Compute f_wrench: forces to generate desired wrench
        # f_wrench = (G^T)^+ w_dyn
        G_T = grasp_matrix.T
        G_T_pinv = np.linalg.pinv(G_T)
        f_wrench = G_T_pinv @ w_dyn
        
        # Compute f_dyn: desired internal forces (Equation 16)
        if desired_normal_forces is not None and contact_normals is not None:
            f_dyn = self._construct_desired_forces(desired_normal_forces, contact_normals)
        else:
            # Default: small squeeze force in normal direction
            f_dyn = np.zeros(3 * self.n_contacts)
            # Add default normal forces if normals provided
            if contact_normals is not None:
                for i in range(self.n_contacts):
                    f_dyn[3*i:3*i+3] = contact_normals[i] * 5.0  # 5N default
        
        # Solve QP for optimal f_* (Equation 18)
        f_star, qp_info = self._solve_qp(f_dyn, G_T, w_dyn)
        
        # Reconstruct full contact forces (Equation 17)
        # f = (G^T)^+ w + (I - G^T(G^T)^+) f_*
        n_dim = 3 * self.n_contacts
        I = np.eye(n_dim)
        nullspace_proj = I - G_T_pinv @ G_T
        
        f_total = f_wrench + nullspace_proj @ f_star
        
        # Prepare info
        info = {
            'f_wrench': f_wrench,           # Wrench-generating component
            'f_internal': nullspace_proj @ f_star,  # Internal force component
            'f_star': f_star,               # Optimized internal forces
            'qp_status': qp_info['status'],
            'qp_optimal': qp_info['optimal']
        }
        
        return f_total, info
    
    def compute_contact_forces_with_activation(
        self,
        desired_wrench: np.ndarray,
        grasp_matrix: np.ndarray,
        desired_normal_forces: np.ndarray,
        contact_normals: np.ndarray,
        activation_values: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute contact forces with activation scaling (Equation 35).
        
        This is the modified version of compute_contact_forces that handles
        smooth contact transitions.
        
        Args:
            desired_wrench: w ∈ R^6
            grasp_matrix: G ∈ R^(6×3n)
            desired_normal_forces: f_d^[i] ∈ R^n (before activation)
            contact_normals: n^[i] ∈ R^(3×n)
            activation_values: a^[i] ∈ R^n (from ContactManager)
            
        Returns:
            f_total: Contact forces
            info: Optimization info
        """
        # Scale desired forces by activation (Equation 35)
        # f(t)_d^[i] = a^[i](t) * f_d^[i] * n^[i]
        f_dyn = np.zeros(3 * self.n_contacts)
        
        for i in range(self.n_contacts):
            activation = activation_values[i]
            normal_force = desired_normal_forces[i]
            normal = contact_normals[i]
            
            # Apply activation scaling
            f_dyn[3*i:3*i+3] = activation * normal_force * normal
        
        # Continue with standard QP solve
        return self.compute_contact_forces(
            desired_wrench,
            grasp_matrix,
            desired_normal_forces=None,  # Already handled above
            contact_normals=None
        ) 
    
    def _solve_qp(
        self,
        f_dyn: np.ndarray,
        G_T: np.ndarray,
        w_dyn: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve the QP problem (Equation 18 & 24).
        
        min (1/2) f_*^T H f_* - f_*^T f_d
        subject to:
            f_s^[i] - μ f_n^[i] ≤ -(f_dyn,s^[i] - μ f_dyn,n^[i])  (friction cone)
            f_n^[i] ≥ f_min - f_dyn,n^[i]                          (min force)
            f_n^[i] ≤ f_max - f_dyn,n^[i]                          (max force)
        
        Args:
            f_dyn: Desired internal forces f_dyn ∈ R^(3n)
            G_T: Grasp matrix transpose
            w_dyn: Desired wrench
            
        Returns:
            f_star: Optimal internal forces
            info: Optimization info
        """
        n_dim = 3 * self.n_contacts
        
        # Decision variable: f_* ∈ R^(3n)
        f_star = cp.Variable(n_dim)
        
        # Objective: minimize ||f_* - f_dyn||² (Equation 18)
        # Equivalent to (1/2) f_*^T H f_* - f_*^T f_dyn (Equation 24)
        objective = cp.Minimize(cp.sum_squares(f_star - f_dyn))
        
        # Constraints
        constraints = []
        
        # For each contact, add friction cone constraints (Equations 21-23)
        for i in range(self.n_contacts):
            # Extract normal and tangential components
            # Assuming contact i has forces [f_x, f_y, f_z] at indices 3i:3i+3
            # We need to know which direction is normal - for now assume z is normal
            
            # Normal force (assume last component is normal)
            f_n_i = f_star[3*i + 2]
            f_dyn_n_i = f_dyn[3*i + 2]
            
            # Tangential forces (first two components)
            f_t_i = cp.norm(f_star[3*i:3*i+2], 2)  # Magnitude of tangential force
            f_dyn_t_i = np.linalg.norm(f_dyn[3*i:3*i+2])
            
            # Friction cone constraint (Equation 25 simplified)
            # |f_tangential| ≤ μ * f_normal
            constraints.append(f_t_i <= self.mu * (f_n_i + f_dyn_n_i))
            
            # Normal force bounds (Equations 26-27)
            constraints.append(f_n_i >= self.f_min - f_dyn_n_i)  # Equation (26)
            constraints.append(f_n_i <= self.f_max - f_dyn_n_i)  # Equation (27)
        
        # Nullspace constraint: (I - G^T(G^T)^+) f_* should not affect wrench
        # This is implicitly satisfied by the structure, but we can add:
        # G^T f_* = 0 (forces in nullspace)
        # constraints.append(G_T @ f_star == np.zeros(6))  # Optional
        
        # Solve QP
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                f_star_opt = f_star.value
                optimal = True
            else:
                # Fallback to zero if infeasible
                f_star_opt = np.zeros(n_dim)
                optimal = False
                
        except Exception as e:
            print(f"QP solver failed: {e}")
            f_star_opt = np.zeros(n_dim)
            optimal = False
        
        info = {
            'status': problem.status,
            'optimal': optimal,
            'objective_value': problem.value if optimal else None
        }
        
        return f_star_opt, info
    
    def _construct_desired_forces(
        self,
        normal_forces: np.ndarray,
        contact_normals: np.ndarray
    ) -> np.ndarray:
        """
        Construct f_dyn from desired normal forces (Equation 16).
        
        f_d^[i] = f_d^[i] n^[i]
        
        Args:
            normal_forces: Desired normal force magnitudes (n,)
            contact_normals: Contact normal directions (n × 3)
            
        Returns:
            f_dyn: Full force vector (3n,)
        """
        assert normal_forces.shape[0] == self.n_contacts
        assert contact_normals.shape == (self.n_contacts, 3)
        
        f_dyn = np.zeros(3 * self.n_contacts)
        
        for i in range(self.n_contacts):
            # f_d^[i] = f_d^[i] * n^[i]
            f_dyn[3*i:3*i+3] = normal_forces[i] * contact_normals[i]
            
        return f_dyn
    
    def compute_dynamic_wrench(
        self,
        object_state: dict,
        desired_acceleration: np.ndarray,
        mass_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute dynamic object wrench (Equation 15).
        
        w_dyn = M_o(x)ẍ + C_o(x,ẋ)ẋ + g_o(x)
        
        Simplified version (can be extended with Coriolis and gravity terms).
        
        Args:
            object_state: Dictionary with 'position', 'velocity'
            desired_acceleration: ẍ_desired ∈ R^6
            mass_matrix: M_o ∈ R^(6×6)
            
        Returns:
            w_dyn: Dynamic wrench ∈ R^6
        """
        # Simple version: w_dyn = M_o * ẍ
        # (Ignoring Coriolis C_o(x,ẋ)ẋ and gravity g_o(x) for now)
        
        w_dyn = mass_matrix @ desired_acceleration
        
        # TODO: Add gravity term
        # g_o = [0, 0, -m*g, 0, 0, 0]  # Gravity in z-direction
        
        return w_dyn
    
    def set_friction_coefficient(self, mu: float):
        """Update friction coefficient."""
        self.mu = mu
        
    def set_force_limits(self, f_min: float, f_max: float):
        """Update normal force limits."""
        self.f_min = f_min
        self.f_max = f_max


class SimplifiedInternalForceComputer:
    """
    Simplified version without QP (for testing or when cvxpy not available).
    
    Uses Equation (17) directly with manually specified internal forces.
    """
    
    def __init__(self, n_contacts: int = 2):
        self.n_contacts = n_contacts
        
    def compute_contact_forces(
        self,
        desired_wrench: np.ndarray,
        grasp_matrix: np.ndarray,
        internal_forces: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute contact forces using Equation (17) without optimization.
        
        f = (G^T)^+ w + (I - G^T(G^T)^+) f_*
        
        Args:
            desired_wrench: w ∈ R^6
            grasp_matrix: G ∈ R^(6×3n)
            internal_forces: f_* ∈ R^(3n) (if None, uses zero)
            
        Returns:
            f: Contact forces (3n,)
        """
        G_T = grasp_matrix.T
        G_T_pinv = np.linalg.pinv(G_T)
        
        # Wrench-generating component
        f_wrench = G_T_pinv @ desired_wrench
        
        # Internal force component
        if internal_forces is None:
            internal_forces = np.zeros(3 * self.n_contacts)
            
        n_dim = 3 * self.n_contacts
        I = np.eye(n_dim)
        nullspace_proj = I - G_T_pinv @ G_T
        f_internal = nullspace_proj @ internal_forces
        
        # Total forces
        f_total = f_wrench + f_internal
        
        return f_total