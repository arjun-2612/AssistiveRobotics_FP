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
    min ||f_* - f_d||²
    subject to friction cone constraints on f_int (Equations 21-27)
    """
    
    def __init__(self, n_contacts: int = 4, friction_coefficient: float = 0.8, f_min: float = 0.1, f_max: float = 50.0):
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
        
    def compute_contact_forces(self, desired_wrench: np.ndarray, grasp_matrix: np.ndarray, 
                               desired_normal_forces: Optional[np.ndarray] = None,
                               contact_normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Compute optimal contact forces using QP.
        
        Solves Equation (18) in the paper:
        min ||f_* - f_d||²
        subject to friction cone constraints on f_int
        
        Then returns f_int using Equation (19):
        f_int = (G^T)^+ w_dyn + (I - G^T(G^T)^+) f_*
        
        Args:
            desired_wrench: w ∈ R^6 - desired object wrench
            grasp_matrix: G ∈ R^(6×3n) - grasp matrix
            desired_normal_forces: f_d^[i] ∈ R^n - desired normal forces (optional)
            contact_normals: n^[i] ∈ R^(3×n) - contact normal vectors (optional)
            
        Returns:
            f_int: Optimal contact forces (3n vector)
            info: Dictionary with optimization info
        """
        # w_dyn = desired_wrench (Equation 15, simplified)
        w_dyn = desired_wrench
        
        # Compute f_d: desired internal forces (Equation 16)
        if desired_normal_forces is not None and contact_normals is not None:
            f_d = self._construct_desired_forces(desired_normal_forces, contact_normals)
        else:
            # Default: small squeeze force in normal direction
            f_d = np.zeros(3 * self.n_contacts)
            if contact_normals is not None:
                for i in range(self.n_contacts):
                    f_d[3*i:3*i+3] = contact_normals[i] * 5.0  # 5N default
        
        # Solve QP for optimal f_* (Equation 18)
        G_T = grasp_matrix.T
        f_star, qp_info = self._solve_qp(f_d, G_T, w_dyn, contact_normals)
        
        # Compute final forces using Equation (19)
        # f_int = (G^T)^+ w_dyn + (I - G^T(G^T)^+) f_*
        G_T_pinv = np.linalg.pinv(G_T)
        f_wrench = G_T_pinv @ w_dyn
        
        n_dim = 3 * self.n_contacts
        I = np.eye(n_dim)
        nullspace_proj = I - G_T_pinv @ G_T
        
        f_int = f_wrench + nullspace_proj @ f_star
        
        # Prepare info
        info = {
            'f_wrench': f_wrench,                    # Wrench-generating component
            'f_internal': nullspace_proj @ f_star,   # Internal force component
            'f_star': f_star,                        # Optimized f_*
            'qp_status': qp_info['status'],
            'qp_optimal': qp_info['optimal']
        }
        
        return f_int, info
    
    def _solve_qp(self, f_d: np.ndarray, G_T: np.ndarray, w_dyn: np.ndarray, 
                contact_normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Solve the QP problem (Equations 18, 24-27).
        
        Strategy: Since constraints are on f_int = f_wrench + N @ f_*, 
        we can reformulate as constraints directly on f_*.
        """
        n_dim = 3 * self.n_contacts
        
        # Precompute components for Equation (19)
        G_T_pinv = np.linalg.pinv(G_T)
        I = np.eye(n_dim)
        N = I - G_T_pinv @ G_T  # Nullspace projection
        f_wrench = G_T_pinv @ w_dyn
        
        # Decision variable: f_* ∈ R^(3n)
        f_star = cp.Variable(n_dim)
        
        # Objective: minimize ||f_* - f_d||² (Equation 18)
        objective = cp.Minimize(cp.sum_squares(f_star - f_d))
        
        # Constraints will be on each contact's force
        constraints = []
        
        for i in range(self.n_contacts):
            # Extract indices for contact i
            idx = slice(3*i, 3*(i+1))
            
            # Compute f_int for contact i: f_int_i = f_wrench_i + N_i @ f_*
            # This is affine in f_*, so CVXPY can handle it
            f_wrench_i = f_wrench[idx]
            N_i = N[idx, :]
            f_int_i = f_wrench_i + N_i @ f_star
            
            if contact_normals is not None:
                n_i = contact_normals[i]  # Normal vector
                
                # Build orthogonal tangent basis
                if abs(n_i[2]) < 0.9:
                    t1 = np.array([n_i[1], -n_i[0], 0.0])
                else:
                    t1 = np.array([0.0, n_i[2], -n_i[1]])
                t1 = t1 / np.linalg.norm(t1)
                t2 = np.cross(n_i, t1)
                t2 = t2 / np.linalg.norm(t2)
                
                # Force components (these are scalar expressions in f_*)
                f_n = n_i @ f_int_i   # Normal force
                f_t1 = t1 @ f_int_i   # Tangential 1
                f_t2 = t2 @ f_int_i   # Tangential 2
                
                # Friction pyramid constraints (4 linear inequalities)
                # Inscribe pyramid in friction cone
                mu_scaled = self.mu / np.sqrt(2)
                
                constraints.append(f_t1 <= mu_scaled * f_n)
                constraints.append(-f_t1 <= mu_scaled * f_n)
                constraints.append(f_t2 <= mu_scaled * f_n)
                constraints.append(-f_t2 <= mu_scaled * f_n)
                
                # Normal force bounds
                constraints.append(f_n >= self.f_min)
                constraints.append(f_n <= self.f_max)
            else:
                # Fallback: assume z is normal direction
                f_n = f_int_i[2]
                f_t1 = f_int_i[0]
                f_t2 = f_int_i[1]
                
                mu_scaled = self.mu / np.sqrt(2)
                constraints.append(f_t1 <= mu_scaled * f_n)
                constraints.append(-f_t1 <= mu_scaled * f_n)
                constraints.append(f_t2 <= mu_scaled * f_n)
                constraints.append(-f_t2 <= mu_scaled * f_n)
                constraints.append(f_n >= self.f_min)
                constraints.append(f_n <= self.f_max)
        
        # Solve QP
        problem = cp.Problem(objective, constraints)
        
        try:
            # Try OSQP first (fast for QP)
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                f_star_opt = f_star.value
                if f_star_opt is None:
                    f_star_opt = f_d  # Fallback to desired
                optimal = True
            else:
                print(f"⚠ QP solver status: {problem.status}")
                # Try fallback: use desired forces
                f_star_opt = f_d
                optimal = False
                
        except Exception as e:
            print(f"❌ QP solver failed: {e}")
            # Fallback: use desired internal forces directly
            f_star_opt = f_d
            optimal = False
        
        info = {
            'status': problem.status if 'problem' in locals() else 'error',
            'optimal': optimal,
            'objective_value': problem.value if optimal else None
        }
        
        return f_star_opt, info
    
    def _construct_desired_forces(self, normal_forces: np.ndarray, contact_normals: np.ndarray) -> np.ndarray:
        """
        Construct f_d from desired normal forces (Equation 16).
        
        f_d^[i] = f_d^[i] n^[i]
        
        Args:
            normal_forces: Desired normal force magnitudes (n,)
            contact_normals: Contact normal directions (n × 3)
            
        Returns:
            f_d: Full force vector (3n,)
        """
        assert normal_forces.shape[0] == self.n_contacts
        assert contact_normals.shape == (self.n_contacts, 3)
        
        f_d = np.zeros(3 * self.n_contacts)
        
        for i in range(self.n_contacts):
            # f_d^[i] = f_d^[i] * n^[i]
            f_d[3*i:3*i+3] = normal_forces[i] * contact_normals[i]
            
        return f_d
    
    def compute_contact_forces_with_activation(self, desired_wrench: np.ndarray, grasp_matrix: np.ndarray,
                                               desired_normal_forces: np.ndarray, contact_normals: np.ndarray, 
                                               activation_values: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute contact forces with activation scaling (Equation 35).
        
        Args:
            desired_wrench: w ∈ R^6
            grasp_matrix: G ∈ R^(6×3n)
            desired_normal_forces: f_d^[i] ∈ R^n (before activation)
            contact_normals: n^[i] ∈ R^(3×n)
            activation_values: a^[i] ∈ R^n (from ContactManager)
            
        Returns:
            f_int: Contact forces
            info: Optimization info
        """
        # Scale desired forces by activation
        scaled_forces = desired_normal_forces * activation_values
        
        return self.compute_contact_forces(
            desired_wrench,
            grasp_matrix,
            desired_normal_forces=scaled_forces,
            contact_normals=contact_normals
        )
    
    def compute_dynamic_wrench(self, object_state: dict, desired_acceleration: np.ndarray,
                              mass_matrix: np.ndarray) -> np.ndarray:
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
        w_dyn = mass_matrix @ desired_acceleration
        
        # TODO: Add gravity and Coriolis terms
        # g_o = [0, 0, -m*g, 0, 0, 0]
        # C_o = ...
        
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
        
    def compute_contact_forces(self, desired_wrench: np.ndarray, grasp_matrix: np.ndarray,
                               internal_forces: Optional[np.ndarray] = None) -> np.ndarray:
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