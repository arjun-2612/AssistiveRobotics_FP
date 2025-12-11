import numpy as np
from typing import Tuple, List
import mujoco
import mujoco.viewer
from grasp_model import GraspMatrix
import argparse

"""
TO RUN: 
python mujoco/allegro_hand/impedance_controller.py --model mujoco/allegro_hand/mjcf/scene.xml
"""

# ============================================================
# Math helpers
# ============================================================

def skew(v: np.ndarray) -> np.ndarray:
    """Return skew-symmetric matrix [v]_x for v in R^3."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to ZYX Euler angles: roll (x), pitch (y), yaw (z).

    q is [w, x, y, z] as in MuJoCo.
    """
    w, x, y, z = q

    # Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])

    # ZYX convention
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.array([roll, pitch, yaw])


# ============================================================
# Impedance Controller (Section III.A)
# ============================================================

class ImpedanceController:
    """
    Object-level impedance controller working in contact space.

    Pipeline:
    1. Compute desired contact motion from object error: Δc_x = G^T W^(-1) (x_d - x)
    2. Add grasp maintenance term: Δc_f = (I - G^T(G^T)^+)(c_0 - c)
    3. Combine: Δc = Δc_x + Δc_f
    4. Apply spring-damper: f_x = K_x Δc + D_x Δċ  (we use Δċ ≈ -c_dot)

    From Equations (10)-(14) in the paper.
    """

    def __init__(self, contact_stiffness: float = 3.0,
                 contact_damping: float = 0.1,
                 n_contacts: int = 2):
        """
        Initialize impedance controller.

        Args:
            contact_stiffness: K_x - Stiffness coefficient (N/m)
            contact_damping: D_x - Damping coefficient (N·s/m)
            n_contacts: Number of contact points (e.g., 2 for 2-finger grasp)
        """
        self.K_x = contact_stiffness
        self.D_x = contact_damping
        self.n_contacts = n_contacts
        self.grasp_matrix = GraspMatrix(n_contacts=4)

        # Store initial contact configuration (set during first call or externally)
        self.c_0: np.ndarray | None = None  # (3n,)

    def set_initial_contacts(self, contact_positions: np.ndarray):
        """
        Store the initial grasp configuration c_0.

        Args:
            contact_positions: Initial contact positions (3n vector)
                               [c1_x, c1_y, c1_z, c2_x, ...]
        """
        assert contact_positions.shape[0] == 3 * self.n_contacts, \
            f"Expected {3 * self.n_contacts} contact coordinates, got {contact_positions.shape[0]}"
        self.c_0 = contact_positions.copy()

    def compute_contact_forces(
        self,
        x_current: np.ndarray,
        x_desired: np.ndarray,
        contact_positions: np.ndarray,
        contact_velocities: np.ndarray,
        grasp_matrix: np.ndarray,
        W_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute desired contact forces using impedance control.

        Implements Equations (10)-(14) from the paper.

        Args:
            x_current: Current object pose (6D: [x, y, z, roll, pitch, yaw])
            x_desired: Desired object pose (6D)
            contact_positions: Current contact positions (3n vector)
            contact_velocities: Current contact velocities (3n vector)
            grasp_matrix: G matrix (6 x 3n)
            W_matrix: W matrix (6 x 6) - maps twist to pose changes

        Returns:
            f_x: Desired contact forces (3n vector)
        """
        # Validate inputs
        assert x_current.shape == (6,), "Current pose must be 6D"
        assert x_desired.shape == (6,), "Desired pose must be 6D"
        assert contact_positions.shape == (3 * self.n_contacts,), "contact_positions must be 3n"
        assert contact_velocities.shape == (3 * self.n_contacts,), "contact_velocities must be 3n"
        assert grasp_matrix.shape[0] == 6, "Grasp matrix must have 6 rows"
        assert grasp_matrix.shape[1] == 3 * self.n_contacts, \
            f"Grasp matrix must have {3 * self.n_contacts} columns"
        assert W_matrix.shape == (6, 6), "W matrix must be 6x6"

        # Initialize c_0 if not set
        if self.c_0 is None:
            self.c_0 = contact_positions.copy()

        # Step 1: Compute object pose error
        pose_error = self._compute_pose_error(x_current, x_desired)

        # Step 2: Compute desired contact motion for object control (Eq. 11)
        # Δc_x = G^T W^(-1) (x_d - x)
        W_inv = np.linalg.inv(W_matrix)
        delta_c_x = grasp_matrix.T @ W_inv @ pose_error

        # Step 3: Compute grasp maintenance term (Eq. 12)
        # Δc_f = (I - G^T(G^T)^+)(c_0 - c)
        delta_c_f = self._compute_grasp_maintenance_term(contact_positions, grasp_matrix)

        # Step 4: Combine desired contact motions (Eq. 13)
        # Δc = Δc_x + Δc_f
        delta_c = delta_c_x + delta_c_f

        # Step 5: Apply spring-damper law (Eq. 14)
        # f_x = K_x Δc + D_x Δċ
        # If desired contact velocity is ~0, then Δċ = -c_dot:
        delta_c_dot = -contact_velocities
        f_x = self.K_x * delta_c + self.D_x * delta_c_dot

        return f_x

    def _compute_grasp_maintenance_term(self, contact_positions: np.ndarray,
                                        grasp_matrix: np.ndarray) -> np.ndarray:
        """
        Compute grasp maintenance term (Equation 12).

        Δc_f = (I - G^T(G^T)^+)(c_0 - c)

        This projects the difference between initial and current contact positions
        into the nullspace of the grasp matrix, so it doesn't affect object motion.
        """
        n_dim = 3 * self.n_contacts

        # Contact position error relative to initial grasp
        c_error = self.c_0 - contact_positions

        # Compute Moore-Penrose pseudoinverse of G^T
        G = grasp_matrix
        G_pinv = np.linalg.pinv(G)

        # Nullspace projector: (I - G^T(G^T)^+)
        I = np.eye(n_dim)
        nullspace_projector = I - G_pinv @ G

        # Project contact error into nullspace
        delta_c_f = nullspace_projector @ c_error

        return delta_c_f

    def _compute_pose_error(self, x_current: np.ndarray, x_desired: np.ndarray) -> np.ndarray:
        """
        Compute 6D pose error (x_d - x).

        Args:
            x_current: Current pose [x, y, z, roll, pitch, yaw]
            x_desired: Desired pose [x, y, z, roll, pitch, yaw]

        Returns:
            6D error vector
        """
        # Position error
        position_error = x_desired[:3] - x_current[:3]

        # Orientation error (wrap angles to [-π, π])
        orientation_error = self._angle_difference(x_desired[3:], x_current[3:])

        return np.concatenate([position_error, orientation_error])

    @staticmethod
    def _angle_difference(angle_desired: np.ndarray, angle_current: np.ndarray) -> np.ndarray:
        """Compute angular difference, wrapping to [-π, π]."""
        diff = angle_desired - angle_current
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return diff

    @staticmethod
    def compute_W_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        Compute W matrix that maps object twist to position/Euler angle changes.

        W ∈ R^(6x6) maps twist (linear + angular velocity) to changes in
        position + Euler angles (ZYX convention).

        Args:
            euler_angles: Current Euler angles [roll, pitch, yaw] in radians

        Returns:
            W: 6x6 matrix
        """
        roll, pitch, yaw = euler_angles

        c_r = np.cos(roll)
        s_r = np.sin(roll)
        c_p = np.cos(pitch)
        s_p = np.sin(pitch)
        t_p = np.tan(pitch)

        # Euler angle Jacobian E (body angular velocity -> Euler rates)
        E = np.array([
            [1.0, s_r * t_p, c_r * t_p],
            [0.0, c_r,      -s_r],
            [0.0, s_r / c_p, c_r / c_p]
        ])

        # Full W matrix
        W = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), E]
        ])

        return W

    def set_stiffness(self, stiffness: float):
        """Update contact stiffness K_x."""
        self.K_x = stiffness

    def set_damping(self, damping: float):
        """Update contact damping D_x."""
        self.D_x = damping

    def get_parameters(self) -> Tuple[float, float]:
        """Get current controller parameters (K_x, D_x)."""
        return self.K_x, self.D_x


# ============================================================
# MuJoCo helpers
# ============================================================

def get_object_pose(model: mujoco.MjModel,
                    data: mujoco.MjData,
                    body_name: str) -> np.ndarray:
    """
    Get object pose [x, y, z, roll, pitch, yaw] in world frame.
    """
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()      # (3,)
    quat = data.xquat[body_id].copy()    # (4,)
    euler = quat_to_euler_zyx(quat)      # (3,)
    return np.concatenate([pos, euler])


def get_contact_positions_velocities_from_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use fingertip bodies as 'contact points'.

    Returns:
        contact_positions: (3n,) world positions of each body
        contact_velocities: (3n,) world linear velocities of each body
    """
    positions = []
    velocities = []

    for name in body_names:
        body_id = model.body(name).id
        pos = data.xpos[body_id].copy()          # (3,)
        spatial_vel = data.cvel[body_id].copy()  # [wx, wy, wz, vx, vy, vz]
        linvel = spatial_vel[3:]                 # (3,)

        positions.append(pos)
        velocities.append(linvel)

    positions = np.concatenate(positions, axis=0)
    velocities = np.concatenate(velocities, axis=0)
    return positions, velocities


def get_hand_dof_indices(model: mujoco.MjModel,
                         joint_names: List[str]) -> np.ndarray:
    """
    Get DOF indices for a list of 1-DOF joints (typical for Allegro).
    """
    indices = []
    for name in joint_names:
        j_id = model.joint(name).id
        dof_adr = model.jnt_dofadr[j_id]
        # assuming 1 dof per joint
        indices.append(dof_adr)
    return np.array(indices, dtype=int)


def build_contact_jacobian_from_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: List[str],
    hand_dof_indices: np.ndarray,
) -> np.ndarray:
    """
    Build stacked contact Jacobian J (3n x m) for given fingertip bodies.

    For each body, we compute the linear Jacobian Jp (3 x nv) at the body origin
    and then restrict it to the hand DOFs.
    """
    nv = model.nv
    n_contacts = len(body_names)
    J_full = np.zeros((3 * n_contacts, nv))

    for i, name in enumerate(body_names):
        body_id = model.body(name).id
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
        J_full[3 * i:3 * i + 3, :] = Jp

    J_hand = J_full[:, hand_dof_indices]
    return J_hand



# ============================================================
# Example main control loop
# ============================================================

# This is the free object defined in scene.xml (after you uncomment it)
OBJECT_BODY_NAME = "object"

# We’ll treat the finger-tip bodies as contact points
CONTACT_BODY_NAMES = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]

# Hand joint names from left_hand.xml
HAND_JOINT_NAMES = [
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
]

def main():
    print("Starting impedance control simulation...")

    parser = argparse.ArgumentParser(description="MuJoCo Allegro Hand Impedance Control")
    parser.add_argument("--model", type=str, default="mujoco/allegro_hand/mjcf/scene.xml",
                        help="Path to MuJoCo model XML file")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hand_dof_indices = get_hand_dof_indices(model, HAND_JOINT_NAMES)

    n_contacts = len(CONTACT_BODY_NAMES)
    controller = ImpedanceController(
        contact_stiffness=3.0,
        contact_damping=0.1,
        n_contacts=n_contacts,
    )

    x_init = get_object_pose(model, data, OBJECT_BODY_NAME)
    x_target = x_init.copy()
    x_target[2] += 0.05
    x_target[5] += np.deg2rad(20.0)
    x_desired = x_init.copy()

    dt = model.opt.timestep
    sim_time = 0.0
    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():  # <-- Only this condition
            if sim_time < 2.0:
                alpha = sim_time / 2.0
                x_desired = x_init + alpha * (x_target - x_init)
            else:
                x_desired = x_target

            x_current = get_object_pose(model, data, OBJECT_BODY_NAME)
            contact_pos, contact_vel = get_contact_positions_velocities_from_bodies(
                model, data, CONTACT_BODY_NAMES
            )

            if controller.c_0 is None:
                controller.set_initial_contacts(contact_pos)

            euler = x_current[3:]
            W = controller.compute_W_matrix(euler)
            object_pos = x_current[:3]
            G = controller.grasp_matrix.compute_grasp_matrix(contact_pos, object_pos)

            f_x = controller.compute_contact_forces(
                x_current=x_current,
                x_desired=x_desired,
                contact_positions=contact_pos,
                contact_velocities=contact_vel,
                grasp_matrix=G,
                W_matrix=W,
            )

            J = build_contact_jacobian_from_bodies(
                model, data, CONTACT_BODY_NAMES, hand_dof_indices
            )
            tau_hand = J.T @ f_x

            # if step_count % 50 == 0:
            #     print(
            #         f"t = {sim_time:.3f} s | "
            #         f"f = {np.linalg.norm(f_x.reshape(4, 3), axis=1)} | "
            #         f"||tau|| = {np.linalg.norm(tau_hand):.6f}"
            #     )

            data.qfrc_applied[hand_dof_indices] = tau_hand

            mujoco.mj_step(model, data)
            sim_time += dt
            step_count += 1

            viewer.sync()

    print("Simulation finished.")


if __name__ == "__main__":
    main()