#!/usr/bin/env python3
"""
MuJoCo script that:
  • Loads an MJCF (.xml) model
  • Picks a random target position for each actuated hinge/slide joint (within limits)
  • Drives joints to those targets with explicit PD torque control (motor actuators)
  • Resamples random targets periodically
  • Displays the interactive viewer

Usage:
    python mujoco_pd_random_targets.py --model path/to/model.xml \
        --kp 60 --kd 2 --seed 0 --simtime 15 --resample 3.0

Notes:
- This assumes your MJCF already contains actuators that target joints (typical <motor/position/velocity> with joint transmission).
- We compute explicit PD torques and send them via `data.ctrl` (works best if your actuators are torque/motor-like). If your actuators are <position>, you can switch to setting `data.ctrl[:] = qpos_targets_in_joint_space` instead of torques.
"""

import argparse
import time
import sys
from typing import List, Tuple

import numpy as np

try:
    import mujoco as mj
    import mujoco.viewer as mjv
except Exception as e:
    print("[ERROR] Failed to import mujoco. Install mujoco>=3.x (pip install mujoco).\n", e)
    sys.exit(1)


def get_actuated_joint_dofs(model: mj.MjModel) -> List[Tuple[int, int]]:
    """Return list of (actuator_id, dof_address) for actuators that drive a single hinge/slide joint.

    We look for transmission type == JOINT. For such actuators, `actuator_trnid[i,0]` holds the joint id.
    Then `jnt_dofadr[j]` maps to the corresponding DOF address in qpos/qvel.
    """
    pairs: List[Tuple[int, int]] = []
    for i in range(model.nu):
        trn_type = model.actuator_trntype[i]
        if trn_type != mj.mjtTrn.mjTRN_JOINT:
            continue
        j = model.actuator_trnid[i][0]
        if j < 0:
            continue
        jtype = model.jnt_type[j]
        if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        dof_adr = model.jnt_dofadr[j]
        pairs.append((i, dof_adr))
    return pairs


def sample_joint_targets(model: mj.MjModel, pairs: List[Tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    """Sample a qpos target vector for the full model, using joint limits where available.

    For each actuated single-DOF joint, sample uniformly in `jnt_range`. If range is (0,0), fall back to defaults.
    Returns a full-length qpos target (size model.nq) that can be compared to data.qpos.
    """
    qpos_target = np.copy(model.qpos0)
    for _, dof_adr in pairs:
        # map dof back to joint index
        # The joint index that owns this dof has jnt_dofadr[j] == dof_adr
        j_idx = int(np.where(model.jnt_dofadr == dof_adr)[0][0])
        jtype = model.jnt_type[j_idx]
        lo, hi = model.jnt_range[j_idx]
        if lo == 0.0 and hi == 0.0:
            if jtype == mj.mjtJoint.mjJNT_HINGE:
                lo, hi = -np.pi, np.pi
            else:  # SLIDE
                lo, hi = -0.1, 0.1
        qpos_target[dof_adr] = rng.uniform(lo, hi)
    return qpos_target


def apply_pd_torques(model: mj.MjModel, data: mj.MjData, pairs: List[Tuple[int, int]],
                     qpos_target: np.ndarray, kp: float, kd: float):
    """Compute PD torques in joint space and write them into data.ctrl for each actuator.

    torque = kp*(qpos_target - qpos) - kd*qvel
    """
    for (act_id, dof_adr) in pairs:
        q = data.qpos[dof_adr]
        qd = data.qvel[dof_adr]
        tau = kp*(qpos_target[dof_adr] - q) - kd*qd
        data.ctrl[act_id] = tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to MJCF model (.xml)')
    ap.add_argument('--kp', type=float, default=60.0, help='PD proportional gain')
    ap.add_argument('--kd', type=float, default=2.0, help='PD derivative gain')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--simtime', type=float, default=15.0, help='Total simulation time (s)')
    ap.add_argument('--resample', type=float, default=3.0, help='Seconds between new random targets')
    ap.add_argument('--timestep', type=float, default=0.0, help='Override model timestep if > 0')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    model = mj.MjModel.from_xml_path(args.model)
    if args.timestep > 0:
        model.opt.timestep = float(args.timestep)
    data = mj.MjData(model)

    pairs = get_actuated_joint_dofs(model)
    if not pairs:
        print('[WARN] No joint-driven actuators found. Ensure your MJCF defines <actuator> with joint transmissions.')

    # Initial target sample
    qpos_target = sample_joint_targets(model, pairs, rng)

    with mjv.launch_passive(model, data) as viewer:
        t0 = time.time()
        next_ui_sync = t0
        next_resample = t0 + args.resample
        while viewer.is_running() and (time.time() - t0) < args.simtime:
            now = time.time()

            if now >= next_resample:
                qpos_target = sample_joint_targets(model, pairs, rng)
                next_resample = now + args.resample

            # PD control
            # apply_pd_torques(model, data, pairs, qpos_target, args.kp, args.kd)

            # Step simulation
            mj.mj_step(model, data)

            # Sync viewer ~60 Hz
            if now >= next_ui_sync:
                viewer.sync()
                next_ui_sync = now + 1.0/60.0

    print('[INFO] Done.')


if __name__ == '__main__':
    main()
