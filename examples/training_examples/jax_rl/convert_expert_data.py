"""
Convert HDF5 expert trajectories to Trajectory format for AMP training.

This converts the HDF5 file from generate_wildrobot_dataset.py into a
loco-mujoco Trajectory object that can be loaded by experiment.py.

Usage:
    python convert_expert_data.py \
        --input wildrobot_expert_motions/wildrobot_expert_dataset.h5 \
        --output wildrobot_expert_traj.npz
"""
import argparse
import h5py
import numpy as np
import mujoco
from loco_mujoco import RLFactory
from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel


def compute_forward_kinematics(env, qpos_all, qvel_all):
    """
    Compute forward kinematics (xpos, xquat, etc.) from qpos/qvel using MuJoCo.

    Args:
        env: Environment instance
        qpos_all: Array of qpos data [num_timesteps, nq]
        qvel_all: Array of qvel data [num_timesteps, nv]

    Returns:
        Dictionary with xpos, xquat, cvel, subtree_com, site_xpos, site_xmat arrays
    """
    model = env._model
    data = mujoco.MjData(model)

    num_steps = len(qpos_all)

    # Preallocate arrays
    xpos_all = np.zeros((num_steps, model.nbody, 3))
    xquat_all = np.zeros((num_steps, model.nbody, 4))
    cvel_all = np.zeros((num_steps, model.nbody, 6))
    subtree_com_all = np.zeros((num_steps, model.nbody, 3))
    site_xpos_all = np.zeros((num_steps, model.nsite, 3))
    site_xmat_all = np.zeros((num_steps, model.nsite, 9))

    print(f"Computing forward kinematics for {num_steps} timesteps...")

    for i in range(num_steps):
        # Set state
        data.qpos[:] = qpos_all[i]
        data.qvel[:] = qvel_all[i]

        # Compute forward kinematics
        mujoco.mj_forward(model, data)

        # Extract data
        xpos_all[i] = data.xpos.copy()
        xquat_all[i] = data.xquat.copy()
        cvel_all[i] = data.cvel.copy()
        subtree_com_all[i] = data.subtree_com.copy()
        site_xpos_all[i] = data.site_xpos.copy()
        site_xmat_all[i] = data.site_xmat.copy()

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{num_steps} timesteps")

    return {
        'xpos': xpos_all,
        'xquat': xquat_all,
        'cvel': cvel_all,
        'subtree_com': subtree_com_all,
        'site_xpos': site_xpos_all,
        'site_xmat': site_xmat_all,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 expert data to Trajectory format')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output npz file')
    parser.add_argument('--env_name', type=str, default='MjxWildRobot',
                        help='Environment name (for obs_container)')
    args = parser.parse_args()

    print(f"Loading HDF5 data from {args.input}")

    # Load HDF5 file
    with h5py.File(args.input, 'r') as f:
        num_trajs = f.attrs['num_trajectories']
        print(f"Found {num_trajs} trajectories")

        # Collect all trajectories
        all_qpos = []
        all_qvel = []

        for i in range(num_trajs):
            traj_group = f[f'trajectory_{i:03d}']
            all_qpos.append(np.array(traj_group['qpos']))
            all_qvel.append(np.array(traj_group['qvel']))

        # Concatenate all trajectories
        qpos = np.concatenate(all_qpos, axis=0)
        qvel = np.concatenate(all_qvel, axis=0)

    print(f"Loaded {len(qpos)} total timesteps from {num_trajs} trajectories")

    # Create temporary environment to get obs_container and compute forward kinematics
    print(f"Creating temporary {args.env_name} environment...")

    # Use CPU version (WildRobot, not MjxWildRobot) for data processing
    cpu_env_name = args.env_name.replace("Mjx", "")  # MjxWildRobot -> WildRobot

    env = RLFactory.make(
        cpu_env_name,
        horizon=600,
        headless=True,
        reward_type="LocomotionReward",
        goal_type="GoalForwardRootVelocity",
    )

    # Compute forward kinematics
    fk_data = compute_forward_kinematics(env, qpos, qvel)

    # Create TrajectoryData with forward kinematics
    traj_data = TrajectoryData(
        qpos=qpos,
        qvel=qvel,
        xpos=fk_data['xpos'],
        xquat=fk_data['xquat'],
        cvel=fk_data['cvel'],
        subtree_com=fk_data['subtree_com'],
        site_xpos=fk_data['site_xpos'],
        site_xmat=fk_data['site_xmat'],
        split_points=np.array([0, len(qpos)]),  # Single concatenated trajectory
    )

    # Create TrajectoryModel
    model = env._model
    traj_model = TrajectoryModel(
        njnt=model.njnt,
        jnt_type=np.array([model.jnt_type[i] for i in range(model.njnt)]),
        nbody=model.nbody,
        body_rootid=model.body_rootid.copy(),
        body_weldid=model.body_weldid.copy(),
        body_mocapid=model.body_mocapid.copy(),
        body_pos=model.body_pos.copy(),
        body_quat=model.body_quat.copy(),
        body_ipos=model.body_ipos.copy(),
        body_iquat=model.body_iquat.copy(),
        nsite=model.nsite,
        site_bodyid=model.site_bodyid.copy(),
        site_pos=model.site_pos.copy(),
        site_quat=model.site_quat.copy(),
    )

    # Create TrajectoryInfo
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(model.nbody)]
    site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                  for i in range(model.nsite)]

    traj_info = TrajectoryInfo(
        joint_names=joint_names,
        body_names=body_names,
        site_names=site_names,
        model=traj_model,
        frequency=50,  # 50 Hz (dt=0.02)
    )

    # Create Trajectory object
    print("Creating Trajectory object...")
    trajectory = Trajectory(
        info=traj_info,
        data=traj_data,
        obs_container=env.obs_container,
    )

    # Save as npz
    print(f"Saving to {args.output}")
    trajectory.save(args.output)

    print("\n" + "="*80)
    print("✓ Conversion complete!")
    print("="*80)
    print(f"\nYou can now use this file in your AMP config:")
    print(f"  custom_expert_path: \"{args.output}\"")


if __name__ == '__main__':
    main()
