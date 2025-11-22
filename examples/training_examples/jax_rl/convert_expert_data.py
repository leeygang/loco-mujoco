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
from loco_mujoco import RLFactory
from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo


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
        all_ctrl = []
        all_time = []

        for i in range(num_trajs):
            traj_group = f[f'trajectory_{i:03d}']
            all_qpos.append(np.array(traj_group['qpos']))
            all_qvel.append(np.array(traj_group['qvel']))
            all_ctrl.append(np.array(traj_group['ctrl']))
            all_time.append(np.array(traj_group['time']))

        # Concatenate all trajectories
        qpos = np.concatenate(all_qpos, axis=0)
        qvel = np.concatenate(all_qvel, axis=0)
        ctrl = np.concatenate(all_ctrl, axis=0)
        time = np.concatenate(all_time, axis=0)

    print(f"Loaded {len(qpos)} total timesteps from {num_trajs} trajectories")

    # Create temporary environment to get obs_container
    print(f"Creating temporary {args.env_name} environment...")
    env = RLFactory.make(
        args.env_name,
        horizon=600,
        headless=True,
        reward_type="LocomotionReward",
        goal_type="GoalForwardRootVelocity",
    )

    # Create TrajectoryData
    traj_data = TrajectoryData(
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        time=time,
    )

    # Create TrajectoryInfo
    traj_info = TrajectoryInfo(
        dt=0.02,  # 50 Hz
        freq=50,
        n_substeps=env.n_substeps,
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
