"""
Subsample expert trajectory to reduce memory usage.

Usage:
    python subsample_expert_data.py \
        --input wildrobot_expert_traj.npz \
        --output wildrobot_expert_traj_small.npz \
        --target_size 30000
"""
import argparse
import numpy as np
from loco_mujoco.trajectory import Trajectory


def main():
    parser = argparse.ArgumentParser(description='Subsample expert trajectory')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input npz file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output npz file')
    parser.add_argument('--target_size', type=int, default=30000,
                        help='Target number of timesteps (default: 30000)')
    args = parser.parse_args()

    print(f"Loading trajectory from {args.input}")
    traj = Trajectory.load(args.input, backend=np)

    original_size = len(traj.data.qpos)
    print(f"Original size: {original_size} timesteps")

    if original_size <= args.target_size:
        print(f"Trajectory already smaller than target size, copying as-is")
        traj.save(args.output)
        return

    # Subsample uniformly
    indices = np.linspace(0, original_size - 1, args.target_size, dtype=int)
    print(f"Subsampling to {len(indices)} timesteps ({100*len(indices)/original_size:.1f}%)")

    # Create subsampled data
    subsampled_data = traj.data.replace(
        qpos=traj.data.qpos[indices],
        qvel=traj.data.qvel[indices],
        xpos=traj.data.xpos[indices],
        xquat=traj.data.xquat[indices],
        cvel=traj.data.cvel[indices],
        subtree_com=traj.data.subtree_com[indices],
        site_xpos=traj.data.site_xpos[indices],
        site_xmat=traj.data.site_xmat[indices],
        split_points=np.array([0, len(indices)]),
    )

    # Create new trajectory
    subsampled_traj = Trajectory(
        info=traj.info,
        data=subsampled_data,
        obs_container=traj.obs_container,
    )

    # Save
    print(f"Saving to {args.output}")
    subsampled_traj.save(args.output)

    print("\n" + "="*80)
    print("✓ Subsampling complete!")
    print("="*80)
    print(f"\nUpdate your config to use the smaller file:")
    print(f'  custom_expert_path: "../jax_rl/{args.output}"')


if __name__ == '__main__':
    main()
