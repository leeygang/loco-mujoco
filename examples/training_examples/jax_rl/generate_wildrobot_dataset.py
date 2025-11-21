"""
Generate WildRobot expert trajectories from a trained PPO policy.

This script:
1. Loads a trained PPO agent
2. Rolls out episodes to collect diverse walking motions
3. Saves trajectories in loco-mujoco format for AMP training

Usage:
    python generate_wildrobot_dataset.py --policy_path outputs/PPOJax_saved.pkl --output_dir wildrobot_motions
"""
import argparse
import pickle
import os
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List
import h5py

from loco_mujoco import RLFactory


def collect_trajectories(
    env,
    agent,
    num_episodes: int = 100,
    seed: int = 0,
    min_episode_length: int = 100,
) -> List[Dict]:
    """
    Collect trajectories from the trained agent.

    Returns:
        List of trajectory dictionaries with keys:
            - qpos: (T, nq) joint positions
            - qvel: (T, nv) joint velocities
            - ctrl: (T, nu) control inputs
            - time: (T,) timestamps
    """
    print(f"Collecting {num_episodes} trajectories...")

    rng = random.PRNGKey(seed)
    trajectories = []

    episodes_collected = 0
    attempts = 0
    max_attempts = num_episodes * 3  # Allow some failed episodes

    while episodes_collected < num_episodes and attempts < max_attempts:
        attempts += 1
        rng, reset_rng = random.split(rng)

        # Reset environment
        obs, state = env.reset(reset_rng)

        # Storage for this episode
        qpos_list = []
        qvel_list = []
        ctrl_list = []
        time_list = []

        done = False
        step_count = 0
        max_steps = 600  # Match your horizon

        while not done and step_count < max_steps:
            # Get action from policy
            rng, action_rng = random.split(rng)
            action, _ = agent.get_action(obs, action_rng)

            # Store MuJoCo state
            qpos_list.append(np.array(state.pipeline_state.qpos))
            qvel_list.append(np.array(state.pipeline_state.qvel))
            ctrl_list.append(np.array(action))
            time_list.append(state.pipeline_state.time)

            # Step environment
            rng, step_rng = random.split(rng)
            obs, state, reward, done, info = env.step(state, action, step_rng)

            step_count += 1

        # Only save if episode is long enough (robot didn't fall immediately)
        if step_count >= min_episode_length:
            trajectory = {
                'qpos': np.array(qpos_list),
                'qvel': np.array(qvel_list),
                'ctrl': np.array(ctrl_list),
                'time': np.array(time_list),
            }
            trajectories.append(trajectory)
            episodes_collected += 1

            if episodes_collected % 10 == 0:
                print(f"  Collected {episodes_collected}/{num_episodes} episodes (avg length: {step_count})")
        else:
            print(f"  Skipping short episode (length: {step_count})")

    print(f"Successfully collected {episodes_collected} trajectories in {attempts} attempts")
    return trajectories


def save_trajectories_hdf5(trajectories: List[Dict], output_path: str):
    """
    Save trajectories in HDF5 format compatible with loco-mujoco.

    Format matches the structure used by AMASS/CMU datasets.
    """
    print(f"Saving {len(trajectories)} trajectories to {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Save each trajectory as a separate group
        for i, traj in enumerate(trajectories):
            group = f.create_group(f'trajectory_{i:03d}')

            # Store trajectory data
            group.create_dataset('qpos', data=traj['qpos'], compression='gzip')
            group.create_dataset('qvel', data=traj['qvel'], compression='gzip')
            group.create_dataset('ctrl', data=traj['ctrl'], compression='gzip')
            group.create_dataset('time', data=traj['time'], compression='gzip')

            # Metadata
            group.attrs['length'] = len(traj['time'])
            group.attrs['dt'] = np.mean(np.diff(traj['time'])) if len(traj['time']) > 1 else 0.02

        # Global metadata
        f.attrs['num_trajectories'] = len(trajectories)
        f.attrs['source'] = 'PPO policy'

    print(f"Saved to {output_path}")


def save_trajectories_npz(trajectories: List[Dict], output_dir: str):
    """
    Save each trajectory as a separate .npz file (alternative format).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {len(trajectories)} trajectories to {output_dir}/*.npz")

    for i, traj in enumerate(trajectories):
        output_path = os.path.join(output_dir, f'wildrobot_walk_{i:03d}.npz')
        np.savez_compressed(
            output_path,
            qpos=traj['qpos'],
            qvel=traj['qvel'],
            ctrl=traj['ctrl'],
            time=traj['time'],
        )

    print(f"Saved {len(trajectories)} files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate WildRobot expert trajectories from PPO policy')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to saved PPO agent (.pkl file)')
    parser.add_argument('--output_dir', type=str, default='wildrobot_expert_motions',
                        help='Output directory for trajectories')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of trajectories to collect')
    parser.add_argument('--min_episode_length', type=int, default=100,
                        help='Minimum episode length to save')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--format', type=str, choices=['hdf5', 'npz', 'both'], default='hdf5',
                        help='Output format')

    args = parser.parse_args()

    # Load trained agent
    print(f"Loading policy from {args.policy_path}")
    with open(args.policy_path, 'rb') as f:
        agent = pickle.load(f)
    print("Policy loaded successfully")

    # Create environment (same as training config)
    print("Creating environment...")
    env = RLFactory.make(
        "MjxWildRobot",
        horizon=600,
        headless=True,
        reward_type="LocomotionReward",
        reward_params={
            "tracking_w_exp_xy": 6.0,
            "tracking_w_exp_yaw": 4.0,
            "tracking_w_sum_xy": 3.5,
            "tracking_w_sum_yaw": 1.0,
            "air_time_coeff": 0.1,
            "joint_acc_coeff": 2.0e-05,
            "air_time_max": 0.5,
            "joint_torque_coeff": 2.0e-07,
            "joint_position_limit_coeff": 2.0,
            "action_rate_coeff": 0.02,
            "symmetry_air_coeff": 0.005,
            "energy_coeff": 1.0e-05,
        },
        goal_type="GoalForwardRootVelocity",
        goal_params={
            "visualize_goal": False,
            "min_x_vel": 0.8,
            "max_x_vel": 2.0,
            "min_y_vel": -0.2,
            "max_y_vel": 0.2,
            "min_yaw_vel": -0.3,
            "max_yaw_vel": 0.3,
        },
        terminal_state_type="HeightBasedTerminalStateHandler",
        terminal_state_params={
            "min_height": 0.0,
            "max_height": 2.0,
        },
    )
    print("Environment created")

    # Collect trajectories
    trajectories = collect_trajectories(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        seed=args.seed,
        min_episode_length=args.min_episode_length,
    )

    # Save trajectories
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format in ['hdf5', 'both']:
        hdf5_path = os.path.join(args.output_dir, 'wildrobot_expert_dataset.h5')
        save_trajectories_hdf5(trajectories, hdf5_path)

    if args.format in ['npz', 'both']:
        npz_dir = os.path.join(args.output_dir, 'npz_files')
        save_trajectories_npz(trajectories, npz_dir)

    print("\nDone! Next steps:")
    print(f"1. Check trajectories in: {args.output_dir}/")
    print(f"2. Use these motions for AMP training by updating your config:")
    print(f"   - Change task_factory to ImitationFactory")
    print(f"   - Point to your custom dataset instead of AMASS")


if __name__ == '__main__':
    main()
