"""
Generate WildRobot expert trajectories from a trained PPO policy (Step 1 output).

This script loads a trained Step 1 policy and collects episodes for Step 2 AMP training.

Usage:
    python generate_wildrobot_dataset.py \
        --policy_path outputs/2025-11-22/11-19-29/PPOJax_saved.pkl \
        --num_episodes 200 \
        --output_dir wildrobot_expert_motions
"""
import argparse
import pickle
import os
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, Dict
import h5py

from loco_mujoco import RLFactory


def main():
    parser = argparse.ArgumentParser(description='Generate WildRobot expert trajectories from PPO policy')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to saved PPO agent (.pkl file)')
    parser.add_argument('--output_dir', type=str, default='wildrobot_expert_motions',
                        help='Output directory for trajectories')
    parser.add_argument('--num_episodes', type=int, default=200,
                        help='Number of trajectories to collect')
    parser.add_argument('--min_episode_length', type=int, default=100,
                        help='Minimum episode length to save')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--format', type=str, choices=['hdf5', 'npz', 'both'], default='hdf5',
                        help='Output format')
    # Velocity goal parameters
    parser.add_argument('--min_x_vel', type=float, default=0.3,
                        help='Minimum forward velocity (m/s)')
    parser.add_argument('--max_x_vel', type=float, default=0.8,
                        help='Maximum forward velocity (m/s)')
    parser.add_argument('--min_y_vel', type=float, default=-0.1,
                        help='Minimum lateral velocity (m/s)')
    parser.add_argument('--max_y_vel', type=float, default=0.1,
                        help='Maximum lateral velocity (m/s)')
    parser.add_argument('--min_yaw_vel', type=float, default=-0.2,
                        help='Minimum yaw velocity (rad/s)')
    parser.add_argument('--max_yaw_vel', type=float, default=0.2,
                        help='Maximum yaw velocity (rad/s)')
    args = parser.parse_args()

    # ========== LOAD AGENT ==========
    print(f"Loading policy from {args.policy_path}")
    with open(args.policy_path, 'rb') as f:
        agent = pickle.load(f)

    # Extract components
    network = agent['agent_conf']['network']
    train_state_dict = agent['agent_state']['train_state']
    params = train_state_dict['params']
    run_stats = train_state_dict['run_stats']

    # Check if this is actually a multi-seed agent (n_seeds > 1 with vmap)
    # Multi-seed params would have ALL parameters starting with an extra seed dimension
    # To detect: check if params have a consistent leading dimension across all leaves
    config = agent['agent_conf']['config']
    n_seeds = config.get('experiment', {}).get('n_seeds', 1)

    if n_seeds > 1:
        print(f"Multi-seed agent detected (n_seeds={n_seeds}). Using seed 0")
        # Extract first seed from vmapped params
        params = jax.tree.map(lambda x: x[0], params)
        run_stats = jax.tree.map(lambda x: x[0], run_stats)

    print("Policy loaded successfully")

    # ========== CREATE ENVIRONMENT ==========
    print("Creating environment...")
    print(f"  Using velocity goals: x=[{args.min_x_vel}, {args.max_x_vel}] m/s")
    env = RLFactory.make(
        "MjxWildRobot",
        horizon=600,
        headless=True,
        reward_type="LocomotionReward",
        reward_params={
            "tracking_w_exp_xy": 15.0,
            "tracking_w_exp_yaw": 4.0,
            "tracking_w_sum_xy": 5.0,
            "tracking_w_sum_yaw": 1.0,
            "z_vel_coeff": 2.0,
            "roll_pitch_vel_coeff": 0.08,
            "roll_pitch_pos_coeff": 0.3,
            "nominal_joint_pos_coeff": 0.005,
            "joint_position_limit_coeff": 5.0,
            "joint_vel_coeff": 5e-5,
            "joint_acc_coeff": 2e-5,
            "joint_torque_coeff": 2e-7,
            "action_rate_coeff": 0.02,
            "air_time_max": 0.15,
            "air_time_coeff": 0.15,
            "symmetry_air_coeff": 0.08,
            "energy_coeff": 2e-5,
        },
        goal_type="GoalForwardRootVelocity",
        goal_params={
            "visualize_goal": False,
            "min_x_vel": args.min_x_vel,
            "max_x_vel": args.max_x_vel,
            "min_y_vel": args.min_y_vel,
            "max_y_vel": args.max_y_vel,
            "min_yaw_vel": args.min_yaw_vel,
            "max_yaw_vel": args.max_yaw_vel,
        },
        terminal_state_type="HeightBasedTerminalStateHandler",
        terminal_state_params={
            "min_height": 0.2,
            "max_height": 0.6,
        },
    )
    print("Environment created")

    # ========== COLLECT TRAJECTORIES ==========
    print(f"Collecting {args.num_episodes} trajectories...")

    # JIT-compiled policy sampling (stochastic)
    @jax.jit
    def get_action(params, run_stats, obs, rng):
        # Call network allowing run_stats updates (but we'll ignore them)
        variables = {"params": params, "run_stats": run_stats}
        # mutable=['run_stats'] allows updates, returns (output, updated_vars)
        (pi, _), updated_vars = network.apply(variables, obs, mutable=['run_stats'])
        action = pi.sample(seed=rng)
        # We ignore updated_vars - use the original run_stats for all episodes
        return action

    rng = random.PRNGKey(args.seed)
    trajectories = []
    episodes_collected = 0
    attempts = 0
    max_attempts = args.num_episodes * 3

    while episodes_collected < args.num_episodes and attempts < max_attempts:
        attempts += 1
        rng, reset_rng = random.split(rng)

        # Reset environment
        obs = env.reset(reset_rng)

        # Storage for this episode
        qpos_list = []
        qvel_list = []
        ctrl_list = []
        time_list = []

        done = False
        step_count = 0

        while not done and step_count < 600:
            # Sample action
            rng, action_rng = random.split(rng)
            action = get_action(params, run_stats, obs, action_rng)

            # Store current state
            qpos_list.append(np.array(env.data.qpos))
            qvel_list.append(np.array(env.data.qvel))
            ctrl_list.append(np.array(action))
            time_list.append(float(env.data.time))

            # Step environment (MuJoCo env returns: obs, reward, absorbing, done, info)
            obs, reward, absorbing, done, info = env.step(action)
            done = bool(done)
            step_count += 1

        # Save if episode is long enough
        if step_count >= args.min_episode_length:
            trajectory = {
                'qpos': np.array(qpos_list),
                'qvel': np.array(qvel_list),
                'ctrl': np.array(ctrl_list),
                'time': np.array(time_list),
            }
            trajectories.append(trajectory)
            episodes_collected += 1

            if episodes_collected % 10 == 0:
                print(f"  Collected {episodes_collected}/{args.num_episodes} episodes (length: {step_count})")
        else:
            print(f"  Skipping short episode (length: {step_count})")

    print(f"Successfully collected {episodes_collected} trajectories in {attempts} attempts")

    # ========== SAVE TRAJECTORIES ==========
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format in ['hdf5', 'both']:
        hdf5_path = os.path.join(args.output_dir, 'wildrobot_expert_dataset.h5')
        print(f"Saving {len(trajectories)} trajectories to {hdf5_path}")

        with h5py.File(hdf5_path, 'w') as f:
            for i, traj in enumerate(trajectories):
                group = f.create_group(f'trajectory_{i:03d}')
                group.create_dataset('qpos', data=traj['qpos'], compression='gzip')
                group.create_dataset('qvel', data=traj['qvel'], compression='gzip')
                group.create_dataset('ctrl', data=traj['ctrl'], compression='gzip')
                group.create_dataset('time', data=traj['time'], compression='gzip')
                group.attrs['length'] = len(traj['time'])
                group.attrs['dt'] = np.mean(np.diff(traj['time'])) if len(traj['time']) > 1 else 0.02

            f.attrs['num_trajectories'] = len(trajectories)
            f.attrs['source'] = 'PPO Step 1 policy'

        print(f"Saved to {hdf5_path}")

    if args.format in ['npz', 'both']:
        npz_dir = os.path.join(args.output_dir, 'npz_files')
        os.makedirs(npz_dir, exist_ok=True)
        print(f"Saving {len(trajectories)} trajectories to {npz_dir}/*.npz")

        for i, traj in enumerate(trajectories):
            output_path = os.path.join(npz_dir, f'wildrobot_walk_{i:03d}.npz')
            np.savez_compressed(output_path, **traj)

        print(f"Saved {len(trajectories)} files to {npz_dir}/")

    print("\n" + "="*80)
    print("✓ Expert data generation complete!")
    print("="*80)
    print(f"\nNext steps for Step 2 training:")
    print(f"1. Update conf_step2_humanlike_amp.yaml:")
    print(f"   rel_dataset_path:")
    print(f"     - \"{os.path.basename(args.output_dir)}\"")
    print(f"\n2. Run Step 2 training:")
    print(f"   python experiment.py --config-name conf_step2_humanlike_amp")


if __name__ == '__main__':
    main()
