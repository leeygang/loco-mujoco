"""
Visualize trained AMP policy motion and compare with expert data.

Usage:
    python visualize_policy_motion.py \
        --policy_path outputs/.../AMPJax_saved.pkl \
        --expert_path wildrobot_expert_traj_fast.npz \
        --num_episodes 5
"""
import argparse
import pickle
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

from loco_mujoco import RLFactory
from loco_mujoco.trajectory import Trajectory


def collect_policy_data(policy_path, num_episodes=5, seed=0):
    """Collect motion data from trained policy."""

    print(f"Loading policy from {policy_path}")
    with open(policy_path, 'rb') as f:
        agent = pickle.load(f)

    network = agent['agent_conf']['network']
    params = agent['agent_state']['train_state']['params']
    run_stats = agent['agent_state']['train_state']['run_stats']

    # Handle multi-seed agents
    config = agent['agent_conf']['config']
    n_seeds = config.get('experiment', {}).get('n_seeds', 1)
    if n_seeds > 1:
        params = jax.tree.map(lambda x: x[0], params)
        run_stats = jax.tree.map(lambda x: x[0], run_stats)

    print("Creating environment...")
    env = RLFactory.make(
        "MjxWildRobot",
        horizon=600,
        headless=True,
        reward_type="LocomotionReward",
        goal_type="GoalForwardRootVelocity",
        goal_params={
            "visualize_goal": False,
            "min_x_vel": 0.6,
            "max_x_vel": 1.2,
            "min_y_vel": -0.1,
            "max_y_vel": 0.1,
            "min_yaw_vel": -0.2,
            "max_yaw_vel": 0.2,
        },
    )

    @jax.jit
    def get_action(params, run_stats, obs, rng):
        variables = {"params": params, "run_stats": run_stats}
        (pi, _), _ = network.apply(variables, obs, mutable=['run_stats'])
        action = pi.sample(seed=rng)
        return action

    print(f"Collecting {num_episodes} episodes...")
    rng = random.PRNGKey(seed)

    all_root_pos = []
    all_root_vel = []
    all_joint_vel = []

    for ep in range(num_episodes):
        rng, reset_rng = random.split(rng)
        obs = env.reset(reset_rng)

        root_pos_ep = []
        joint_vel_ep = []

        for step in range(600):
            rng, action_rng = random.split(rng)
            action = get_action(params, run_stats, obs, action_rng)

            # Store data
            root_pos = np.array(env.data.xpos[env._model.body(env.root_body_name).id])
            root_pos_ep.append(root_pos)
            joint_vel_ep.append(np.array(env.data.qvel))

            obs, reward, absorbing, done, info = env.step(action)
            if done:
                break

        root_pos_ep = np.array(root_pos_ep)
        all_root_pos.append(root_pos_ep)
        all_joint_vel.append(np.array(joint_vel_ep))

        # Compute velocity
        root_vel_ep = np.diff(root_pos_ep, axis=0) / 0.02
        all_root_vel.append(root_vel_ep)

        print(f"  Episode {ep+1}: {len(root_pos_ep)} steps")

    return {
        'root_pos': all_root_pos,
        'root_vel': all_root_vel,
        'joint_vel': all_joint_vel,
    }


def analyze_expert_data(expert_path):
    """Extract motion statistics from expert data."""

    print(f"Loading expert data from {expert_path}")
    traj = Trajectory.load(expert_path, backend=np)

    root_pos = traj.data.xpos[:, 0, :]  # First body (root)
    root_vel = np.diff(root_pos, axis=0) / 0.02
    joint_vel = traj.data.qvel

    return {
        'root_pos': root_pos,
        'root_vel': root_vel,
        'joint_vel': joint_vel,
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize policy vs expert motion')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to trained policy (.pkl)')
    parser.add_argument('--expert_path', type=str, required=True,
                        help='Path to expert trajectory (.npz)')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of policy episodes to collect')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    # Collect data
    print("\n" + "="*80)
    print("COLLECTING POLICY DATA")
    print("="*80)
    policy_data = collect_policy_data(args.policy_path, args.num_episodes, args.seed)

    print("\n" + "="*80)
    print("LOADING EXPERT DATA")
    print("="*80)
    expert_data = analyze_expert_data(args.expert_path)

    # Analyze
    print("\n" + "="*80)
    print("MOTION COMPARISON")
    print("="*80)

    # Policy stats (average across episodes)
    policy_forward_vels = [vel[:, 0] for vel in policy_data['root_vel']]
    policy_mean_vel = np.mean([np.mean(v) for v in policy_forward_vels])
    policy_std_vel = np.mean([np.std(v) for v in policy_forward_vels])

    # Expert stats
    expert_forward_vel = expert_data['root_vel'][:, 0]
    expert_mean_vel = np.mean(expert_forward_vel)
    expert_std_vel = np.std(expert_forward_vel)

    print(f"\nForward Velocity:")
    print(f"  Expert:  {expert_mean_vel:.3f} ± {expert_std_vel:.3f} m/s")
    print(f"  Policy:  {policy_mean_vel:.3f} ± {policy_std_vel:.3f} m/s")
    print(f"  Difference: {abs(policy_mean_vel - expert_mean_vel):.3f} m/s")

    # Joint velocities
    expert_joint_vel_mean = np.mean(np.abs(expert_data['joint_vel']))
    policy_joint_vel_mean = np.mean([np.mean(np.abs(jv)) for jv in policy_data['joint_vel']])

    print(f"\nJoint Velocity (mean abs):")
    print(f"  Expert:  {expert_joint_vel_mean:.3f} rad/s")
    print(f"  Policy:  {policy_joint_vel_mean:.3f} rad/s")
    print(f"  Difference: {abs(policy_joint_vel_mean - expert_joint_vel_mean):.3f} rad/s")

    # Plot comparison
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Expert forward velocity
    time_expert = np.arange(len(expert_forward_vel)) * 0.02
    axes[0, 0].plot(time_expert, expert_forward_vel, linewidth=1, alpha=0.7, color='blue')
    axes[0, 0].axhline(expert_mean_vel, color='blue', linestyle='--',
                       label=f'Mean: {expert_mean_vel:.3f} m/s')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Forward velocity (m/s)')
    axes[0, 0].set_title('Expert Motion')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Policy forward velocity (multiple episodes)
    for i, vel in enumerate(policy_forward_vels):
        time_policy = np.arange(len(vel)) * 0.02
        axes[0, 1].plot(time_policy, vel, linewidth=1, alpha=0.5,
                        label=f'Episode {i+1}', color='orange')
    axes[0, 1].axhline(policy_mean_vel, color='red', linestyle='--',
                       label=f'Mean: {policy_mean_vel:.3f} m/s', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Forward velocity (m/s)')
    axes[0, 1].set_title('Policy Motion')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Velocity distribution comparison
    axes[1, 0].hist(expert_forward_vel, bins=50, alpha=0.5, label='Expert', color='blue', density=True)
    for vel in policy_forward_vels:
        axes[1, 0].hist(vel, bins=50, alpha=0.3, color='orange', density=True)
    axes[1, 0].axvline(expert_mean_vel, color='blue', linestyle='--', linewidth=2, label=f'Expert mean: {expert_mean_vel:.3f}')
    axes[1, 0].axvline(policy_mean_vel, color='red', linestyle='--', linewidth=2, label=f'Policy mean: {policy_mean_vel:.3f}')
    axes[1, 0].set_xlabel('Forward velocity (m/s)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Velocity Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Trajectory in XY plane
    expert_xy = expert_data['root_pos'][:, :2]
    axes[1, 1].plot(expert_xy[:, 0], expert_xy[:, 1], linewidth=2, alpha=0.7,
                    label='Expert', color='blue')
    for i, pos in enumerate(policy_data['root_pos']):
        axes[1, 1].plot(pos[:, 0], pos[:, 1], linewidth=1, alpha=0.5,
                        label=f'Policy Ep{i+1}', color='orange')
    axes[1, 1].set_xlabel('X position (m)')
    axes[1, 1].set_ylabel('Y position (m)')
    axes[1, 1].set_title('Walking Trajectory (Top View)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')

    plt.tight_layout()
    output_path = args.policy_path.replace('.pkl', '_vs_expert.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print("="*80)

    # Summary verdict
    vel_match = abs(policy_mean_vel - expert_mean_vel) < 0.15
    print(f"\nVelocity match: {'✅ GOOD' if vel_match else '❌ MISMATCH'}")
    print(f"Policy is walking at {policy_mean_vel:.3f} m/s vs expert {expert_mean_vel:.3f} m/s")


if __name__ == '__main__':
    main()
