"""
Inspect expert trajectory data to verify what motion is captured.

Usage:
    python inspect_expert_data.py --input wildrobot_expert_traj_fast.npz
"""
import argparse
import numpy as np
from loco_mujoco.trajectory import Trajectory
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Inspect expert trajectory')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to trajectory npz file')
    args = parser.parse_args()

    print(f"Loading trajectory from {args.input}")
    traj = Trajectory.load(args.input, backend=np)

    print("\n" + "="*80)
    print("TRAJECTORY INFO")
    print("="*80)

    # Basic stats
    num_timesteps = len(traj.data.qpos)
    print(f"\nTotal timesteps: {num_timesteps}")
    print(f"Duration: {num_timesteps * 0.02:.1f} seconds (at 50 Hz)")
    print(f"Number of trajectories: {traj.data.n_trajectories}")

    # Joint info
    print(f"\nJoints: {len(traj.info.joint_names)}")
    print(f"Bodies: {len(traj.info.body_names)}")
    print(f"Sites: {len(traj.info.site_names)}")

    # Analyze motion characteristics
    print("\n" + "="*80)
    print("MOTION CHARACTERISTICS")
    print("="*80)

    # Root position (assuming first body is root or pelvis)
    root_pos = traj.data.xpos[:, 0, :]  # [timesteps, 3] - x,y,z
    root_vel = np.diff(root_pos, axis=0) / 0.02  # velocity = diff / dt

    # Forward velocity (x-axis)
    forward_vel = root_vel[:, 0]
    print(f"\nForward velocity:")
    print(f"  Mean: {np.mean(forward_vel):.3f} m/s")
    print(f"  Std:  {np.std(forward_vel):.3f} m/s")
    print(f"  Min:  {np.min(forward_vel):.3f} m/s")
    print(f"  Max:  {np.max(forward_vel):.3f} m/s")

    # Lateral velocity (y-axis)
    lateral_vel = root_vel[:, 1]
    print(f"\nLateral velocity:")
    print(f"  Mean: {np.mean(lateral_vel):.3f} m/s")
    print(f"  Std:  {np.std(lateral_vel):.3f} m/s")

    # Height
    height = root_pos[:, 2]
    print(f"\nRoot height:")
    print(f"  Mean: {np.mean(height):.3f} m")
    print(f"  Std:  {np.std(height):.3f} m")
    print(f"  Min:  {np.min(height):.3f} m")
    print(f"  Max:  {np.max(height):.3f} m")

    # Estimate step length (distance between forward velocity peaks)
    # Simple approximation: average distance traveled per gait cycle
    forward_distance = np.cumsum(forward_vel * 0.02)
    total_distance = forward_distance[-1]
    duration_seconds = num_timesteps * 0.02

    print(f"\nDistance traveled: {total_distance:.1f} m over {duration_seconds:.1f} s")

    # Joint velocities
    joint_vel = traj.data.qvel
    print(f"\nJoint velocities:")
    print(f"  Mean abs: {np.mean(np.abs(joint_vel)):.3f} rad/s")
    print(f"  Max abs:  {np.max(np.abs(joint_vel)):.3f} rad/s")

    # Plot
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    time = np.arange(len(root_pos)) * 0.02

    # Root position
    axes[0].plot(time, root_pos[:, 0], label='X (forward)', linewidth=1.5)
    axes[0].plot(time, root_pos[:, 1], label='Y (lateral)', linewidth=1.5)
    axes[0].plot(time, root_pos[:, 2], label='Z (height)', linewidth=1.5)
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Root Position Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Root velocity
    time_vel = np.arange(len(root_vel)) * 0.02
    axes[1].plot(time_vel, forward_vel, label='Forward velocity', linewidth=1.5)
    axes[1].axhline(np.mean(forward_vel), color='r', linestyle='--',
                    label=f'Mean: {np.mean(forward_vel):.3f} m/s')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_title('Forward Velocity Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Joint velocities (sample first 6 joints)
    for i in range(min(6, joint_vel.shape[1])):
        axes[2].plot(time, joint_vel[:, i], label=traj.info.joint_names[i],
                     linewidth=0.8, alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Joint velocity (rad/s)')
    axes[2].set_title('Sample Joint Velocities')
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = args.input.replace('.npz', '_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    print("\n" + "="*80)
    print("✓ Inspection complete!")
    print("="*80)


if __name__ == '__main__':
    main()
