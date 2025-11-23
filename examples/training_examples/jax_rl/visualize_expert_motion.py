"""
Visualize expert trajectory data by playing it back in the environment.

This lets you SEE what motion AMP will learn from before training.

Usage:
    python visualize_expert_motion.py \
        --input wildrobot_expert_traj_fast.npz \
        --record \
        --duration 10
"""
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
from loco_mujoco import RLFactory
from loco_mujoco.trajectory import Trajectory


def main():
    parser = argparse.ArgumentParser(description='Visualize expert trajectory motion')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to expert trajectory (.npz)')
    parser.add_argument('--record', action='store_true',
                        help='Record video instead of interactive viewer')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration to visualize (seconds)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: auto-generated)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed (1.0 = real-time)')
    args = parser.parse_args()

    print(f"Loading expert trajectory from {args.input}")
    traj = Trajectory.load(args.input, backend=np)

    # Print stats
    num_timesteps = len(traj.data.qpos)
    duration_total = num_timesteps * 0.02
    print(f"Trajectory contains {num_timesteps} timesteps ({duration_total:.1f}s total)")

    # Analyze motion
    root_pos = traj.data.xpos[:, 0, :]
    root_vel = np.diff(root_pos, axis=0) / 0.02
    forward_vel = root_vel[:, 0]

    print(f"\nMotion characteristics:")
    print(f"  Forward velocity: {np.mean(forward_vel):.3f} ± {np.std(forward_vel):.3f} m/s")
    print(f"  Height: {np.mean(root_pos[:, 2]):.3f} ± {np.std(root_pos[:, 2]):.3f} m")

    # Create environment (CPU version for visualization)
    print("\nCreating environment...")
    env = RLFactory.make(
        "WildRobot",  # CPU version (not Mjx) for visualization
        horizon=600,
        headless=args.record,  # Headless if recording
        reward_type="LocomotionReward",
        goal_type="GoalForwardRootVelocity",
    )

    model = env._model
    data = mujoco.MjData(model)

    # Calculate how many frames to show
    max_frames = int(args.duration / 0.02)
    max_frames = min(max_frames, num_timesteps)

    print(f"\nVisualizing {max_frames} frames ({max_frames * 0.02:.1f}s)")

    if args.record:
        # RECORDING MODE
        print("Recording video...")

        # Setup camera
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()

        # Camera settings - follow the robot
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = model.body(env.root_body_name).id
        camera.distance = 4.0
        camera.elevation = -20
        camera.azimuth = 135

        # Render settings
        renderer = mujoco.Renderer(model, height=720, width=1280)

        frames = []

        for i in range(max_frames):
            # Set state from trajectory
            data.qpos[:] = traj.data.qpos[i]
            data.qvel[:] = traj.data.qvel[i]

            # Forward kinematics
            mujoco.mj_forward(model, data)

            # Update camera to follow robot
            camera.lookat[:] = data.xpos[model.body(env.root_body_name).id]

            # Render frame
            renderer.update_scene(data, camera, option)
            pixels = renderer.render()
            frames.append(pixels)

            if (i + 1) % 250 == 0:
                print(f"  Rendered {i+1}/{max_frames} frames")

        # Save video
        if args.output is None:
            output_path = args.input.replace('.npz', '_preview.mp4')
        else:
            output_path = args.output

        print(f"\nSaving video to {output_path}")
        fps = int(50 / args.speed)  # Adjust FPS based on playback speed
        media.write_video(output_path, frames, fps=fps)

        print("\n" + "="*80)
        print("✓ Video saved!")
        print("="*80)
        print(f"Output: {output_path}")
        print(f"Duration: {len(frames) / fps:.1f}s")
        print(f"Frames: {len(frames)}")

    else:
        # INTERACTIVE MODE
        print("\nStarting interactive viewer...")
        print("Controls:")
        print("  Space: Pause/Resume")
        print("  Right arrow: Step forward")
        print("  Esc: Exit")

        frame_idx = [0]  # Mutable container for closure
        paused = [False]

        def key_callback(keycode):
            if keycode == 32:  # Space
                paused[0] = not paused[0]
                print("Paused" if paused[0] else "Playing")
            elif keycode == 262:  # Right arrow
                if frame_idx[0] < max_frames - 1:
                    frame_idx[0] += 1

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            key_callback=key_callback,
        ) as viewer:

            # Camera setup
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body(env.root_body_name).id
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135

            while viewer.is_running():
                if not paused[0]:
                    # Set state from trajectory
                    if frame_idx[0] < max_frames:
                        data.qpos[:] = traj.data.qpos[frame_idx[0]]
                        data.qvel[:] = traj.data.qvel[frame_idx[0]]

                        # Forward kinematics
                        mujoco.mj_forward(model, data)

                        # Update camera position
                        viewer.cam.lookat[:] = data.xpos[model.body(env.root_body_name).id]

                        frame_idx[0] += 1
                    else:
                        # Loop back
                        frame_idx[0] = 0

                # Sync with wall clock (adjusted for playback speed)
                viewer.sync()

        print("\nViewer closed")


if __name__ == '__main__':
    main()
