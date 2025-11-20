"""
Demonstration of using IMU sensor observations in WildRobot environment.

This script shows how to use custom observation types that are external to
the loco-mujoco library. This pattern allows you to keep WildRobot-specific
code separate from loco-mujoco, making it easy to maintain them independently.

Key Pattern:
    1. Import custom observations from wildrobot_extensions
    2. Register them with loco-mujoco
    3. Use them like built-in observation types
"""

import sys
from pathlib import Path

# Add wildrobot_extensions to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from loco_mujoco import RLFactory
from loco_mujoco.core import ObservationType

# Import and register WildRobot custom observations
from wildrobot_extensions import IMUSensor, AllIMUSensors

# Register custom observation types with loco-mujoco
IMUSensor.register()
AllIMUSensors.register()


def example_1_individual_imus():
    """
    Example 1: Include individual IMU sensors in the observation space.
    """
    print("="*80)
    print("EXAMPLE 1: Individual IMU Sensors")
    print("="*80)

    # Define observation specification with individual IMU sensors
    observation_spec = [
        # Robot state
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),
        ObservationType.JointVelArray("joint_vel", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),

        # Physical IMU sensors (as on the real robot)
        # These are now registered from wildrobot_extensions, not from loco-mujoco!
        ObservationType.IMUSensor("chest_imu",
            ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),  # 9-DOF (BNO085)
        ObservationType.IMUSensor("left_knee_imu",
            ["left_knee_imu_gyro", "left_knee_imu_accel"]),  # 6-DOF (ICM45686)
        ObservationType.IMUSensor("right_knee_imu",
            ["right_knee_imu_gyro", "right_knee_imu_accel"]),  # 6-DOF (ICM45686)

        # Pelvis orientation (for reference)
        ObservationType.ProjectedGravityVector("gravity", "waist_freejoint"),

        # Last action
        ObservationType.LastAction("last_action"),
    ]

    # Create environment
    env = RLFactory.make(
        "WildRobot",
        observation_specification=observation_spec,
        reward_type="LocomotionReward",
        goal_type="GoalRandomRootVelocity"
    )

    print(f"\nObservation space dimension: {env.observation_space.shape[0]}")
    print("\nObservation breakdown:")
    for obs_name, obs in env.observation_specification.items():
        print(f"  {obs_name:20s}: dim={obs.dim:3d}  indices={obs.obs_ind[0]:4d}-{obs.obs_ind[-1]:4d}")

    # Run a few steps
    print("\n" + "-"*80)
    print("Running simulation and reading IMU data...")
    print("-"*80)

    obs = env.reset()
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Extract IMU readings
        chest_imu = env._get_from_obs(obs, "chest_imu")
        left_knee_imu = env._get_from_obs(obs, "left_knee_imu")
        right_knee_imu = env._get_from_obs(obs, "right_knee_imu")

        print(f"\nStep {i+1}:")
        print(f"  Chest IMU (gyro+accel+mag): {chest_imu}")
        print(f"  Left knee IMU (gyro+accel):  {left_knee_imu}")
        print(f"  Right knee IMU (gyro+accel): {right_knee_imu}")

    env.close()
    print("\n✅ Example 1 completed successfully!\n")


def example_2_all_imus():
    """
    Example 2: Use AllIMUSensors to automatically include all IMU sensors.
    """
    print("="*80)
    print("EXAMPLE 2: All IMU Sensors (Automatic)")
    print("="*80)

    observation_spec = [
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),
        ObservationType.JointVelArray("joint_vel", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),

        # Automatically include ALL IMU sensors (gyro + accel + mag)
        ObservationType.AllIMUSensors("all_imus", include_magnetometer=True),

        ObservationType.LastAction("last_action"),
    ]

    env = RLFactory.make(
        "WildRobot",
        observation_specification=observation_spec,
        reward_type="LocomotionReward",
        goal_type="GoalRandomRootVelocity"
    )

    print(f"\nObservation space dimension: {env.observation_space.shape[0]}")
    print("\nObservation breakdown:")
    for obs_name, obs in env.observation_specification.items():
        print(f"  {obs_name:20s}: dim={obs.dim:3d}  indices={obs.obs_ind[0]:4d}-{obs.obs_ind[-1]:4d}")

    # Run a few steps
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    all_imus = env._get_from_obs(obs, "all_imus")
    print(f"\nAll IMU data (dimension {len(all_imus)}):")
    print(f"  {all_imus}")

    env.close()
    print("\n✅ Example 2 completed successfully!\n")


def example_3_mjx_compatibility():
    """
    Example 3: Test IMU observations with MJX (GPU-accelerated) environment.
    """
    print("="*80)
    print("EXAMPLE 3: MJX Environment with IMU Sensors")
    print("="*80)

    observation_spec = [
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),
        ObservationType.JointVelArray("joint_vel", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),

        # IMU sensors
        ObservationType.IMUSensor("chest_imu",
            ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),
        ObservationType.IMUSensor("left_knee_imu",
            ["left_knee_imu_gyro", "left_knee_imu_accel"]),
        ObservationType.IMUSensor("right_knee_imu",
            ["right_knee_imu_gyro", "right_knee_imu_accel"]),

        ObservationType.ProjectedGravityVector("gravity", "waist_freejoint"),
        ObservationType.LastAction("last_action"),
    ]

    # Create MJX environment (note "Mjx" prefix)
    env = RLFactory.make(
        "MjxWildRobot",  # MJX version for GPU acceleration
        observation_specification=observation_spec,
        reward_type="LocomotionReward",
        goal_type="GoalRandomRootVelocity"
    )

    print(f"\nMJX Environment created successfully!")
    print(f"Observation space dimension: {env.observation_space.shape[0]}")

    env.close()
    print("\n✅ Example 3 completed successfully!\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMU SENSOR OBSERVATION EXAMPLES FOR WILDROBOT")
    print("="*80)
    print("\nThis demonstrates using EXTERNAL custom observations")
    print("that are separate from the loco-mujoco library.")
    print("="*80)

    try:
        example_1_individual_imus()
        example_2_all_imus()
        example_3_mjx_compatibility()

        print("="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        print("\nKey Takeaway:")
        print("  Custom observations are in wildrobot_extensions/")
        print("  They are registered at runtime, not in loco-mujoco")
        print("  This keeps your code separate and maintainable!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
