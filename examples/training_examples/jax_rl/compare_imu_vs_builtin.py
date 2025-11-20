"""
Comparison: Custom IMU Sensors vs Built-in Observations

This script demonstrates the difference (or lack thereof) between:
1. Custom IMU sensors (reading from sensordata)
2. Built-in observations (reading from simulation state)

It also shows the value when sensor noise is added.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import mujoco
from loco_mujoco import RLFactory
from loco_mujoco.core import ObservationType
from wildrobot_extensions import IMUSensor

# Register custom observations
IMUSensor.register()


def compare_without_noise():
    """
    Compare IMU sensors vs built-in observations WITHOUT noise.
    Expected: They should be very similar (reading same perfect state).
    """
    print("="*80)
    print("COMPARISON 1: Without Sensor Noise")
    print("="*80)

    # Create environment with both observation types
    observation_spec = [
        # Custom IMU sensor
        ObservationType.IMUSensor("chest_imu_gyro_only", ["chest_imu_gyro"]),
        ObservationType.IMUSensor("chest_imu_accel_only", ["chest_imu_accel"]),

        # Built-in observations (for comparison)
        ObservationType.BodyVel("pelvis_vel", "waist"),  # Includes angular + linear vel
        ObservationType.ProjectedGravityVector("gravity", "waist_freejoint"),

        # Common
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),
        ObservationType.LastAction("last_action"),
    ]

    env = RLFactory.make(
        "WildRobot",
        observation_specification=observation_spec,
        reward_type="LocomotionReward",
        goal_type="GoalRandomRootVelocity"
    )

    # Collect data
    obs = env.reset()
    imu_gyro_data = []
    body_vel_data = []

    for _ in range(100):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

        # IMU gyro (custom sensor)
        chest_gyro = env._get_from_obs(obs, "chest_imu_gyro_only")
        imu_gyro_data.append(chest_gyro)

        # Body velocity (built-in observation)
        pelvis_vel = env._get_from_obs(obs, "pelvis_vel")
        body_angular_vel = pelvis_vel[:3]  # First 3 components are angular velocity
        body_vel_data.append(body_angular_vel)

    imu_gyro_data = np.array(imu_gyro_data)
    body_vel_data = np.array(body_vel_data)

    # Compare
    print("\nAngular Velocity Comparison (100 steps):")
    print(f"  IMU Gyro mean:     {np.mean(imu_gyro_data, axis=0)}")
    print(f"  BodyVel mean:      {np.mean(body_vel_data, axis=0)}")
    print(f"  Difference:        {np.mean(np.abs(imu_gyro_data - body_vel_data)):.6f}")

    print("\n" + "-"*80)
    if np.mean(np.abs(imu_gyro_data - body_vel_data)) < 0.01:
        print("✓ IMU sensors and BodyVel are nearly IDENTICAL (no noise)")
        print("  → Custom IMU provides NO advantage without sensor noise")
    else:
        print("⚠ Unexpected difference detected")

    env.close()


def demonstrate_with_noise():
    """
    Show what happens when we add sensor noise to IMU.
    This demonstrates the VALUE of custom IMU observations.
    """
    print("\n" + "="*80)
    print("COMPARISON 2: With Sensor Noise (Simulated)")
    print("="*80)
    print("\nNote: To actually test this, you need to add 'noise' attributes")
    print("to sensors in wildrobot.xml. This demo simulates the effect.")

    # Create environment
    observation_spec = [
        ObservationType.IMUSensor("chest_imu",
            ["chest_imu_gyro", "chest_imu_accel"]),
        ObservationType.BodyVel("pelvis_vel", "waist"),
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),
        ObservationType.LastAction("last_action"),
    ]

    env = RLFactory.make(
        "WildRobot",
        observation_specification=observation_spec,
        reward_type="LocomotionReward",
        goal_type="GoalRandomRootVelocity"
    )

    # Manually add noise to demonstrate effect
    obs = env.reset()
    print("\n" + "-"*80)
    print("Simulating sensor noise (add 'noise' attribute to XML for real effect):")

    for i in range(5):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

        # Get IMU and BodyVel
        chest_imu = env._get_from_obs(obs, "chest_imu")
        pelvis_vel = env._get_from_obs(obs, "pelvis_vel")

        # Simulate what noisy IMU would look like
        gyro_noise = np.random.normal(0, 0.002, 3)  # BNO085 noise spec
        accel_noise = np.random.normal(0, 0.015, 3)
        noisy_imu = chest_imu + np.concatenate([gyro_noise, accel_noise])

        print(f"\nStep {i+1}:")
        print(f"  Perfect IMU:  gyro={chest_imu[:3]}")
        print(f"  Noisy IMU:    gyro={noisy_imu[:3]}")
        print(f"  BodyVel:      angvel={pelvis_vel[:3]}")

    print("\n" + "-"*80)
    print("With noise in XML:")
    print("  ✓ Noisy IMU = Realistic sensor (matches real robot)")
    print("  ✓ BodyVel = Perfect state (privileged information)")
    print("  ✓ Use both for teacher-student training")
    print("  ✓ Deploy only noisy IMU to real robot")

    env.close()


def show_value_proposition():
    """
    Summary of when custom IMU sensors provide value.
    """
    print("\n" + "="*80)
    print("VALUE PROPOSITION SUMMARY")
    print("="*80)

    print("\n📊 CURRENT STATE (No Noise in XML):")
    print("  ❌ IMU sensors ≈ BodyVel/ProjectedGravity")
    print("  ❌ Both read perfect simulation state")
    print("  ❌ No advantage for sim-to-real transfer")
    print("  ❌ Just a different API to same data")

    print("\n✅ WITH NOISE in XML (Recommended):")
    print("  ✅ IMU sensors = Realistic, noisy measurements")
    print("  ✅ Matches actual BNO085/ICM45686 hardware")
    print("  ✅ Agent learns to handle measurement uncertainty")
    print("  ✅ Better sim-to-real transfer")
    print("  ✅ Same observation structure for sim and real robot")

    print("\n🎯 VALUE REALIZATION:")
    print("  1. Add sensor noise to wildrobot.xml:")
    print('     <gyro site="chest_imu" name="chest_imu_gyro"')
    print('           noise="0.0002" cutoff="100"/>')
    print()
    print("  2. Train with noisy IMU observations")
    print()
    print("  3. Deploy to real robot (zero code changes)")
    print()
    print("  4. Policy handles real sensor noise gracefully ✓")

    print("\n📚 RECOMMENDED READING:")
    print("  - add_sensor_noise_to_wildrobot.md")
    print("  - Search 'teacher-student' or 'privileged learning'")

    print("\n🔧 TO ADD NOISE:")
    print("  See: add_sensor_noise_to_wildrobot.md")
    print("  Edit: loco_mujoco/models/wildrobot/wildrobot.xml")
    print("  Add 'noise' and 'cutoff' attributes to <gyro>, <accelerometer>")

    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMU SENSOR VALUE ANALYSIS")
    print("="*80)
    print("\nThis script demonstrates when custom IMU sensors provide value")
    print("versus built-in observations like BodyVel and ProjectedGravityVector.")
    print("="*80)

    try:
        compare_without_noise()
        demonstrate_with_noise()
        show_value_proposition()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
