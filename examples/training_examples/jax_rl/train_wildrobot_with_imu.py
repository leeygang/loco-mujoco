"""
Training script for WildRobot with custom IMU observations.

This demonstrates how to use external custom observation types that are
separate from the loco-mujoco library, allowing you to keep WildRobot code
independent and maintain loco-mujoco as an external dependency.

Usage:
    python train_wildrobot_with_imu.py
"""

import sys
from pathlib import Path

# Add wildrobot_extensions to Python path
sys.path.insert(0, str(Path(__file__).parent))

from loco_mujoco import RLFactory
from loco_mujoco.core import ObservationType
from loco_mujoco.algorithms import PPOJax
import wandb

# Import and register WildRobot custom observations
from wildrobot_extensions import IMUSensor, AllIMUSensors

# Register custom observation types
IMUSensor.register()
AllIMUSensors.register()


def create_observation_spec():
    """
    Create observation specification with IMU sensors.

    This shows the pattern for using custom observations that are
    external to loco-mujoco.
    """
    return [
        # Joint positions
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),

        # Joint velocities
        ObservationType.JointVelArray("joint_vel", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),

        # Physical IMU sensors (custom observations from wildrobot_extensions)
        ObservationType.IMUSensor("chest_imu",
            ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),
        ObservationType.IMUSensor("left_knee_imu",
            ["left_knee_imu_gyro", "left_knee_imu_accel"]),
        ObservationType.IMUSensor("right_knee_imu",
            ["right_knee_imu_gyro", "right_knee_imu_accel"]),

        # Gravity direction
        ObservationType.ProjectedGravityVector("gravity", "waist_freejoint"),

        # Last action
        ObservationType.LastAction("last_action"),
    ]


def main():
    """Main training function."""

    # Configuration
    config = {
        "env_name": "MjxWildRobot",
        "num_envs": 2048,
        "horizon": 600,
        "num_updates": 1000,
        "learning_rate": 3e-4,
        "reward_type": "LocomotionReward",
        "goal_type": "GoalRandomRootVelocity",
    }

    print("="*80)
    print("WILDROBOT TRAINING WITH CUSTOM IMU OBSERVATIONS")
    print("="*80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create observation specification
    observation_spec = create_observation_spec()

    print(f"\nObservation specification ({len(observation_spec)} types):")
    for obs in observation_spec:
        obs_type = obs.__class__.__name__
        is_custom = "✓ CUSTOM" if obs_type in ["IMUSensor", "AllIMUSensors"] else ""
        print(f"  - {obs.name:20s} ({obs_type:30s}) {is_custom}")

    # Create environment
    print("\n" + "="*80)
    print("Creating environment...")
    print("="*80)

    env = RLFactory.make(
        config["env_name"],
        observation_specification=observation_spec,
        reward_type=config["reward_type"],
        goal_type=config["goal_type"],
    )

    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape[0]}")
    print(f"  Action space: {env.action_space.shape[0]}")

    # Initialize wandb (optional)
    # run = wandb.init(
    #     project="wildrobot-imu",
    #     config=config,
    #     name="wildrobot_ppo_with_imu",
    # )

    # Initialize PPO agent
    print("\n" + "="*80)
    print("Initializing PPO agent...")
    print("="*80)

    agent_conf = PPOJax.init_agent_conf(
        env=env,
        num_envs=config["num_envs"],
        horizon=config["horizon"],
        init_std=0.8,
        actor_hidden_layer_sizes=[256, 256],
        critic_hidden_layer_sizes=[256, 256],
    )

    print("✓ Agent initialized")

    # Build and compile training function
    print("\n" + "="*80)
    print("Building training function (JIT compilation)...")
    print("="*80)

    train_fn = PPOJax.build_train_fn(
        env=env,
        agent_conf=agent_conf,
        num_envs=config["num_envs"],
        horizon=config["horizon"],
        num_updates=config["num_updates"],
        learning_rate=config["learning_rate"],
    )

    print("✓ Training function compiled")

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    # Note: Actual training would go here
    # agent_state, metrics = train_fn(...)

    print("\n✓ Training setup complete!")
    print("\nTo run full training, uncomment the training code and run:")
    print("  python train_wildrobot_with_imu.py")

    env.close()

    print("\n" + "="*80)
    print("Key Architectural Benefits:")
    print("="*80)
    print("✓ Custom observations (IMUSensor) are in wildrobot_extensions/")
    print("✓ loco-mujoco library remains unmodified")
    print("✓ WildRobot code can be easily separated into its own project")
    print("✓ loco-mujoco can be used as an external pip dependency")
    print("="*80)


if __name__ == "__main__":
    main()
