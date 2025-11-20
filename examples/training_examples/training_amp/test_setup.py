#!/usr/bin/env python3
"""
Quick test script to verify the training_amp setup is working.

This script:
1. Registers custom observations
2. Creates the WildRobot environment
3. Runs a few steps to verify everything works
"""

import sys
from pathlib import Path

# Add current directory to path for wildrobot_extensions
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("TRAINING_AMP SETUP VERIFICATION")
print("="*80)

# Test 1: Import and register custom observations
print("\n1. Testing custom observation imports...")
try:
    from wildrobot_extensions import IMUSensor, AllIMUSensors
    IMUSensor.register()
    AllIMUSensors.register()
    print("   ✅ Custom observations imported and registered")
except Exception as e:
    print(f"   ❌ Failed to import custom observations: {e}")
    sys.exit(1)

# Test 2: Import loco-mujoco
print("\n2. Testing loco-mujoco imports...")
try:
    from loco_mujoco import ImitationFactory, DefaultDatasetConf
    from loco_mujoco.core import ObservationType
    from loco_mujoco.algorithms import AMPJax
    print("   ✅ loco-mujoco imports successful")
except Exception as e:
    print(f"   ❌ Failed to import loco-mujoco: {e}")
    sys.exit(1)

# Test 3: Create observation spec
print("\n3. Creating observation specification...")
try:
    observation_spec = [
        ObservationType.IMUSensor("chest_imu",
            ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),
        ObservationType.IMUSensor("left_knee_imu",
            ["left_knee_imu_gyro", "left_knee_imu_accel"]),
        ObservationType.JointPosArray("joint_pos", [
            "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
            "right_ankle_pitch", "right_foot_roll",
            "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
            "left_ankle_pitch", "left_foot_roll",
            "waist_yaw"
        ]),
        ObservationType.LastAction("last_action"),
    ]
    print("   ✅ Observation spec created")
except Exception as e:
    print(f"   ❌ Failed to create observation spec: {e}")
    sys.exit(1)

# Test 4: Create environment (without datasets first)
print("\n4. Creating WildRobot environment (without mocap)...")
try:
    from loco_mujoco import RLFactory
    env = RLFactory.make(
        "WildRobot",
        observation_specification=observation_spec,
        reward_type="LocomotionReward",
        goal_type="GoalRandomRootVelocity"
    )
    print("   ✅ Environment created")
    print(f"   Observation dim: {env.observation_space.shape[0]}")
    print(f"   Action dim: {env.action_space.shape[0]}")
except Exception as e:
    print(f"   ❌ Failed to create environment: {e}")
    sys.exit(1)

# Test 5: Run a few steps
print("\n5. Running simulation steps...")
try:
    obs = env.reset()
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    # Test observation extraction
    chest_imu = env._get_from_obs(obs, "chest_imu")
    left_knee_imu = env._get_from_obs(obs, "left_knee_imu")

    print(f"   ✅ Simulation steps successful")
    print(f"   Chest IMU dim: {len(chest_imu)}")
    print(f"   Left knee IMU dim: {len(left_knee_imu)}")

    env.close()
except Exception as e:
    print(f"   ❌ Failed to run simulation: {e}")
    sys.exit(1)

# Test 6: Check for datasets (optional)
print("\n6. Checking for mocap datasets...")
try:
    import os
    cache_path = os.path.expanduser("~/.loco-mujoco-caches")
    if os.path.exists(cache_path):
        datasets = os.listdir(cache_path)
        print(f"   ✅ Found {len(datasets)} datasets in cache")
        print(f"   Cache path: {cache_path}")
    else:
        print(f"   ⚠️  No dataset cache found at {cache_path}")
        print(f"   Run: loco-mujoco-download")
except Exception as e:
    print(f"   ⚠️  Could not check datasets: {e}")

# Summary
print("\n" + "="*80)
print("SETUP VERIFICATION COMPLETE")
print("="*80)
print("""
✅ All core components working!

Next steps:
1. Download datasets (if not done):
   loco-mujoco-download

2. Run quick test:
   python experiment.py --config-name=conf_wildrobot_amp_phase1 num_updates=10

3. Start full training:
   python experiment.py --config-name=conf_wildrobot_amp_phase1
""")
print("="*80)
