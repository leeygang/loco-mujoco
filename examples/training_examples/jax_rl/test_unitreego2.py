#!/usr/bin/env python
"""
Test if MjxUnitreeGo2 has the same issue as MjxWildRobot.
This will tell us if the issue is WildRobot-specific or affects all MJX environments.
"""
import jax
import jax.numpy as jnp
from loco_mujoco import RLFactory
from loco_mujoco.core.wrappers import VecEnv, LogWrapper

print("=" * 80)
print("Testing MjxUnitreeGo2 (the default working model)")
print("=" * 80)

# Configuration from conf.yaml (the default that supposedly works)
env_params = {
    'env_name': 'MjxUnitreeGo2',
    'horizon': 1000,
    'terminal_state_type': 'HeightBasedTerminalStateHandler',
    'goal_type': 'GoalRandomRootVelocity',
    'goal_params': {'visualize_goal': True},
    'headless': True,
    'reward_type': 'LocomotionReward',
    'reward_params': {
        'tracking_w_exp_xy': 4.0,
        'tracking_w_exp_yaw': 4.0,
        'tracking_w_sum_xy': 2.0,
        'tracking_w_sum_yaw': 1.0,
        'air_time_coeff': 0.1,
        'joint_acc_coeff': 2.0e-05,
        'air_time_max': 0.5,
        'joint_torque_coeff': 2.0e-07,
        'joint_position_limit_coeff': 2.0,
        'action_rate_coeff': 0.1,
        'symmetry_air_coeff': 0.005,
        'energy_coeff': 5.0e-05,
    }
}

print("\n1. Creating base environment...")
try:
    env = RLFactory.make(**env_params)
    print(f"   ✓ Environment created: {env.__class__.__name__}")
    print(f"   - MJX enabled: {env.mjx_enabled}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n2. Wrapping with VecEnv (like PPO does)...")
env = LogWrapper(env)
env = VecEnv(env)
print("   ✓ Environment wrapped")

print("\n3. Testing vectorized reset (512 parallel environments)...")
try:
    reset_keys = jax.random.split(jax.random.PRNGKey(42), 512)
    obs, states = env.reset(reset_keys)
    print("   ✓ Vectorized reset successful!")
    print(f"   - Observations shape: {obs.shape}")
except Exception as e:
    print(f"   ✗ FAILED with TracerArrayConversionError (SAME BUG AS WILDROBOT!)")
    print(f"   Error: {e}")
    # Don't print full traceback, we know what it is
    exit(1)

print("\n4. Testing vectorized step...")
try:
    actions = jax.random.uniform(jax.random.PRNGKey(99), shape=(512, env.env.info.action_space.shape[0]),
                                 minval=-1.0, maxval=1.0)
    next_obs, rewards, dones, truncated, infos, next_states = env.step(states, actions)
    print("   ✓ Vectorized step successful!")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

print("\n" + "=" * 80)
print("CONCLUSION: MjxUnitreeGo2 works WITHOUT the fix!")
print("This means the bug is specific to WildRobot, not all MJX environments.")
print("=" * 80)
