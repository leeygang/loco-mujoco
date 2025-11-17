#!/usr/bin/env python
"""
Test the fix for WildRobot MJX reset issue.
Run this to verify the fix works.
"""
import jax
import jax.numpy as jnp
from loco_mujoco import RLFactory
from loco_mujoco.core.wrappers import VecEnv, LogWrapper

print("=" * 80)
print("Testing WildRobot MJX Reset Fix")
print("=" * 80)

# Configuration from conf_quickcheck.yaml
env_params = {
    'env_name': 'MjxWildRobot',
    'horizon': 600,
    'terminal_state_type': 'HeightBasedTerminalStateHandler',
    'terminal_state_params': {
        'min_height': 0.0,
        'max_height': 2.0
    },
    'goal_type': 'GoalRandomRootVelocity',
    'goal_params': {
        'visualize_goal': True,
        'max_x_vel': 1.2,
        'max_y_vel': 0.2,
        'max_yaw_vel': 0.3
    },
    'headless': True,
    'reward_type': 'LocomotionReward',
    'reward_params': {
        'tracking_w_exp_xy': 6.0,
        'tracking_w_exp_yaw': 4.0,
        'tracking_w_sum_xy': 3.5,
        'tracking_w_sum_yaw': 1.0,
        'air_time_coeff': 0.1,
        'joint_acc_coeff': 2.0e-05,
        'air_time_max': 0.5,
        'joint_torque_coeff': 2.0e-07,
        'joint_position_limit_coeff': 2.0,
        'action_rate_coeff': 0.02,
        'symmetry_air_coeff': 0.005,
        'energy_coeff': 1.0e-05
    }
}

print("\n1. Creating base environment...")
env = RLFactory.make(**env_params)
print(f"   ✓ Environment created: {env.__class__.__name__}")
print(f"   - MJX enabled: {env.mjx_enabled}")

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
    print(f"   - States type: {type(states)}")
    print(f"   - Contains NaN: {jnp.isnan(obs).any()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n4. Testing vectorized step...")
try:
    actions = jax.random.uniform(jax.random.PRNGKey(99), shape=(512, env.env.info.action_space.shape[0]),
                                 minval=-1.0, maxval=1.0)
    next_obs, rewards, dones, truncated, infos, next_states = env.step(states, actions)
    print("   ✓ Vectorized step successful!")
    print(f"   - Rewards shape: {rewards.shape}")
    print(f"   - Rewards contain NaN: {jnp.isnan(rewards).any()}")
    print(f"   - Reward stats: min={float(jnp.nanmin(rewards)):.4f}, max={float(jnp.nanmax(rewards)):.4f}, mean={float(jnp.nanmean(rewards)):.4f}")
    print(f"   - Dones: {int(dones.sum())} / {len(dones)}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Testing JIT compilation...")
try:
    @jax.jit
    def test_reset(keys):
        return env.reset(keys)

    reset_keys = jax.random.split(jax.random.PRNGKey(123), 512)
    obs, states = test_reset(reset_keys)
    print("   ✓ JIT compilation successful!")
    print(f"   - Observations shape: {obs.shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED! The fix works correctly.")
print("=" * 80)
