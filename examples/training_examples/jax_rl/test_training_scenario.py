#!/usr/bin/env python
"""
Simulate the actual training scenario to see if WildRobot works
"""
import jax
import jax.numpy as jnp
from loco_mujoco import RLFactory
from loco_mujoco.core.wrappers import LogWrapper, VecEnv

print("=" * 80)
print("Testing ACTUAL training scenario (like PPO does)")
print("=" * 80)

# WildRobot config from conf_quickcheck.yaml
wr_params = {
    'env_name': 'MjxWildRobot',
    'horizon': 600,
    'terminal_state_type': 'HeightBasedTerminalStateHandler',
    'terminal_state_params': {'min_height': 0.0, 'max_height': 2.0},
    'goal_type': 'GoalRandomRootVelocity',
    'goal_params': {'visualize_goal': True, 'max_x_vel': 1.2, 'max_y_vel': 0.2, 'max_yaw_vel': 0.3},
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

print("\n1. Creating and wrapping WildRobot (like PPO._wrap_env does)...")
env = RLFactory.make(**wr_params)
env = LogWrapper(env)
env = VecEnv(env)
print(f"   ✓ Environment wrapped")
print(f"   - Unwrapped env type: {type(env.env.env.env)}")

NUM_ENVS = 64
print(f"\n2. Testing reset with {NUM_ENVS} parallel environments...")
try:
    reset_keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    obs, states = env.reset(reset_keys)
    print(f"   ✓ Reset successful!")
    print(f"   - Observations shape: {obs.shape}")
    print(f"   - States type: {type(states)}")
    print(f"   - Obs contains NaN: {jnp.isnan(obs).any()}")
except Exception as e:
    print(f"   ✗ Reset FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n3. Testing one step...")
try:
    actions = jax.random.uniform(jax.random.PRNGKey(1), shape=(NUM_ENVS, env.env.env.info.action_space.shape[0]),
                                 minval=-1.0, maxval=1.0)
    next_obs, rewards, dones, truncated, infos, next_states = env.step(states, actions)
    print(f"   ✓ Step successful!")
    print(f"   - Rewards shape: {rewards.shape}")
    print(f"   - Rewards: min={float(jnp.min(rewards)):.4f}, max={float(jnp.max(rewards)):.4f}, mean={float(jnp.mean(rewards)):.4f}")
    print(f"   - Rewards contain NaN: {jnp.isnan(rewards).any()}")
    print(f"   - Num done: {int(jnp.sum(dones))} / {NUM_ENVS}")
except Exception as e:
    print(f"   ✗ Step FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n4. Testing 10 steps to check for immediate termination...")
episode_lengths = []
episode_rewards = []

for step_num in range(10):
    actions = jax.random.uniform(jax.random.PRNGKey(100+step_num), shape=(NUM_ENVS, env.env.env.info.action_space.shape[0]),
                                 minval=-1.0, maxval=1.0)
    next_obs, rewards, dones, truncated, infos, next_states = env.step(next_states, actions)

    # Track episodes that finished
    if jnp.any(dones):
        done_indices = jnp.where(dones)[0]
        for idx in done_indices:
            ep_len = int(next_states.metrics.returned_episode_lengths[idx])
            ep_ret = float(next_states.metrics.returned_episode_returns[idx])
            episode_lengths.append(ep_len)
            episode_rewards.append(ep_ret)

print(f"   - Completed {len(episode_lengths)} episodes")
if episode_lengths:
    print(f"   - Episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={sum(episode_lengths)/len(episode_lengths):.1f}")
    print(f"   - Episode returns: min={min(episode_rewards):.2f}, max={max(episode_rewards):.2f}, mean={sum(episode_rewards)/len(episode_rewards):.2f}")

    # Check for the symptoms you reported
    if min(episode_lengths) <= 1:
        print(f"   ⚠️  WARNING: Episodes terminating immediately (length=1)!")
    if any(jnp.isnan(r) for r in episode_rewards):
        print(f"   ⚠️  WARNING: NaN episode returns detected!")
else:
    print(f"   - No episodes completed yet")

print("\n" + "=" * 80)
print("Test complete - If no warnings above, WildRobot works fine!")
print("=" * 80)
