#!/usr/bin/env python
"""
Debug script to diagnose WildRobot training issues.
Run this on your remote machine to identify why the robot doesn't walk.
"""
import jax
import jax.numpy as jnp
from loco_mujoco import RLFactory

print("=" * 80)
print("WildRobot Debug Script")
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

print("\n1. Creating environment...")
try:
    env = RLFactory.make(**env_params)
    print("   ✓ Environment created successfully")
except Exception as e:
    print(f"   ✗ FAILED to create environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n2. Environment info:")
print(f"   - Observation space: {env.info.observation_space.shape}")
print(f"   - Action space: {env.info.action_space.shape}")
print(f"   - Observation container keys: {list(env.obs_container.keys())}")
print(f"   - Root body name: {env.root_body_name}")
print(f"   - Root height healthy range: {env.root_height_healthy_range}")

print(f"\n3. Testing single environment reset...")
key = jax.random.PRNGKey(0)
try:
    state = env.reset(key)
    print(f"   ✓ Reset successful")
    print(f"   - State shape: {state.shape}")
    print(f"   - State contains NaN: {jnp.isnan(state).any()}")
    print(f"   - State min/max: [{float(jnp.min(state)):.4f}, {float(jnp.max(state)):.4f}]")
except Exception as e:
    print(f"   ✗ FAILED to reset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n4. Testing single step with random action...")
try:
    action = jax.random.uniform(jax.random.PRNGKey(1), shape=(env.info.action_space.shape[0],),
                                minval=-1.0, maxval=1.0)
    next_state, reward, done, truncated, info = env.step(action)
    print(f"   ✓ Step successful")
    print(f"   - Next state shape: {next_state.shape}")
    print(f"   - Next state contains NaN: {jnp.isnan(next_state).any()}")
    print(f"   - Reward: {float(reward)}")
    print(f"   - Reward is NaN: {jnp.isnan(reward)}")
    print(f"   - Done: {done}")
    print(f"   - Truncated: {truncated}")

    # Check internal state
    if hasattr(env, '_data'):
        print(f"   - Robot height (qpos[2]): {float(env._data.qpos[2]):.4f}")
        print(f"   - Robot qvel: {env._data.qvel[:6]}")

except Exception as e:
    print(f"   ✗ FAILED to step: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n5. Testing 10 steps with random actions...")
state = env.reset(jax.random.PRNGKey(2))
episode_rewards = []
episode_done = False
for i in range(10):
    action = jax.random.uniform(jax.random.PRNGKey(100+i), shape=(env.info.action_space.shape[0],),
                                minval=-1.0, maxval=1.0)
    state, reward, done, truncated, info = env.step(action)
    episode_rewards.append(float(reward))
    if done or truncated:
        print(f"   ! Episode ended at step {i+1}")
        episode_done = True
        break

print(f"   - Episode completed {len(episode_rewards)} steps")
print(f"   - Rewards: {episode_rewards}")
print(f"   - Any NaN rewards: {any(jnp.isnan(r).item() if hasattr(r, 'item') else False for r in episode_rewards)}")
print(f"   - Episode terminated early: {episode_done}")

print(f"\n6. Testing vectorized environment (512 envs)...")
try:
    # Create mjx environment with multiple parallel envs
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(42), 512))
    print(f"   ✓ Vectorized reset successful")
    print(f"   - States shape: {states.shape}")
    print(f"   - States contain NaN: {jnp.isnan(states).any()}")

    # Take one step
    actions = jax.random.uniform(jax.random.PRNGKey(99), shape=(512, env.info.action_space.shape[0]),
                                 minval=-1.0, maxval=1.0)
    step_fn = jax.vmap(env.step)
    next_states, rewards, dones, truncateds, infos = step_fn(actions)

    print(f"   ✓ Vectorized step successful")
    print(f"   - Rewards shape: {rewards.shape}")
    print(f"   - Rewards contain NaN: {jnp.isnan(rewards).any()}")
    print(f"   - Num NaN rewards: {int(jnp.isnan(rewards).sum())}")
    print(f"   - Reward stats: min={float(jnp.nanmin(rewards)):.4f}, max={float(jnp.nanmax(rewards)):.4f}, mean={float(jnp.nanmean(rewards)):.4f}")
    print(f"   - Dones: {int(dones.sum())} / {len(dones)}")
    print(f"   - Truncated: {int(truncateds.sum())} / {len(truncateds)}")

except Exception as e:
    print(f"   ✗ FAILED vectorized test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("Debug script completed!")
print("=" * 80)
