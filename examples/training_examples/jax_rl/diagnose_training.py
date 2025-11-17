#!/usr/bin/env python
"""
Comprehensive diagnostics for WildRobot training issue.
This will identify why Episode Length = 1 and returns are NaN.
"""
import jax
import jax.numpy as jnp
import numpy as np
from loco_mujoco import RLFactory
from loco_mujoco.core.wrappers import LogWrapper, VecEnv

print("=" * 80)
print("WildRobot Training Diagnostics")
print("=" * 80)

# Exact config from conf_quickcheck.yaml
config = {
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

print("\n1. Creating environment...")
env = RLFactory.make(**config)
print(f"   ✓ Base environment: {env.__class__.__name__}")
print(f"   - Horizon: {env.horizon}")
print(f"   - Root height healthy range: {env.root_height_healthy_range}")

# Wrap like training does
env = LogWrapper(env)
env = VecEnv(env)

NUM_ENVS = 512  # Same as conf_quickcheck
print(f"\n2. Testing initial reset with {NUM_ENVS} envs...")
reset_keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
obs, states = env.reset(reset_keys)

print(f"   ✓ Reset successful")
print(f"   - Observation shape: {obs.shape}")

# Check initial robot states from the MjxState
print(f"\n3. Checking initial robot configuration...")
# Access the underlying MjxState through the wrapper chain
mjx_states = states.env_state  # LogEnvState -> MjxState
initial_heights = mjx_states.data.qpos[:, 2]  # z position of root
print(f"   - Initial heights: min={float(jnp.min(initial_heights)):.4f}, max={float(jnp.max(initial_heights)):.4f}, mean={float(jnp.mean(initial_heights)):.4f}")
print(f"   - Height bounds: ({env.env.env.root_height_healthy_range[0]}, {env.env.env.root_height_healthy_range[1]})")

# Check if any robots are already outside bounds
below_min = jnp.sum(initial_heights < env.env.env.root_height_healthy_range[0])
above_max = jnp.sum(initial_heights > env.env.env.root_height_healthy_range[1])
print(f"   - Robots below min height: {int(below_min)} / {NUM_ENVS}")
print(f"   - Robots above max height: {int(above_max)} / {NUM_ENVS}")

if below_min > 0 or above_max > 0:
    print(f"   ⚠️  WARNING: Some robots start outside healthy height range!")

print(f"\n4. Testing with ZERO actions (robot should stand still)...")
zero_actions = jnp.zeros((NUM_ENVS, env.env.env.info.action_space.shape[0]))
obs1, rew1, done1, trunc1, info1, states1 = env.step(states, zero_actions)

print(f"   - Rewards: min={float(jnp.min(rew1)):.4f}, max={float(jnp.max(rew1)):.4f}, mean={float(jnp.mean(rew1)):.4f}")
print(f"   - NaN rewards: {int(jnp.sum(jnp.isnan(rew1)))}")
print(f"   - Done after 1 step: {int(jnp.sum(done1))} / {NUM_ENVS}")

if jnp.sum(done1) > NUM_ENVS * 0.5:
    print(f"   ⚠️  WARNING: More than 50% of episodes terminated after 1 step!")

    # Investigate why they're done
    done_indices = jnp.where(done1)[0][:5]  # Check first 5
    print(f"\n   Investigating first {len(done_indices)} terminated episodes:")
    for idx in done_indices:
        height = mjx_states.data.qpos[idx, 2]
        height_after = states1.env_state.data.qpos[idx, 2]
        print(f"     Env {idx}: height {float(height):.4f} → {float(height_after):.4f}, reward={float(rew1[idx]):.4f}")

print(f"\n5. Testing with RANDOM actions (init_std=0.8)...")
random_actions = jax.random.normal(jax.random.PRNGKey(42), shape=(NUM_ENVS, env.env.env.info.action_space.shape[0])) * 0.8
obs2, rew2, done2, trunc2, info2, states2 = env.step(states, random_actions)

print(f"   - Rewards: min={float(jnp.min(rew2)):.4f}, max={float(jnp.max(rew2)):.4f}, mean={float(jnp.mean(rew2)):.4f}")
print(f"   - NaN rewards: {int(jnp.sum(jnp.isnan(rew2)))}")
print(f"   - Done after 1 step: {int(jnp.sum(done2))} / {NUM_ENVS}")

print(f"\n6. Running 50 steps to collect episode statistics...")
states_rollout = states
episode_lengths = []
episode_returns = []
step_dones = []

for step in range(50):
    actions = jax.random.normal(jax.random.PRNGKey(100 + step),
                                shape=(NUM_ENVS, env.env.env.info.action_space.shape[0])) * 0.8
    obs_new, rew, done, trunc, info, states_rollout = env.step(states_rollout, actions)

    step_dones.append(int(jnp.sum(done)))

    # Collect finished episodes
    if jnp.any(done):
        done_mask = done
        ep_lens = states_rollout.metrics.returned_episode_lengths[done_mask]
        ep_rets = states_rollout.metrics.returned_episode_returns[done_mask]

        for ep_len, ep_ret in zip(ep_lens, ep_rets):
            episode_lengths.append(int(ep_len))
            episode_returns.append(float(ep_ret))

print(f"\n   Results after 50 steps:")
print(f"   - Total episodes completed: {len(episode_lengths)}")
print(f"   - Dones per step: {step_dones[:10]}...")  # First 10 steps

if episode_lengths:
    print(f"   - Episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
    print(f"   - Episode returns: min={min(episode_returns):.2f}, max={max(episode_returns):.2f}, mean={np.mean(episode_returns):.2f}")

    # Count episodes with length = 1
    immediate_term = sum(1 for l in episode_lengths if l == 1)
    print(f"   - Episodes with length = 1: {immediate_term} / {len(episode_lengths)} ({100*immediate_term/len(episode_lengths):.1f}%)")

    # Check for NaN returns
    nan_returns = sum(1 for r in episode_returns if np.isnan(r))
    if nan_returns > 0:
        print(f"   ⚠️  WARNING: {nan_returns} episodes have NaN returns!")

    if immediate_term > len(episode_lengths) * 0.5:
        print(f"\n   ⚠️  PROBLEM IDENTIFIED: More than 50% of episodes terminate immediately!")
        print(f"   This suggests:")
        print(f"   1. Terminal condition is too strict")
        print(f"   2. Initial robot pose is unstable")
        print(f"   3. Reward computation has issues")
else:
    print(f"   - No episodes completed in 50 steps (horizon={env.env.env.horizon})")

print(f"\n7. Checking reward components on first step...")
# Do one step and examine the reward
test_state = states
test_action = jnp.zeros((1, env.env.env.info.action_space.shape[0]))
single_env_idx = 0

# Get single env from batch
single_mjx_state = jax.tree_map(lambda x: x[single_env_idx:single_env_idx+1], states.env_state)

print(f"   Initial state qpos[:7]: {single_mjx_state.data.qpos[0, :7]}")
print(f"   Initial state qvel[:6]: {single_mjx_state.data.qvel[0, :6]}")

# Step single env
from loco_mujoco.core.wrappers.mjx import LocoMjxWrapper
base_env = env.env.env  # Get the unwrapped MjxWildRobot
wrapper = LocoMjxWrapper(base_env)
next_single_state = wrapper.env.mjx_step(single_mjx_state, test_action[0])

print(f"   After step reward: {float(next_single_state.reward):.4f}")
print(f"   After step done: {bool(next_single_state.done)}")
print(f"   After step height: {float(next_single_state.data.qpos[0, 2]):.4f}")

print("\n" + "=" * 80)
print("Diagnostics complete!")
print("=" * 80)
