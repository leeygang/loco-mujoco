#!/usr/bin/env python
"""
Compare WildRobot vs UnitreeGo2 to find why WildRobot fails but Go2 works.
"""
from loco_mujoco import RLFactory
import jax

print("=" * 80)
print("Comparing WildRobot vs UnitreeGo2 - Finding the Difference")
print("=" * 80)

# Create both envs with same goal config
goal_params = {
    'goal_type': 'GoalRandomRootVelocity',
    'goal_params': {'visualize_goal': True, 'max_x_vel': 1.2, 'max_y_vel': 0.2, 'max_yaw_vel': 0.3},
    'headless': True,
    'reward_type': 'LocomotionReward',
    'terminal_state_type': 'HeightBasedTerminalStateHandler',
}

print("\n1. Creating both environments...")
wildrobot = RLFactory.make('MjxWildRobot', **goal_params)
print(f"   ✓ WildRobot created: {wildrobot.__class__.__name__}")

go2 = RLFactory.make('MjxUnitreeGo2', **goal_params)
print(f"   ✓ UnitreeGo2 created: {go2.__class__.__name__}")

print("\n2. Comparing observation containers...")
print(f"   WildRobot observations: {list(wildrobot.obs_container.keys())}")
print(f"   UnitreeGo2 observations: {list(go2.obs_container.keys())}")

# Check if GoalRandomRootVelocity is in both
wr_has_goal = 'GoalRandomRootVelocity' in wildrobot.obs_container
go2_has_goal = 'GoalRandomRootVelocity' in go2.obs_container
print(f"\n   WildRobot has GoalRandomRootVelocity: {wr_has_goal}")
print(f"   UnitreeGo2 has GoalRandomRootVelocity: {go2_has_goal}")

if wr_has_goal and go2_has_goal:
    wr_goal = wildrobot.obs_container['GoalRandomRootVelocity']
    go2_goal = go2.obs_container['GoalRandomRootVelocity']
    print(f"\n   WildRobot goal class: {wr_goal.__class__}")
    print(f"   UnitreeGo2 goal class: {go2_goal.__class__}")
    print(f"   Same class? {wr_goal.__class__ == go2_goal.__class__}")

print("\n3. Comparing class hierarchies...")
print(f"   WildRobot MRO: {[c.__name__ for c in wildrobot.__class__.__mro__]}")
print(f"   UnitreeGo2 MRO: {[c.__name__ for c in go2.__class__.__mro__]}")

print("\n4. Comparing reset methods...")
print(f"   WildRobot has 'reset': {hasattr(wildrobot, 'reset')}")
print(f"   WildRobot has 'mjx_reset': {hasattr(wildrobot, 'mjx_reset')}")
print(f"   UnitreeGo2 has 'reset': {hasattr(go2, 'reset')}")
print(f"   UnitreeGo2 has 'mjx_reset': {hasattr(go2, 'mjx_reset')}")

# Check which class defines reset
print(f"\n   WildRobot.reset defined in: {wildrobot.reset.__qualname__}")
print(f"   UnitreeGo2.reset defined in: {go2.reset.__qualname__}")

print("\n5. Testing single reset (non-vmapped)...")
try:
    key = jax.random.PRNGKey(0)
    obs_wr = wildrobot.reset(key)
    print(f"   ✓ WildRobot single reset works")
except Exception as e:
    print(f"   ✗ WildRobot single reset FAILED: {e}")

try:
    key = jax.random.PRNGKey(0)
    obs_go2 = go2.reset(key)
    print(f"   ✓ UnitreeGo2 single reset works")
except Exception as e:
    print(f"   ✗ UnitreeGo2 single reset FAILED: {e}")

print("\n6. Testing vmapped reset (THIS IS WHERE WILDROBOT FAILS)...")
try:
    keys = jax.random.split(jax.random.PRNGKey(42), 10)
    reset_fn = jax.vmap(wildrobot.reset)
    obs_batch = reset_fn(keys)
    print(f"   ✓ WildRobot vmapped reset works! Shape: {obs_batch.shape}")
except Exception as e:
    print(f"   ✗ WildRobot vmapped reset FAILED!")
    print(f"      Error type: {type(e).__name__}")
    print(f"      Error: {str(e)[:200]}")

try:
    keys = jax.random.split(jax.random.PRNGKey(42), 10)
    reset_fn = jax.vmap(go2.reset)
    obs_batch = reset_fn(keys)
    print(f"   ✓ UnitreeGo2 vmapped reset works! Shape: {obs_batch.shape}")
except Exception as e:
    print(f"   ✗ UnitreeGo2 vmapped reset FAILED!")
    print(f"      Error type: {type(e).__name__}")
    print(f"      Error: {str(e)[:200]}")

print("\n7. Checking parent classes and their reset implementations...")
# Check if either has a custom reset in their specific class file
import inspect

print(f"\n   WildRobot.reset source file: {inspect.getfile(wildrobot.reset)}")
print(f"   UnitreeGo2.reset source file: {inspect.getfile(go2.reset)}")

# Get the actual class that defines reset (not inherited)
for cls in wildrobot.__class__.__mro__:
    if 'reset' in cls.__dict__:
        print(f"   WildRobot.reset actually defined in: {cls.__name__}")
        break

for cls in go2.__class__.__mro__:
    if 'reset' in cls.__dict__:
        print(f"   UnitreeGo2.reset actually defined in: {cls.__name__}")
        break

print("\n8. Checking if either overrides mjx_reset...")
for cls in wildrobot.__class__.__mro__:
    if 'mjx_reset' in cls.__dict__:
        print(f"   WildRobot.mjx_reset defined in: {cls.__name__}")
        break
else:
    print(f"   WildRobot.mjx_reset: Not overridden")

for cls in go2.__class__.__mro__:
    if 'mjx_reset' in cls.__dict__:
        print(f"   UnitreeGo2.mjx_reset defined in: {cls.__name__}")
        break
else:
    print(f"   UnitreeGo2.mjx_reset: Not overridden")

print("\n" + "=" * 80)
print("Analysis Complete - Look for differences above!")
print("=" * 80)
