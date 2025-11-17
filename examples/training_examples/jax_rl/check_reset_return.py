#!/usr/bin/env python
"""
Check what reset() actually returns for WildRobot vs UnitreeGo2
"""
import jax
from loco_mujoco import RLFactory

print("=" * 80)
print("Checking reset() return values")
print("=" * 80)

# Create both envs
wildrobot = RLFactory.make('MjxWildRobot',
                           goal_type='GoalRandomRootVelocity',
                           headless=True,
                           reward_type='LocomotionReward')

go2 = RLFactory.make('MjxUnitreeGo2',
                     goal_type='GoalRandomRootVelocity',
                     headless=True,
                     reward_type='LocomotionReward')

print("\n1. Testing direct reset() call...")
key = jax.random.PRNGKey(0)

wr_result = wildrobot.reset(key)
print(f"   WildRobot reset() returns: {type(wr_result)}")
if isinstance(wr_result, tuple):
    print(f"   - Tuple length: {len(wr_result)}")
    print(f"   - Element types: {[type(x).__name__ for x in wr_result]}")
else:
    print(f"   - Single value type: {type(wr_result).__name__}")
    if hasattr(wr_result, '__dataclass_fields__'):
        print(f"   - Dataclass fields: {list(wr_result.__dataclass_fields__.keys())}")

go2_result = go2.reset(key)
print(f"\n   UnitreeGo2 reset() returns: {type(go2_result)}")
if isinstance(go2_result, tuple):
    print(f"   - Tuple length: {len(go2_result)}")
    print(f"   - Element types: {[type(x).__name__ for x in go2_result]}")
else:
    print(f"   - Single value type: {type(go2_result).__name__}")
    if hasattr(go2_result, '__dataclass_fields__'):
        print(f"   - Dataclass fields: {list(go2_result.__dataclass_fields__.keys())}")

print("\n2. Testing mjx_reset() call...")
wr_mjx_result = wildrobot.mjx_reset(key)
print(f"   WildRobot mjx_reset() returns: {type(wr_mjx_result)}")
if hasattr(wr_mjx_result, '__dataclass_fields__'):
    print(f"   - Dataclass fields: {list(wr_mjx_result.__dataclass_fields__.keys())}")

go2_mjx_result = go2.mjx_reset(key)
print(f"\n   UnitreeGo2 mjx_reset() returns: {type(go2_mjx_result)}")
if hasattr(go2_mjx_result, '__dataclass_fields__'):
    print(f"   - Dataclass fields: {list(go2_mjx_result.__dataclass_fields__.keys())}")

print("\n3. Checking if MjxState unpacks like a tuple...")
# Try to unpack
try:
    obs, state = wr_mjx_result
    print(f"   ✓ WildRobot MjxState CAN unpack as (obs, state)")
    print(f"     - obs type: {type(obs)}")
    print(f"     - state type: {type(state)}")
except (TypeError, ValueError) as e:
    print(f"   ✗ WildRobot MjxState CANNOT unpack: {e}")

try:
    obs, state = go2_mjx_result
    print(f"   ✓ UnitreeGo2 MjxState CAN unpack as (obs, state)")
    print(f"     - obs type: {type(obs)}")
    print(f"     - state type: {type(state)}")
except (TypeError, ValueError) as e:
    print(f"   ✗ UnitreeGo2 MjxState CANNOT unpack: {e}")

print("\n" + "=" * 80)
