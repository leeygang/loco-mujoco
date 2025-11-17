#!/usr/bin/env python
"""
Check what LogWrapper.env.reset() actually returns
"""
import jax
from loco_mujoco import RLFactory
from loco_mujoco.core.wrappers import LogWrapper

print("=" * 80)
print("Checking what LogWrapper sees")
print("=" * 80)

# Create wrapped envs
wildrobot = RLFactory.make('MjxWildRobot',
                           goal_type='GoalRandomRootVelocity',
                           headless=True,
                           reward_type='LocomotionReward')

go2 = RLFactory.make('MjxUnitreeGo2',
                     goal_type='GoalRandomRootVelocity',
                     headless=True,
                     reward_type='LocomotionReward')

# Wrap with LogWrapper
wr_wrapped = LogWrapper(wildrobot)
go2_wrapped = LogWrapper(go2)

print("\n1. Checking LogWrapper.env attribute...")
print(f"   WildRobot LogWrapper.env type: {type(wr_wrapped.env)}")
print(f"   UnitreeGo2 LogWrapper.env type: {type(go2_wrapped.env)}")
print(f"   Are they the same as original? WR={wr_wrapped.env is wildrobot}, Go2={go2_wrapped.env is go2}")

print("\n2. Checking if wrapped env has different reset...")
print(f"   WildRobot wrapped.env.reset: {wr_wrapped.env.reset}")
print(f"   UnitreeGo2 wrapped.env.reset: {go2_wrapped.env.reset}")

print("\n3. Testing what wrapped.env.reset() returns...")
key = jax.random.PRNGKey(0)

try:
    wr_result = wr_wrapped.env.reset(key)
    print(f"   WildRobot wrapped.env.reset() returns: {type(wr_result)}")
    if isinstance(wr_result, tuple):
        print(f"      - Tuple of {len(wr_result)} elements")
    else:
        print(f"      - Single {type(wr_result).__name__}")
except Exception as e:
    print(f"   ✗ WildRobot wrapped.env.reset() FAILED: {e}")

try:
    go2_result = go2_wrapped.env.reset(key)
    print(f"   UnitreeGo2 wrapped.env.reset() returns: {type(go2_result)}")
    if isinstance(go2_result, tuple):
        print(f"      - Tuple of {len(go2_result)} elements")
    else:
        print(f"      - Single {type(go2_result).__name__}")
except Exception as e:
    print(f"   ✗ UnitreeGo2 wrapped.env.reset() FAILED: {e}")

print("\n4. Checking if envs have mjx_reset attribute...")
print(f"   WildRobot has mjx_reset: {hasattr(wildrobot, 'mjx_reset')}")
print(f"   UnitreeGo2 has mjx_reset: {hasattr(go2, 'mjx_reset')}")
print(f"   WildRobot wrapped.env has mjx_reset: {hasattr(wr_wrapped.env, 'mjx_reset')}")
print(f"   UnitreeGo2 wrapped.env has mjx_reset: {hasattr(go2_wrapped.env, 'mjx_reset')}")

print("\n" + "=" * 80)
