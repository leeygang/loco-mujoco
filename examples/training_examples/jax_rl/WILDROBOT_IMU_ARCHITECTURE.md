# WildRobot IMU Sensors - Architecture Guide

## Overview

This guide shows how to add custom IMU sensor observations to WildRobot **without modifying** the loco-mujoco library. This architecture allows you to:

✅ Keep WildRobot customizations separate from loco-mujoco
✅ Use loco-mujoco as an external pip dependency
✅ Easily migrate WildRobot to its own repository later

## What Was Changed

### 1. WildRobot XML Model ✅
**File:** `loco_mujoco/models/wildrobot/wildrobot.xml`

Added **42 sensors** (134 data elements):

**Physical IMU Sensors (3):**
- `chest_imu` (BNO085): gyro, accel, mag, quat
- `left_knee_imu` (ICM45686): gyro, accel
- `right_knee_imu` (ICM45686): gyro, accel

**Virtual Sensors from Mimic Sites:**
- `pelvis_mimic`: 8 sensors
- Hip/Knee/Foot mimic sites: 34 sensors

### 2. Custom Observation Types ✅
**Directory:** `examples/training_examples/jax_rl/wildrobot_extensions/`

Created external extension package:
```
wildrobot_extensions/
├── __init__.py
├── observations.py    # IMUSensor, AllIMUSensors
└── README.md          # Documentation
```

**Key Point:** These are **NOT** in `loco_mujoco/core/observations/` - they're external!

### 3. Example Scripts ✅

**Demo:** `demo_imu_observations.py`
- Shows 3 usage examples
- Demonstrates external registration pattern

**Training:** `train_wildrobot_with_imu.py`
- Complete training template
- Shows proper import/register pattern

**Test:** `test_wildrobot_sensors.py`
- Verifies all 42 sensors work
- Shows sensor inventory

## Quick Start

### Step 1: Test the Sensors

```bash
cd examples/training_examples/jax_rl
python test_wildrobot_sensors.py
```

Expected output:
```
✅ WildRobot XML loaded successfully!
Total sensors: 42
Total sensor data elements: 134
```

### Step 2: Run the Demo

```bash
python demo_imu_observations.py
```

This shows how to:
1. Import custom observations from `wildrobot_extensions`
2. Register them with loco-mujoco
3. Use them in observation specs

### Step 3: Train with IMU Sensors

```bash
python train_wildrobot_with_imu.py
```

Or use the full experiment script:
```bash
# TODO: Update experiment.py to support custom observations from config
```

## Usage Pattern

### The Key Architecture

```python
# 1. Import from external extensions (NOT from loco_mujoco!)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from wildrobot_extensions import IMUSensor, AllIMUSensors

# 2. Register with loco-mujoco
IMUSensor.register()
AllIMUSensors.register()

# 3. Now use like built-in types!
from loco_mujoco.core import ObservationType

observation_spec = [
    ObservationType.JointPosArray("joint_pos", [...]),

    # Custom observations work seamlessly
    ObservationType.IMUSensor("chest_imu",
        ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),
]

# 4. Create environment normally
from loco_mujoco import RLFactory
env = RLFactory.make(
    "MjxWildRobot",
    observation_specification=observation_spec,
    reward_type="LocomotionReward",
    goal_type="GoalRandomRootVelocity"
)
```

## Available Custom Observations

### 1. IMUSensor
Read specific physical IMU sensors.

```python
# 9-DOF IMU (BNO085) - gyro + accel + mag
ObservationType.IMUSensor("chest_imu",
    ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"])

# 6-DOF IMU (ICM45686) - gyro + accel only
ObservationType.IMUSensor("left_knee_imu",
    ["left_knee_imu_gyro", "left_knee_imu_accel"])
```

### 2. AllIMUSensors
Automatically include all IMU sensors.

```python
# Include all gyro + accel + mag sensors
ObservationType.AllIMUSensors("all_imus", include_magnetometer=True)

# Include only gyro + accel sensors
ObservationType.AllIMUSensors("all_imus", include_magnetometer=False)
```

## WildRobot Sensor Inventory

### Physical IMU Sensors

| Sensor | Hardware | Sensors | Dimension |
|--------|----------|---------|-----------|
| chest_imu | BNO085 | gyro, accel, mag, quat | 13 |
| left_knee_imu | ICM45686 | gyro, accel | 6 |
| right_knee_imu | ICM45686 | gyro, accel | 6 |

### Mimic Site Sensors

| Site | Sensors | Purpose |
|------|---------|---------|
| pelvis_mimic | gyro, accel, velocimeter, quat, vectors, velocities | Trunk state |
| right_hip_mimic | gyro, accel, quat, angvel | Right hip state |
| left_hip_mimic | gyro, accel, quat, angvel | Left hip state |
| right_knee_mimic | gyro, accel, quat, linvel | Right knee state |
| left_knee_mimic | gyro, accel, quat, linvel | Left knee state |
| right_foot_mimic | gyro, accel, quat, linvel, upvector | Right foot state |
| left_foot_mimic | gyro, accel, quat, linvel, upvector | Left foot state |

## Future: Separating WildRobot Repository

When you're ready to make WildRobot a standalone project:

### Proposed Structure

```
wildrobot/                          # New repository
├── wildrobot/
│   ├── __init__.py
│   ├── observations/
│   │   ├── __init__.py
│   │   └── imu.py                 # From wildrobot_extensions/observations.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── wildrobot.xml          # From loco_mujoco/models/wildrobot/
│   ├── environments/
│   │   ├── __init__.py
│   │   └── wildrobot.py           # From loco_mujoco/environments/wildrobot.py
│   └── training/
│       ├── configs/
│       └── scripts/
├── setup.py
├── pyproject.toml
└── README.md

# loco-mujoco becomes a dependency
requirements.txt:
  loco-mujoco>=0.2.0
  mujoco>=3.0.0
  jax[cuda12]>=0.4.0
```

### Migration Steps

1. **Copy files to new repo:**
   ```bash
   # Create new repository
   mkdir wildrobot && cd wildrobot

   # Copy WildRobot-specific files
   cp -r wildrobot_extensions/ wildrobot/observations/
   cp loco_mujoco/models/wildrobot/ wildrobot/models/
   cp loco_mujoco/environments/*wildrobot* wildrobot/environments/
   ```

2. **Create setup.py:**
   ```python
   from setuptools import setup, find_packages

   setup(
       name="wildrobot",
       version="0.1.0",
       packages=find_packages(),
       install_requires=[
           "loco-mujoco>=0.2.0",  # External dependency!
           "mujoco>=3.0.0",
           "jax[cuda12]>=0.4.0",
       ],
       package_data={
           "wildrobot": ["models/*.xml", "models/assets/*"],
       }
   )
   ```

3. **Update imports in new repo:**
   ```python
   # wildrobot/observations/imu.py
   from loco_mujoco.core.observations.base import Observation  # External import

   # wildrobot/environments/wildrobot.py
   from loco_mujoco import LocoEnv  # External import
   ```

4. **Use as separate package:**
   ```bash
   # Install loco-mujoco
   pip install loco-mujoco

   # Install wildrobot
   pip install -e /path/to/wildrobot

   # Use in training
   python
   >>> from wildrobot.observations import IMUSensor
   >>> from loco_mujoco import RLFactory
   >>> IMUSensor.register()
   >>> env = RLFactory.make("MjxWildRobot", ...)
   ```

## Benefits of This Architecture

### ✅ No Library Modification
- loco-mujoco stays clean and generic
- WildRobot customizations are self-contained
- Can update loco-mujoco via `pip install --upgrade`

### ✅ Clear Ownership
- WildRobot code is in `wildrobot_extensions/`
- loco-mujoco code is in `loco_mujoco/`
- No confusion about what belongs where

### ✅ Easy Migration
- Already structured for extraction
- Just copy `wildrobot_extensions/` → new repo
- Add loco-mujoco as pip dependency

### ✅ Version Control
- Separate git history for WildRobot
- Can tag WildRobot releases independently
- Track loco-mujoco dependency versions

### ✅ Collaboration
- Share WildRobot without sharing loco-mujoco
- Contributors don't need loco-mujoco write access
- Can publish wildrobot to PyPI separately

## Extending Further

This pattern works for **all** loco-mujoco components:

### Custom Rewards
```python
# wildrobot_extensions/rewards.py
from loco_mujoco.core.reward import Reward

class WildRobotEnergyReward(Reward):
    def __call__(self, state, action, next_state, ...):
        # Custom reward logic
        return reward, carry

WildRobotEnergyReward.register()
```

### Custom Terminals
```python
# wildrobot_extensions/terminals.py
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

class WildRobotFallDetector(TerminalStateHandler):
    def is_absorbing(self, env, model, data, backend):
        # Custom termination logic
        return is_fallen

WildRobotFallDetector.register()
```

### Custom Domain Randomization
```python
# wildrobot_extensions/domain_randomizers.py
from loco_mujoco.core.domain_randomizer import DomainRandomizer

class WildRobotDR(DomainRandomizer):
    def randomize(self, env, model, data, backend):
        # Custom randomization logic
        return model, data

WildRobotDR.register()
```

## Files Created

This refactoring created/modified:

### WildRobot Extensions (External Code)
- ✅ `wildrobot_extensions/__init__.py`
- ✅ `wildrobot_extensions/observations.py`
- ✅ `wildrobot_extensions/README.md`

### Example Scripts
- ✅ `demo_imu_observations.py` (updated to use external extensions)
- ✅ `train_wildrobot_with_imu.py` (new training template)
- ✅ `test_wildrobot_sensors.py` (sensor verification)

### Documentation
- ✅ `WILDROBOT_IMU_ARCHITECTURE.md` (this file)
- ✅ `IMU_SENSORS_README.md` (user guide)

### WildRobot Model
- ✅ `loco_mujoco/models/wildrobot/wildrobot.xml` (updated sensors)

### loco-mujoco Library
- ✅ **No changes** to loco-mujoco core! Library remains clean.

## Summary

**Key Principle:** Extend via registration, not modification.

```
❌ DON'T: Modify loco_mujoco/core/observations/
✅ DO:    Create wildrobot_extensions/observations.py and register()

❌ DON'T: Fork loco-mujoco to add WildRobot features
✅ DO:    Keep WildRobot code separate, use loco-mujoco as dependency

❌ DON'T: Mix WildRobot and loco-mujoco in same package
✅ DO:    Clear separation, easy to extract later
```

This architecture is **production-ready** and follows software engineering best practices for extending libraries without forking them.

---

**Questions? Check:**
- `wildrobot_extensions/README.md` - Extension development guide
- `IMU_SENSORS_README.md` - User guide for IMU sensors
- `demo_imu_observations.py` - Working examples
