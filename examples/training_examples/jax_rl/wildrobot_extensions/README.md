# WildRobot Custom Extensions

This directory contains WildRobot-specific code that **extends** loco-mujoco without modifying it. This architecture allows you to:

✅ Keep WildRobot code separate from loco-mujoco
✅ Use loco-mujoco as an external library
✅ Easily migrate WildRobot to its own repository

## Directory Structure

```
wildrobot_extensions/
├── __init__.py           # Package initialization
└── observations.py       # Custom IMU observation types

Future additions:
├── rewards.py           # Custom reward functions
├── terminals.py         # Custom termination conditions
└── domain_randomizers.py # Custom domain randomization
```

## Architecture Pattern

### ❌ Old Approach (Modifying loco-mujoco)
```python
# BAD: Adding code directly to loco-mujoco library
loco_mujoco/
  core/
    observations/
      imu.py  ← Added to library
```

### ✅ New Approach (External Extensions)
```python
# GOOD: Keep custom code separate
wildrobot_extensions/
  observations.py  ← Custom observations here

# Use loco-mujoco as external library
from loco_mujoco import RLFactory
from loco_mujoco.core import ObservationType

# Import and register custom extensions
from wildrobot_extensions import IMUSensor
IMUSensor.register()

# Now use like built-in types
observation_spec = [
    ObservationType.IMUSensor("chest_imu", [...])
]
```

## Usage

### 1. Import and Register Custom Observations

```python
import sys
from pathlib import Path

# Add wildrobot_extensions to path
sys.path.insert(0, str(Path(__file__).parent))

from wildrobot_extensions import IMUSensor, AllIMUSensors

# Register with loco-mujoco
IMUSensor.register()
AllIMUSensors.register()
```

### 2. Use in Observation Specification

```python
from loco_mujoco.core import ObservationType

observation_spec = [
    # Built-in loco-mujoco observations
    ObservationType.JointPosArray("joint_pos", [...]),
    ObservationType.JointVelArray("joint_vel", [...]),

    # Custom WildRobot observations
    ObservationType.IMUSensor("chest_imu",
        ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),
    ObservationType.IMUSensor("left_knee_imu",
        ["left_knee_imu_gyro", "left_knee_imu_accel"]),
    ObservationType.IMUSensor("right_knee_imu",
        ["right_knee_imu_gyro", "right_knee_imu_accel"]),
]
```

### 3. Create Environment

```python
from loco_mujoco import RLFactory

env = RLFactory.make(
    "MjxWildRobot",
    observation_specification=observation_spec,
    reward_type="LocomotionReward",
    goal_type="GoalRandomRootVelocity"
)
```

## Available Custom Observations

### IMUSensor
Read from specific physical IMU sensors by name.

```python
ObservationType.IMUSensor(
    obs_name="chest_imu",
    sensor_names=["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]
)
```

**Parameters:**
- `obs_name`: Name for this observation
- `sensor_names`: List of MuJoCo sensor names to read

**Example sensors:**
- `chest_imu_gyro` (3D)
- `chest_imu_accel` (3D)
- `chest_imu_mag` (3D)
- `left_knee_imu_gyro` (3D)
- `left_knee_imu_accel` (3D)
- `right_knee_imu_gyro` (3D)
- `right_knee_imu_accel` (3D)

### AllIMUSensors
Automatically include ALL IMU sensors in one observation.

```python
ObservationType.AllIMUSensors(
    obs_name="all_imus",
    include_magnetometer=True
)
```

**Parameters:**
- `obs_name`: Name for this observation
- `include_magnetometer`: Whether to include magnetometer sensors

## Example Scripts

### Quick Demo
```bash
cd examples/training_examples/jax_rl
python demo_imu_observations.py
```

Shows 3 examples:
1. Individual IMU sensors
2. All IMU sensors at once
3. MJX (GPU) compatibility

### Training Template
```bash
python train_wildrobot_with_imu.py
```

Complete training script showing:
- How to register custom observations
- How to create observation specs
- How to train with PPO

## Future: Separating WildRobot from loco-mujoco

When you're ready to move WildRobot to its own repository:

### 1. Create New Repository Structure

```
wildrobot/
├── wildrobot/
│   ├── __init__.py
│   ├── observations/      # From wildrobot_extensions/
│   │   └── imu.py
│   ├── models/            # From loco_mujoco/models/wildrobot/
│   │   └── wildrobot.xml
│   └── environments/      # From loco_mujoco/environments/
│       └── wildrobot.py
├── setup.py
└── requirements.txt       # loco-mujoco as dependency
```

### 2. Update setup.py

```python
setup(
    name="wildrobot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "loco-mujoco>=0.2.0",  # External dependency
        "mujoco>=3.0.0",
        "jax[cuda12]>=0.4.0",
    ]
)
```

### 3. Import Pattern

```python
# In your new wildrobot package
from loco_mujoco import RLFactory
from loco_mujoco.core import ObservationType

# Your custom observations
from wildrobot.observations import IMUSensor
IMUSensor.register()

# Use normally
env = RLFactory.make(
    "MjxWildRobot",
    observation_specification=[
        ObservationType.IMUSensor("chest_imu", [...])
    ]
)
```

## Adding More Custom Components

### Custom Rewards
```python
# wildrobot_extensions/rewards.py
from loco_mujoco.core.reward import Reward

class WildRobotForwardReward(Reward):
    def __call__(self, state, action, next_state, absorbing, info,
                 env, model, data, carry, backend):
        forward_vel = data.qvel[0]  # X velocity
        return forward_vel, carry

# Register
WildRobotForwardReward.register()
```

### Custom Terminals
```python
# wildrobot_extensions/terminals.py
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

class WildRobotTerminal(TerminalStateHandler):
    def is_absorbing(self, env, model, data, backend):
        height = data.qpos[2]
        return height < 0.3  # Terminate if too low

# Register
WildRobotTerminal.register()
```

## Benefits of This Architecture

### ✅ Modularity
- WildRobot code is self-contained
- loco-mujoco remains generic and reusable
- Clear separation of concerns

### ✅ Maintainability
- Update loco-mujoco independently via pip
- Version control WildRobot separately
- No merge conflicts between projects

### ✅ Portability
- Easy to share WildRobot with collaborators
- Can publish as separate pip package
- Works with any loco-mujoco version (with API compatibility)

### ✅ Testing
- Test WildRobot extensions independently
- Use stable loco-mujoco releases
- Easier CI/CD setup

## Questions?

This architecture follows best practices for extending libraries without forking them. The key principle is:

> **Extend via composition and registration, not by modification.**

All loco-mujoco components support registration, so you can add new:
- Observations (via `Observation.register()`)
- Rewards (via `Reward.register()`)
- Terminals (via `TerminalStateHandler.register()`)
- Goals (via `Goal.register()`)
- Domain randomizers (via `DomainRandomizer.register()`)

This keeps your custom code separate while fully integrating with loco-mujoco's features.
