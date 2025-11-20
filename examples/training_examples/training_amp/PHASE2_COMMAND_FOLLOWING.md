# Phase 2: Command Following - Custom Goal Type

This shows how to create discrete command support for WildRobot.

## Create Custom Command Goal

Create `wildrobot_extensions/goals.py`:

```python
from typing import Any, Dict
from types import ModuleType
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.observations.goals import Goal


@struct.dataclass
class GoalDiscreteCommandState:
    """State for discrete command goal."""
    command_idx: int  # Current command index
    goal_vel_x: float
    goal_vel_y: float
    goal_vel_yaw: float


class GoalDiscreteCommand(Goal):
    """
    Goal that samples from discrete command set.

    Commands:
    - 0: stop (0, 0, 0)
    - 1: walk_slow (0.5, 0, 0)
    - 2: walk (1.0, 0, 0)
    - 3: walk_fast (2.0, 0, 0)
    - 4: turn_left (0.5, 0, +0.5)
    - 5: turn_right (0.5, 0, -0.5)
    """

    # Command definitions [forward_vel, lateral_vel, angular_vel]
    COMMANDS = np.array([
        [0.0, 0.0, 0.0],   # 0: stop
        [0.5, 0.0, 0.0],   # 1: walk_slow
        [1.0, 0.0, 0.0],   # 2: walk
        [2.0, 0.0, 0.0],   # 3: walk_fast
        [0.5, 0.0, +0.5],  # 4: turn_left
        [0.5, 0.0, -0.5],  # 5: turn_right
    ])

    COMMAND_NAMES = ["stop", "walk_slow", "walk", "walk_fast", "turn_left", "turn_right"]

    def __init__(self, env: Any, **kwargs):
        super().__init__(env, **kwargs)
        self._obs_dim = 3  # [vel_x, vel_y, vel_yaw]

    def init_state(self, env: Any, key: Any, model, data, backend: ModuleType):
        """Initialize with stop command."""
        return GoalDiscreteCommandState(
            command_idx=0,
            goal_vel_x=0.0,
            goal_vel_y=0.0,
            goal_vel_yaw=0.0
        )

    def reset_state(self, env: Any, model, data, carry: Any, backend: ModuleType):
        """Sample random command on reset."""
        if backend == np:
            # NumPy: random choice
            command_idx = np.random.randint(0, len(self.COMMANDS))
        else:
            # JAX: use RNG from carry
            key = carry.key
            key, subkey = backend.random.split(key)
            command_idx = backend.random.randint(subkey, (), 0, len(self.COMMANDS))
            carry = carry.replace(key=key)

        command_vel = self.COMMANDS[command_idx]

        state = GoalDiscreteCommandState(
            command_idx=int(command_idx),
            goal_vel_x=float(command_vel[0]),
            goal_vel_y=float(command_vel[1]),
            goal_vel_yaw=float(command_vel[2])
        )

        return state, carry

    def __call__(self, state, model: Model, data: Data, carry: Any, backend: ModuleType):
        """Return current command velocity as observation."""
        goal_state = getattr(carry.observation_states, self._obs_name)

        obs = backend.array([
            goal_state.goal_vel_x,
            goal_state.goal_vel_y,
            goal_state.goal_vel_yaw
        ])

        return obs


# Register the goal
GoalDiscreteCommand.register()
```

## Register in `__init__.py`

Update `wildrobot_extensions/__init__.py`:

```python
from .observations import IMUSensor, AllIMUSensors
from .goals import GoalDiscreteCommand

__all__ = ['IMUSensor', 'AllIMUSensors', 'GoalDiscreteCommand']
```

## Use in Config

Then update your config to use it:

```yaml
experiment:
  env_params:
    goal_type: GoalDiscreteCommand
    goal_params: {}
```

## Training

```bash
# Register custom goal
cd training_amp
python -c "import sys; sys.path.insert(0, '.'); from wildrobot_extensions import GoalDiscreteCommand"

# Train with discrete commands
uv run python experiment.py --config-name=conf_wildrobot_amp_phase2_commands
```

## Expected Behavior

The robot will:
- ✅ Start from standing (if standing motions in dataset)
- ✅ Receive random commands each episode
- ✅ Learn to: stop, walk slow/fast, turn left/right
- ✅ Maintain human-like motion (from discriminator)

## Test Specific Command

After training, you can test specific commands:

```python
from loco_mujoco.algorithms import AMPJax
from loco_mujoco import ImitationFactory

# Load trained policy
agent_conf, agent_state = AMPJax.load_agent("outputs/.../AMPJax_saved.pkl")

# Create env
env = ImitationFactory.make("MjxWildRobot", ...)

# Override goal to test specific command
env.goal_state = GoalDiscreteCommandState(
    command_idx=3,  # walk_fast
    goal_vel_x=2.0,
    goal_vel_y=0.0,
    goal_vel_yaw=0.0
)

# Run and watch it walk fast!
AMPJax.play_policy(env, agent_conf, agent_state, ...)
```
