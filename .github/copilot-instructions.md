# LocoMuJoCo AI Coding Agent Instructions

## Project Overview
LocoMuJoCo is an **imitation learning benchmark** for whole-body locomotion control in MuJoCo. It supports both CPU-based MuJoCo and GPU-accelerated MJX (MuJoCo in JAX), featuring 12 humanoid and 4 quadruped environments with 22,000+ motion capture datasets.

## Core Architecture

### 1. Dual Backend System (CPU/GPU)
- **Mujoco (CPU)**: Base class in `loco_mujoco/core/mujoco_base.py` for single-environment simulation
- **Mjx (GPU/JAX)**: Extends Mujoco in `loco_mujoco/core/mujoco_mjx.py` for parallel vectorized environments
- **LocoEnv**: In `loco_mujoco/environments/base.py`, adds trajectory handling to Mjx
- All environments inherit from `LocoEnv` and can have both CPU and MJX variants (e.g., `UnitreeH1` and `MjxUnitreeH1`)

### 2. Factory Pattern for Environment Creation
Use **TaskFactory** pattern instead of direct instantiation:
```python
# Imitation learning with trajectories
from loco_mujoco import ImitationFactory, DefaultDatasetConf
env = ImitationFactory.make("UnitreeH1", 
    default_dataset_conf=DefaultDatasetConf(["walk", "run"]))

# Reinforcement learning without trajectories  
from loco_mujoco import RLFactory
env = RLFactory.make("UnitreeG1")
```
Factories are in `loco_mujoco/task_factories/` and auto-configure components.

### 3. Modular Component System
All components use a **registration pattern** - define, register, then reference by string name:

**Registration Pattern** (applies to all components):
```python
# 1. Inherit from base class
class CustomReward(Reward):
    def __call__(self, state, action, next_state, absorbing, info, 
                 env, model, data, carry, backend):
        return reward_value, carry

# 2. Register the component
CustomReward.register()

# 3. Use by string name in factory
env = RLFactory.make("UnitreeH1", reward_type="CustomReward")
```

**Core Component Types** (all in `loco_mujoco/core/`):
- **Observations** (`observations/`): Define state features via `ObservationType` classes
- **Rewards** (`reward/`): Base class `Reward`, supports trajectory-based or goal-based
- **Terminal States** (`terminal_state_handler/`): `TerminalStateHandler` subclasses define episode termination
- **Initial States** (`initial_state_handler/`): `InitialStateHandler` controls episode initialization
- **Control Functions** (`control_functions/`): `ControlFunction` for PD control, etc.
- **Domain Randomization** (`domain_randomizer/`): `DomainRandomizer` for sim-to-real transfer
- **Terrain** (`terrain/`): `Terrain` for ground variations

### 4. Trajectory System
**Key Files**: `loco_mujoco/trajectory/`
- `TrajectoryHandler`: Manages motion capture data, interpolation, and state tracking
- `Trajectory`: Contains `TrajectoryInfo`, `TrajectoryData`, `TrajectoryModel`
- Datasets auto-download from HuggingFace (`robfiras/loco-mujoco-datasets`)
- Support AMASS, LAFAN1, and custom trajectories

**Critical**: Trajectories must be converted to JAX format for MJX environments:
```python
env.th.to_jax()  # Before using MJX
env.th.to_numpy()  # For CPU environments
```

### 5. State Management with "Carry"
Environments use a **carry pattern** for stateful information across steps:
- `AdditionalCarry` (base): Contains `key`, `cur_step_in_episode`, `last_action`, states for all components
- `LocoCarry`: Adds `traj_state` for trajectory tracking
- Each component stores its state in carry (e.g., `carry.reward_state`, `carry.observation_states`)

## Development Workflows

### Testing
```bash
pytest                                    # Run all tests
pytest --ignore=tests/test_task_factories.py  # CI command
pytest tests/test_observation.py          # Specific test file
```
Tests are parameterized for both "numpy" and "jax" backends - always test both.

### Building & Installation
```bash
pip install -e .                          # Editable install (development)
make package                              # Build distribution
make install                              # Install from tarball
```

### Dataset Management
```bash
loco-mujoco-download                      # Download all datasets
loco-mujoco-set-all-caches --path <path>  # Cache processed datasets
loco-mujoco-myomodel-init                 # Setup MyoSkeleton (requires license)
```

### Running Examples
- **Tutorials**: `examples/tutorials/` (numbered 00-11) demonstrate each feature
- **Training**: `examples/training_examples/` has JAX RL algorithms (PPO, GAIL, AMP, DeepMimic)
- Install `wandb` for training examples: `pip install wandb`

## Project-Specific Conventions

### 1. Environment Registration
Environments must:
1. Inherit from `LocoEnv`
2. Set `mjx_enabled = True` for MJX variants
3. Implement `_get_observation_specification()` and `_get_action_specification()` static methods
4. Call `.register()` in `loco_mujoco/environments/{humanoids,quadrupeds}/__init__.py`
5. Define info properties: `root_body_name`, `root_free_joint_xml_name`, `root_height_healthy_range`

### 2. Backend Agnostic Code
Components must support both NumPy and JAX:
```python
def compute(self, data, backend):
    # backend is np or jnp - passed as parameter
    return backend.sqrt(backend.sum(data.qpos**2))
```
Use `backend` parameter, never import `numpy` or `jax.numpy` directly in component logic.

### 3. MJX-Specific Methods
MJX environments override methods with `mjx_` prefix:
- `mjx_reset()`, `mjx_step()` vs `reset()`, `step()`
- `mjx_is_done()`, `mjx_is_absorbing()` for episode logic
- `_mjx_simulation_post_step()` for post-step modifications

### 4. Observation Container Pattern
Observations use a container system with grouped indices:
```python
from loco_mujoco.core import ObservationType

observation_spec = [
    ObservationType.JointPos("joint_pos", "knee_angle_l"),
    ObservationType.BodyVel("body_vel", "pelvis"),
]
# Access: env._get_from_obs(obs, "joint_pos")
```

### 5. XML Model Modifications
For MJX optimization, modify specs in `_modify_spec_for_mjx()`:
- Replace complex meshes with primitives (capsules, boxes)
- Limit contact pairs explicitly with `spec.add_pair()`
- Reduce iterations: `spec.option.iterations = 2`
- Disable Euler damping: `spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP`

## Critical Implementation Details

### Dataset Loading Order
1. Check cache at `LOCOMUJOCO_CONVERTED_DEFAULT_PATH` (set via `loco-mujoco-set-all-caches`)
2. Download from HuggingFace if not cached
3. Extend motion if incomplete using `extend_motion()`
4. Interpolate via `TrajectoryHandler` to environment frequency
5. Cache processed result

### Component Lifecycle
Components follow this lifecycle:
1. `__init__()`: Setup (called during env creation)
2. `init_state()`: Initialize state structure (called in `reset()`)
3. `reset()` or `reset_state()`: Reset state values
4. Main operation: `__call__()` for rewards, `is_absorbing()` for terminals, etc.

### StatefulObject Pattern
Base class for all components (`loco_mujoco/core/stateful_object.py`):
- Provides `init_state()`, `reset_state()` interface
- Returns `EmptyState()` if no state needed
- Stores state in `carry.<component>_state`

## Common Pitfalls

1. **JAX/NumPy Mismatch**: Always pass and use `backend` parameter, never hardcode `np` or `jnp`
2. **Missing Registration**: Components won't be found if `.register()` not called in `__init__.py`
3. **Trajectory Backend**: MJX requires `env.th.to_jax()`, CPU requires `env.th.to_numpy()`
4. **Info Properties**: Use `@info_property` decorator, not regular properties, for environment metadata
5. **Free Joint Indexing**: Use `mj_jntname2qposid()` utilities, quaternions are 4D in qpos but 3D in qvel

## Key Files Reference

- `loco_mujoco/core/mujoco_base.py`: Base environment, observation/reward/terminal integration
- `loco_mujoco/environments/base.py`: LocoEnv with trajectory support
- `loco_mujoco/task_factories/imitation_factory.py`: Dataset loading logic
- `loco_mujoco/trajectory/handler.py`: Trajectory state management
- `examples/tutorials/11_creating_custom_modules.py`: Complete custom component example
- `pyproject.toml`: Dependencies and CLI script definitions

## Documentation
- Full docs: https://loco-mujoco.readthedocs.io/
- Discord: https://discord.gg/gEqR3xCVdn
- Examples demonstrate patterns better than docs - check `examples/tutorials/` first
