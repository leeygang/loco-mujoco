# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocoMuJoCo is an imitation learning benchmark for whole-body locomotion control. It supports both CPU-based MuJoCo and GPU-accelerated MJX (MuJoCo in JAX), featuring 12 humanoid and 4 quadruped environments with 22,000+ motion capture datasets.

**Key Capabilities:**
- Dual backend: MuJoCo (CPU, single env) and MJX (GPU/JAX, parallel envs)
- JAX RL algorithms: PPO, GAIL, AMP, DeepMimic (single-file, JIT-compiled)
- Robot-to-robot retargeting for motion datasets
- Trajectory comparison metrics (DTW, discrete Fréchet distance) in JAX
- Modular component system for observations, rewards, terminals, domain randomization

## Common Commands

### Installation & Setup
```bash
# Development install
pip install -e .

# GPU support (JAX with CUDA)
pip install jax["cuda12"]

# Install dev dependencies (pytest, wandb)
pip install -e ".[dev]"

# MyoSkeleton environment (requires license acceptance)
loco-mujoco-myomodel-init
```

### Dataset Management
```bash
# Download all datasets (auto-downloads from HuggingFace on first use)
loco-mujoco-download

# Speed up loading with caches (stores forward kinematics results)
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

### Testing
```bash
# Run all tests
pytest

# CI command (excludes slow factory tests)
pytest --ignore=tests/test_task_factories.py

# Test specific component
pytest tests/test_observation.py
pytest tests/test_reward.py

# Tests are parameterized for "numpy" and "jax" backends - both are tested
```

### Build & Package
```bash
# Build distribution
make package

# Install from tarball
make install

# Clean build artifacts
make clean
```

### Running Examples
```bash
# Tutorials (numbered 00-11, demonstrate each feature)
cd examples/tutorials
python 01_creating_mujoco_env.py

# Training examples (PPO, GAIL, AMP with Hydra configs)
cd examples/training_examples/jax_rl
python experiment.py                                    # Train
python eval.py --path outputs/.../PPOJax_saved.pkl     # Evaluate
```

## High-Level Architecture

### 1. Dual Backend System

**Key Insight:** All code must be backend-agnostic to support both NumPy (CPU) and JAX (GPU).

```
Mujoco (mujoco_base.py)
    └─> Mjx (mujoco_mjx.py) - adds MJX support
            └─> LocoEnv (base.py) - adds trajectory handling
                    └─> Specific environments (e.g., UnitreeH1, MjxUnitreeH1)
```

- `Mujoco`: Base class for single-environment CPU simulation
- `Mjx`: Extends Mujoco for vectorized parallel JAX environments
- `LocoEnv`: Adds trajectory/dataset handling on top of Mjx
- Environments come in pairs: `UnitreeH1` (CPU) and `MjxUnitreeH1` (GPU)

**Critical Pattern:** All component methods receive `backend` parameter (np or jnp):
```python
def compute(self, data, backend):
    # Use backend, not hardcoded numpy/jax
    return backend.sqrt(backend.sum(data.qpos**2))
```

### 2. Factory Pattern (Primary Environment Creation Method)

**Never instantiate environments directly.** Use TaskFactory pattern:

```python
# Imitation learning with trajectories
from loco_mujoco import ImitationFactory, DefaultDatasetConf, LAFAN1DatasetConf
env = ImitationFactory.make("UnitreeH1",
    default_dataset_conf=DefaultDatasetConf(["walk", "run"]),
    lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))

# Pure RL without trajectories
from loco_mujoco import RLFactory
env = RLFactory.make("MjxUnitreeGo2",  # Mjx prefix for GPU
    reward_type="LocomotionReward",
    goal_type="GoalRandomRootVelocity")
```

Factories auto-configure components and handle dataset loading. See `loco_mujoco/task_factories/`.

### 3. Modular Component System

**Registration Pattern** (applies to all components):

```python
# 1. Inherit from base class
class CustomReward(Reward):
    def __call__(self, state, action, next_state, absorbing, info,
                 env, model, data, carry, backend):
        # Compute reward
        return reward_value, carry

# 2. Register (typically in __init__.py)
CustomReward.register()

# 3. Use by string name
env = RLFactory.make("UnitreeH1", reward_type="CustomReward")
```

**Component Types** (all in `loco_mujoco/core/`):
- **Observations** (`observations/`): `ObservationType` classes define state features
- **Rewards** (`reward/`): `Reward` base class, supports trajectory or goal-based
- **Terminal States** (`terminal_state_handler/`): `TerminalStateHandler` for episode termination
- **Initial States** (`initial_state_handler/`): `InitialStateHandler` for episode reset
- **Control Functions** (`control_functions/`): `ControlFunction` (e.g., PD control)
- **Domain Randomization** (`domain_randomizer/`): `DomainRandomizer` for sim-to-real
- **Terrain** (`terrain/`): `Terrain` for ground variations

**Component Lifecycle:**
1. `__init__()`: Setup during environment creation
2. `init_state()`: Initialize state structure (called in reset)
3. `reset()` or `reset_state()`: Reset state values
4. Main operation: `__call__()` for rewards, `is_absorbing()` for terminals, etc.

All components inherit from `StatefulObject` and store state in the carry pattern.

### 4. Carry Pattern (State Management)

Environments use a **carry pattern** for stateful data across steps:

```
AdditionalCarry (base):
    ├─ key (JAX random key)
    ├─ cur_step_in_episode
    ├─ last_action
    ├─ observation_states (dict: obs_name -> state)
    ├─ reward_state
    ├─ terminal_state
    └─ ...

LocoCarry (extends AdditionalCarry):
    └─ traj_state (trajectory tracking)
```

Each component stores/retrieves its state from carry. This enables JAX JIT compilation.

### 5. Trajectory System

**Key Files:** `loco_mujoco/trajectory/`
- `TrajectoryHandler`: Manages mocap data, interpolation, state tracking
- `Trajectory`: Container for `TrajectoryInfo`, `TrajectoryData`, `TrajectoryModel`
- Datasets auto-download from HuggingFace (`robfiras/loco-mujoco-datasets`)

**Critical Backend Conversion:**
```python
env.th.to_jax()    # REQUIRED before using MJX environments
env.th.to_numpy()  # For CPU environments
```

**Dataset Loading Order:**
1. Check cache at `LOCOMUJOCO_CONVERTED_DEFAULT_PATH`
2. Download from HuggingFace if not cached
3. Extend motion if incomplete via `extend_motion()`
4. Interpolate to environment frequency
5. Cache processed result

### 6. Observation Container Pattern

Observations use a container with named, grouped indices:

```python
from loco_mujoco.core import ObservationType

observation_spec = [
    ObservationType.JointPos("joint_pos", "knee_angle_l", "knee_angle_r"),
    ObservationType.BodyVel("body_vel", "pelvis"),
    ObservationType.Goal("goal_vel"),
]

# Access in code:
joint_positions = env._get_from_obs(obs, "joint_pos")
```

Each `ObservationType` registers its slice in the flattened observation vector.

### 7. Training with JAX Algorithms

Training scripts (`examples/training_examples/`) follow this pattern:

**experiment.py workflow:**
1. Hydra config loading (`conf.yaml`)
2. WandB initialization
3. Environment creation via factory
4. Expert dataset loading (for GAIL/AMP)
5. Agent initialization (`PPOJax.init_agent_conf()`)
6. Build training function (`PPOJax.build_train_fn()`)
7. JIT compile + vmap across seeds
8. Execute training in one JIT call
9. Log metrics to WandB
10. Save agent and record video

**Key Performance Features:**
- Entire training loop JIT-compiled as single function
- Vectorized across 2048+ parallel environments
- Optional vmap across multiple seeds
- Typical training: 75-100M steps in 7-20 minutes (RTX 3080 Ti)

## Project-Specific Conventions

### Environment Registration

New environments must:
1. Inherit from `LocoEnv`
2. Set `mjx_enabled = True` for MJX variants
3. Implement `_get_observation_specification()` and `_get_action_specification()` static methods
4. Define info properties: `root_body_name`, `root_free_joint_xml_name`, `root_height_healthy_range`
5. Call `.register()` in `loco_mujoco/environments/{humanoids,quadrupeds}/__init__.py`

### MJX-Specific Methods

MJX environments override methods with `mjx_` prefix:
- `mjx_reset()`, `mjx_step()` vs `reset()`, `step()`
- `mjx_is_done()`, `mjx_is_absorbing()` for episode logic
- `_mjx_simulation_post_step()` for post-step modifications

### XML Model Optimization for MJX

Override `_modify_spec_for_mjx()` to optimize for GPU:
```python
def _modify_spec_for_mjx(self, spec):
    # Replace meshes with primitives
    spec.body("foot_l").geom("foot_mesh").remove()
    spec.body("foot_l").add_geom(type="capsule", size=[0.05, 0.1])

    # Limit contact pairs explicitly
    spec.add_pair(spec.body("foot_l"), spec.body("ground"))

    # Reduce iterations
    spec.option.iterations = 2

    # Disable Euler damping for speed
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
```

### Info Properties

Use `@info_property` decorator (not regular `@property`) for environment metadata:
```python
@info_property
def root_body_name(self):
    return "pelvis"
```

## Common Pitfalls

1. **JAX/NumPy Mismatch**: Always pass and use `backend` parameter, never hardcode `np` or `jnp` in component logic
2. **Missing Registration**: Components won't be found if `.register()` not called in `__init__.py`
3. **Trajectory Backend**: MJX requires `env.th.to_jax()`, CPU requires `env.th.to_numpy()`
4. **Free Joint Indexing**: Quaternions are 4D in qpos but 3D in qvel - use `mj_jntname2qposid()` utilities
5. **Info Properties**: Must use `@info_property`, not `@property`, for environment metadata
6. **Direct Instantiation**: Always use factories, not direct class instantiation

## Key Files Reference

- `loco_mujoco/core/mujoco_base.py`: Base environment with observation/reward/terminal integration
- `loco_mujoco/core/mujoco_mjx.py`: MJX extension for parallel GPU environments
- `loco_mujoco/environments/base.py`: LocoEnv with trajectory support
- `loco_mujoco/task_factories/imitation_factory.py`: Dataset loading logic
- `loco_mujoco/trajectory/handler.py`: Trajectory state management
- `loco_mujoco/algorithms/ppo_jax.py`: JAX PPO implementation (single-file, JIT-compiled)
- `examples/tutorials/11_creating_custom_modules.py`: Complete custom component example

## Resources

- Documentation: https://loco-mujoco.readthedocs.io/
- Discord: https://discord.gg/gEqR3xCVdn
- GitHub: https://github.com/robfiras/loco-mujoco
- Examples demonstrate patterns better than docs - check `examples/tutorials/` first
