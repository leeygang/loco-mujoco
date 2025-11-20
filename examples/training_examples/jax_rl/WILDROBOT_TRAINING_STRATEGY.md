# Training Strategy for Command-Following WildRobot

## Your Requirements

1. ✅ Start from standing pose
2. ✅ Follow high-level commands: stop, walk slow, walk fast, turn left/right
3. ✅ Fall recovery (stand up and continue)
4. ✅ **Human-like motion** (most important!)
5. ✅ Deploy to real robot (sim-to-real)

## Recommended Approach: **AMP (Adversarial Motion Priors)**

### Why AMP is Best for Your Use Case

| Requirement | Pure RL (PPO) | GAIL | **AMP** | DeepMimic |
|-------------|---------------|------|---------|-----------|
| Human-like motion | ❌ Robotic | ✅ Good | ✅✅ **Excellent** | ✅ Good |
| High-level commands | ✅ Easy | ⚠️ Medium | ✅ **Easy** | ❌ Hard |
| Fall recovery | ✅ Easy | ⚠️ Medium | ✅ **Easy** | ❌ Hard |
| Training stability | ✅ Stable | ⚠️ Unstable | ✅ **Stable** | ✅ Stable |
| Sim-to-real | ⚠️ Medium | ✅ Good | ✅✅ **Excellent** | ✅ Good |
| Flexibility | ✅ High | ⚠️ Medium | ✅✅ **Very High** | ❌ Low |

**Verdict: AMP is the clear winner** ✅

### What is AMP?

**AMP (Adversarial Motion Priors)** combines:
1. **RL** for task learning (follow commands, balance, recover)
2. **Discriminator** for motion quality (looks human-like)
3. **Mocap data** as motion reference

```
RL Reward:        Follow command (reach target velocity)
   +
Style Reward:     Look human-like (discriminator score)
   =
Human-like motion while following commands ✓
```

### Key Advantages Over Other Methods

**vs Pure RL (PPO):**
- PPO alone → robotic, unnatural gaits
- AMP → learns natural human walking from mocap

**vs GAIL:**
- GAIL → less stable, harder to add task objectives
- AMP → more stable, easily combines task + style rewards

**vs DeepMimic:**
- DeepMimic → precise trajectory tracking, rigid
- AMP → learns motion style, flexible commands

## Training Strategy (4 Phases)

### Phase 1: Learn Basic Motions with AMP (2-3 days)

**Goal:** Learn human-like walking, running, turning from mocap

**Observation Spec:**
```python
observation_spec = [
    # Student observations (noisy, for real robot)
    ObservationType.IMUSensor("chest_imu",
        ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"],
        group="student"),
    ObservationType.IMUSensor("left_knee_imu",
        ["left_knee_imu_gyro", "left_knee_imu_accel"],
        group="student"),
    ObservationType.IMUSensor("right_knee_imu",
        ["right_knee_imu_gyro", "right_knee_imu_accel"],
        group="student"),

    # Teacher observations (perfect, for discriminator)
    ObservationType.RelativeSiteQuantaties("mimic_obs",
        group="teacher"),

    # Joint states
    ObservationType.JointPosArray("joint_pos", [...]),
    ObservationType.JointVelArray("joint_vel", [...]),

    # Goal (target velocity)
    ObservationType.GoalTrajMimic("goal"),

    # Last action
    ObservationType.LastAction("last_action"),
]
```

**Datasets to Use:**
```python
from loco_mujoco import ImitationFactory, DefaultDatasetConf, LAFAN1DatasetConf

env = ImitationFactory.make(
    "MjxWildRobot",
    observation_specification=observation_spec,

    # Load mocap datasets
    default_dataset_conf=DefaultDatasetConf([
        "walk",      # Normal walking
        "run",       # Running
        # More motions available in loco-mujoco
    ]),

    lafan1_dataset_conf=LAFAN1DatasetConf([
        "walk1_subject1",     # Walking variations
        "walk2_subject1",
        "run1_subject1",      # Running
        "turnRight_subject1", # Turning motions
        "turnLeft_subject1",
    ]),

    reward_type="AMPReward",  # AMP-style reward
    goal_type="GoalTrajMimic",
)
```

**Training Command:**
```bash
cd examples/training_examples/jax_rl
python experiment.py \
  --config-name=conf_wildrobot_amp \
  num_envs=2048 \
  num_updates=5000 \
  track=true \
  wandb_project_name="wildrobot-amp-phase1"
```

**Expected Results:**
- Robot learns smooth, human-like walking
- Can vary speed based on mocap
- Natural turning motions
- **Motion looks human!** ✓

### Phase 2: Add Command Conditioning (1-2 days)

**Goal:** Map discrete commands to continuous velocities

**Command Mapping:**
```python
COMMANDS = {
    "stop":       {"forward_vel": 0.0,  "lateral_vel": 0.0,  "angular_vel": 0.0},
    "walk_slow":  {"forward_vel": 0.5,  "lateral_vel": 0.0,  "angular_vel": 0.0},
    "walk":       {"forward_vel": 1.0,  "lateral_vel": 0.0,  "angular_vel": 0.0},
    "walk_fast":  {"forward_vel": 2.0,  "lateral_vel": 0.0,  "angular_vel": 0.0},
    "turn_left":  {"forward_vel": 0.5,  "lateral_vel": 0.0,  "angular_vel": +0.5},
    "turn_right": {"forward_vel": 0.5,  "lateral_vel": 0.0,  "angular_vel": -0.5},
}
```

**Implementation:**
Create custom goal type that samples from command distribution:

```python
# In wildrobot_extensions/goals.py
from loco_mujoco.core.observations.goals import Goal

class GoalCommandVelocity(Goal):
    """Goal that samples from discrete command set."""

    COMMANDS = {
        "stop":       [0.0, 0.0, 0.0],
        "walk_slow":  [0.5, 0.0, 0.0],
        "walk":       [1.0, 0.0, 0.0],
        "walk_fast":  [2.0, 0.0, 0.0],
        "turn_left":  [0.5, 0.0, +0.5],
        "turn_right": [0.5, 0.0, -0.5],
    }

    def reset_state(self, env, model, data, carry, backend):
        # Sample random command
        command = backend.choice(list(self.COMMANDS.values()))
        # Store in state
        return self._create_state(command)

GoalCommandVelocity.register()
```

**Update Training:**
```python
env = ImitationFactory.make(
    "MjxWildRobot",
    observation_specification=observation_spec,
    default_dataset_conf=DefaultDatasetConf(["walk", "run"]),
    reward_type="AMPReward",
    goal_type="GoalCommandVelocity",  # New command-based goal!
)
```

**Training:**
```bash
python experiment.py \
  --config-name=conf_wildrobot_amp_commands \
  num_updates=3000 \
  load_checkpoint=outputs/phase1/PPOJax_saved.pkl  # Continue from Phase 1
```

**Expected Results:**
- Robot responds to discrete commands
- Smooth transitions between speeds
- Maintains human-like motion ✓

### Phase 3: Add Fall Recovery (2-3 days)

**Goal:** Stand up after falling and continue

**Approach 1: Curriculum Learning (Recommended)**

Train in stages:
1. Start from standing → walk
2. Start from sitting → stand → walk
3. Start from lying → stand → walk
4. Random falls during walking → recover

**Implementation:**
```python
from loco_mujoco.core.initial_state_handler import InitialStateHandler

class FallRecoveryInitialState(InitialStateHandler):
    """Curriculum: Start from progressively harder poses."""

    def __init__(self, curriculum_stage=0):
        self.curriculum_stage = curriculum_stage

    def reset_state(self, env, model, data, carry, backend):
        if self.curriculum_stage == 0:
            # Stage 0: Normal standing
            return self._set_standing_pose(data)
        elif self.curriculum_stage == 1:
            # Stage 1: Start from crouching
            return self._set_crouching_pose(data)
        elif self.curriculum_stage == 2:
            # Stage 2: Start from sitting
            return self._set_sitting_pose(data)
        elif self.curriculum_stage == 3:
            # Stage 3: Start from lying down
            return self._set_lying_pose(data)
```

**Training Curriculum:**
```bash
# Stage 0: Normal training (already done)
python experiment.py --config-name=conf_wildrobot_amp_stage0

# Stage 1: Start from crouch (easier)
python experiment.py --config-name=conf_wildrobot_amp_stage1 \
  load_checkpoint=outputs/stage0/PPOJax_saved.pkl

# Stage 2: Start from sitting
python experiment.py --config-name=conf_wildrobot_amp_stage2 \
  load_checkpoint=outputs/stage1/PPOJax_saved.pkl

# Stage 3: Start from lying (hardest)
python experiment.py --config-name=conf_wildrobot_amp_stage3 \
  load_checkpoint=outputs/stage2/PPOJax_saved.pkl
```

**Approach 2: Add Getup Mocap (Alternative)**

Use mocap data of human getting up:
```python
default_dataset_conf=DefaultDatasetConf([
    "walk",
    "run",
    "getup",      # Add getup motion from mocap
    "standup",
])
```

**Expected Results:**
- Robot can stand up from any pose
- Recovers from falls
- Returns to commanded behavior ✓

### Phase 4: Sim-to-Real Transfer (1-2 weeks)

**Goal:** Deploy to physical WildRobot

**4.1: Domain Randomization**

Add randomness during training to match real-world variation:

```python
from loco_mujoco.core import DomainRandomizer

class WildRobotDR(DomainRandomizer):
    """Domain randomization for sim-to-real."""

    def randomize(self, env, model, data, backend):
        # Randomize mass (±20%)
        for i in range(model.nbody):
            base_mass = model.body_mass[i]
            model.body_mass[i] = backend.uniform(0.8, 1.2) * base_mass

        # Randomize joint friction (±50%)
        for i in range(model.njnt):
            base_friction = model.jnt_frictionloss[i]
            model.jnt_frictionloss[i] = backend.uniform(0.5, 1.5) * base_friction

        # Randomize actuator gains (±30%)
        for i in range(model.nu):
            base_kp = model.actuator_gainprm[i, 0]
            model.actuator_gainprm[i, 0] = backend.uniform(0.7, 1.3) * base_kp

        # Randomize ground friction
        model.geom_friction[0, 0] = backend.uniform(0.5, 1.2)

        return model, data

WildRobotDR.register()
```

**Training with DR:**
```bash
python experiment.py \
  --config-name=conf_wildrobot_amp_dr \
  domain_randomizer="WildRobotDR" \
  num_updates=5000
```

**4.2: System Identification**

Measure real robot parameters:
```python
# Collect data from real robot
real_robot_mass = measure_mass()
real_robot_friction = measure_friction()
real_robot_actuator_gains = measure_gains()

# Update simulation to match
update_mujoco_model(
    mass=real_robot_mass,
    friction=real_robot_friction,
    gains=real_robot_actuator_gains,
)
```

**4.3: Deploy to Real Robot**

```python
# On real WildRobot
from wildrobot_deployment import RealWildRobot
import jax

# Load trained policy
agent_state = load_agent("outputs/final/PPOJax_saved.pkl")

# Create real robot interface
robot = RealWildRobot()

# Control loop
obs = robot.get_observation()  # Read from real IMU sensors
while True:
    # Get action from policy
    action = agent.get_action(obs, deterministic=True)

    # Send to robot
    robot.set_action(action)

    # Read next observation
    obs = robot.get_observation()

    # Check for command change
    if new_command:
        # Commands already learned in training!
        pass  # Policy handles it automatically
```

**Expected Results:**
- Policy transfers to real robot ✓
- Human-like motion on physical WildRobot ✓
- Responds to commands ✓
- Recovers from falls ✓

## Summary: Complete Training Pipeline

```
Phase 1: AMP Basic Training (2-3 days)
├─ Learn human-like walking/running/turning from mocap
├─ Use noisy IMU sensors (sim-to-real prep)
└─ Output: Natural motion policy

Phase 2: Command Conditioning (1-2 days)
├─ Add discrete command interface
├─ Map commands to target velocities
└─ Output: Command-following policy

Phase 3: Fall Recovery (2-3 days)
├─ Curriculum learning: stand, crouch, sit, lie
├─ Or add getup mocap data
└─ Output: Robust policy with recovery

Phase 4: Sim-to-Real (1-2 weeks)
├─ Domain randomization
├─ System identification
├─ Deploy to real robot
└─ Output: Working robot! 🤖
```

## Why Not GAIL or DeepMimic?

### GAIL (Generative Adversarial Imitation Learning)
- ❌ Less stable than AMP
- ❌ Harder to combine task + style objectives
- ❌ No motion prior learning (just trajectory following)
- Use case: When you have perfect expert demonstrations, not mocap

### DeepMimic
- ❌ Too rigid - follows exact trajectories
- ❌ Hard to add high-level commands
- ❌ Less flexible for fall recovery
- ❌ Doesn't learn motion "style", just specific motions
- Use case: When you need exact motion reproduction (animation, not robotics)

### AMP ✅
- ✅ Learns motion style (not just specific trajectories)
- ✅ Easily combines task objectives (follow command) + style (look human)
- ✅ More stable than GAIL
- ✅ More flexible than DeepMimic
- ✅ State-of-the-art for character control
- ✅ Proven sim-to-real success

## Recommended File Structure

```
wildrobot_project/
├── wildrobot_extensions/
│   ├── goals.py              # GoalCommandVelocity
│   ├── domain_randomizers.py # WildRobotDR
│   └── initial_states.py     # FallRecoveryInitialState
├── configs/
│   ├── conf_wildrobot_amp_phase1.yaml
│   ├── conf_wildrobot_amp_phase2.yaml
│   ├── conf_wildrobot_amp_phase3.yaml
│   └── conf_wildrobot_amp_dr.yaml
├── deployment/
│   ├── real_wildrobot.py     # Real robot interface
│   └── deploy.py             # Deployment script
└── README.md
```

## Timeline

| Phase | Duration | Compute | Output |
|-------|----------|---------|--------|
| Phase 1: Basic AMP | 2-3 days | GPU (RTX 3080+) | Natural walking |
| Phase 2: Commands | 1-2 days | GPU | Command following |
| Phase 3: Fall Recovery | 2-3 days | GPU | Robust policy |
| Phase 4: Sim-to-Real | 1-2 weeks | Real robot | Deployed system |
| **Total** | **~3 weeks** | - | **Working robot!** |

## Next Steps

1. **Create AMP config** (I'll help you)
2. **Download mocap datasets** (`loco-mujoco-download`)
3. **Start Phase 1 training**
4. **Monitor progress** (WandB)
5. **Iterate and improve**

Ready to start? I can help you create the config files and training scripts!
