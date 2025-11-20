# WildRobot AMP Training

This directory contains everything needed to train WildRobot with AMP (Adversarial Motion Priors) for human-like motion and command following.

## Directory Structure

```
training_amp/
├── wildrobot_extensions/      # Custom observations, goals, etc.
│   ├── __init__.py
│   ├── observations.py        # IMUSensor, AllIMUSensors
│   └── README.md
├── experiment.py              # Main training script
├── eval.py                    # Evaluation script
├── conf_wildrobot_amp_phase1.yaml  # Phase 1 config
├── QUICKSTART_AMP_TRAINING.md # Quick start guide
├── WILDROBOT_TRAINING_STRATEGY.md  # Complete strategy
└── README.md                  # This file
```

## Quick Start

### 1. Download Mocap Datasets

```bash
cd /Users/ygli/projects/loco-mujoco
loco-mujoco-download
```

### 2. Test Training (Quick)

```bash
cd /Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp

# Quick test (100 updates, ~10 minutes)
python experiment.py \
  --config-name=conf_wildrobot_amp_phase1 \
  num_updates=100
```

### 3. Full Training

```bash
# Full Phase 1 training (5000 updates, 2-3 days on RTX 3080)
python experiment.py --config-name=conf_wildrobot_amp_phase1
```

### 4. Evaluate

```bash
# After training completes
python eval.py --path outputs/[timestamp]/AMPJax_saved.pkl
```

## Training Phases

### Phase 1: Human-like Walking (Current)
- **Goal:** Learn natural walking/running from mocap
- **Duration:** 2-3 days
- **Config:** `conf_wildrobot_amp_phase1.yaml`
- **Output:** Natural motion policy

### Phase 2: Command Following (Next)
- **Goal:** Add discrete commands (stop, walk, turn)
- **Duration:** 1-2 days
- **Config:** TBD (will create after Phase 1)
- **Output:** Command-following policy

### Phase 3: Fall Recovery
- **Goal:** Stand up after falling
- **Duration:** 2-3 days
- **Config:** TBD
- **Output:** Robust recovery policy

### Phase 4: Sim-to-Real
- **Goal:** Deploy to physical WildRobot
- **Duration:** 1-2 weeks
- **Output:** Working robot! 🤖

## Custom Extensions

This project uses external extensions (not in loco-mujoco library):

### Custom Observations
- **IMUSensor** - Read from physical BNO085/ICM45686 sensors
- **AllIMUSensors** - Read all IMU sensors automatically

Located in: `wildrobot_extensions/observations.py`

### Future Extensions
- Custom goals for commands (Phase 2)
- Custom initial states for fall recovery (Phase 3)
- Domain randomizers for sim-to-real (Phase 4)

## Configuration

### Key Parameters in Phase 1 Config

```yaml
# Environment
env_name: "MjxWildRobot"        # GPU-accelerated

# Training
num_envs: 2048                   # Parallel environments
num_updates: 5000                # Total updates (~2-3 days)

# AMP
amp_reward:
  task_reward_weight: 0.5        # Follow goal velocity
  style_reward_weight: 0.5       # Look human-like

# Datasets
default_dataset_conf:
  datasets: ["walk", "run"]      # Mocap data
```

## Sensor Configuration

Physical IMU sensors have **realistic noise** for sim-to-real:

- **chest_imu** (BNO085): noise=0.0002 rad/s
- **knee_imu** (ICM45686): noise=0.00005 rad/s

Mimic site sensors remain **perfect** for discriminator.

## Monitoring Training

Training logs to WandB:
```
Project: wildrobot-amp-phase1
Experiment: wildrobot_amp_humanlike_walking
```

Watch these metrics:
- **Mean Episode Return**: Should increase (target: >500)
- **Mean Episode Length**: Should increase (target: ~600)
- **Style Reward**: Discriminator score (target: >0.8)
- **Task Reward**: Goal following (target: >0.7)

## Outputs

Training saves to:
```
training_amp/outputs/
└── [timestamp]_wildrobot_amp_humanlike_walking/
    ├── AMPJax_saved.pkl          # Trained policy
    ├── config.yaml                # Training config
    ├── metrics.json               # Training metrics
    └── videos/                    # Evaluation videos
```

## Troubleshooting

### Import Error: wildrobot_extensions

The training scripts automatically add the current directory to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Dataset Not Found

```bash
# Download datasets
loco-mujoco-download

# Or set cache path
export LOCOMUJOCO_CONVERTED_DEFAULT_PATH="$HOME/.loco-mujoco-caches"
```

### Out of Memory

```bash
# Reduce parallel environments
python experiment.py \
  --config-name=conf_wildrobot_amp_phase1 \
  num_envs=1024  # Instead of 2048
```

## Documentation

- **QUICKSTART_AMP_TRAINING.md** - Step-by-step quick start
- **WILDROBOT_TRAINING_STRATEGY.md** - Complete 4-phase strategy
- **wildrobot_extensions/README.md** - Custom extensions guide

## Next Steps

1. ✅ Download datasets (`loco-mujoco-download`)
2. ✅ Run quick test (100 updates)
3. ⏳ Start Phase 1 training (5000 updates)
4. ⏳ Evaluate and iterate
5. ⏳ Move to Phase 2 (commands)

## References

- **AMP Paper:** https://arxiv.org/abs/2104.02180
- **loco-mujoco Docs:** https://loco-mujoco.readthedocs.io/
- **MuJoCo MJX:** https://mujoco.readthedocs.io/en/stable/mjx.html

---

**Status:** Phase 1 - Ready to start training! 🚀
