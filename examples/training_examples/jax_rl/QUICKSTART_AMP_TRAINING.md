# Quick Start: Training Command-Following WildRobot

## TL;DR - Your Answer

**Use AMP (Adversarial Motion Priors)** ✅

Why:
- ✅ Learns **human-like motion** from mocap (exactly what you need!)
- ✅ Works with **high-level commands** (stop, walk, turn)
- ✅ Supports **fall recovery**
- ✅ Best **sim-to-real** transfer
- ✅ Already available in loco-mujoco (`AMPJax`)

## Complete Training Pipeline

```
┌─────────────────────────────────────────────────────┐
│ Phase 1: Learn Human-like Walking (2-3 days)       │
│ Algorithm: AMP                                      │
│ Output: Natural walking/running gaits               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Phase 2: Add Command Interface (1-2 days)          │
│ Mod: Custom goal type for commands                 │
│ Output: Responds to stop/walk/turn commands        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Phase 3: Fall Recovery (2-3 days)                  │
│ Method: Curriculum learning                        │
│ Output: Stands up after falling                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Phase 4: Sim-to-Real Transfer (1-2 weeks)          │
│ Techniques: Domain randomization, system ID        │
│ Output: Working on real WildRobot! 🤖              │
└─────────────────────────────────────────────────────┘
```

## Step-by-Step: Phase 1 (Start Here!)

### 1. Download Mocap Datasets

```bash
cd /Users/ygli/projects/loco-mujoco

# Download all datasets (includes walk, run motions)
loco-mujoco-download

# This will take a few minutes
# Downloads from HuggingFace: robfiras/loco-mujoco-datasets
```

### 2. Register Custom IMU Observations

Already done! Just make sure to import in your training script:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from wildrobot_extensions import IMUSensor
IMUSensor.register()
```

### 3. Start Training Phase 1

```bash
cd examples/training_examples/jax_rl

# Quick test (100 updates, ~10 minutes)
python experiment.py \
  --config-name=conf_wildrobot_amp_phase1 \
  num_updates=100

# Full training (5000 updates, ~2-3 days on RTX 3080)
python experiment.py \
  --config-name=conf_wildrobot_amp_phase1
```

### 4. Monitor Training

Training will log to WandB:
```
Logging to: https://wandb.ai/your-username/wildrobot-amp-phase1
```

Watch for:
- **Mean Episode Return**: Should increase (target: >500)
- **Mean Episode Length**: Should increase (target: ~600)
- **Style Reward**: Discriminator score (target: >0.8)
- **Task Reward**: Goal following (target: >0.7)

### 5. Evaluate Results

```bash
# After training completes
python eval.py --path outputs/[timestamp]/AMPJax_saved.pkl

# Watch the robot walk!
# It should look human-like ✓
```

## Algorithm Comparison

| Algorithm | Human-like Motion | Commands | Fall Recovery | Sim-to-Real | Recommended |
|-----------|-------------------|----------|---------------|-------------|-------------|
| **AMP** | ✅✅ Excellent | ✅ Easy | ✅ Easy | ✅✅ Excellent | **YES** ✅ |
| GAIL | ✅ Good | ⚠️ Medium | ⚠️ Medium | ✅ Good | No |
| DeepMimic | ✅ Good | ❌ Hard | ❌ Hard | ✅ Good | No |
| Pure PPO | ❌ Robotic | ✅ Easy | ✅ Easy | ⚠️ Medium | No |

**Clear winner: AMP** 🏆

## Why AMP Beats the Others

### vs GAIL
```
GAIL:
  - Learns to imitate expert demonstrations
  - Less stable training
  - Hard to add task objectives
  - ❌ Not ideal for your use case

AMP:
  - Learns motion STYLE (not just trajectories)
  - More stable
  - Easy to combine task + style
  - ✅ Perfect for your use case
```

### vs DeepMimic
```
DeepMimic:
  - Follows exact trajectories
  - Too rigid for commands
  - Hard to add fall recovery
  - ❌ Not flexible enough

AMP:
  - Learns flexible motion style
  - Easy to add commands
  - Easy to add fall recovery
  - ✅ Much more flexible
```

### vs Pure PPO
```
Pure PPO:
  - Fast to train
  - Motion looks robotic
  - ❌ Doesn't meet "human-like" requirement

AMP:
  - Slightly slower to train
  - Motion looks natural
  - ✅ Meets all requirements
```

## Files Ready for You

✅ **Training config**: `conf_wildrobot_amp_phase1.yaml`
✅ **Strategy guide**: `WILDROBOT_TRAINING_STRATEGY.md`
✅ **Custom observations**: `wildrobot_extensions/observations.py`
✅ **Sensor noise**: Already added to `wildrobot.xml`

## Next Steps After Phase 1

### Phase 2: Add Commands (After Phase 1 completes)

Create custom goal type:

```python
# In wildrobot_extensions/goals.py
class GoalCommandVelocity(Goal):
    COMMANDS = {
        "stop": [0.0, 0.0, 0.0],
        "walk_slow": [0.5, 0.0, 0.0],
        "walk": [1.0, 0.0, 0.0],
        "walk_fast": [2.0, 0.0, 0.0],
        "turn_left": [0.5, 0.0, +0.5],
        "turn_right": [0.5, 0.0, -0.5],
    }
    # Implementation...
```

Register and use:
```bash
python experiment.py \
  --config-name=conf_wildrobot_amp_phase2 \
  load_checkpoint=outputs/phase1/AMPJax_saved.pkl
```

### Phase 3: Fall Recovery

Use curriculum learning or add getup mocap:
```bash
python experiment.py \
  --config-name=conf_wildrobot_amp_phase3 \
  load_checkpoint=outputs/phase2/AMPJax_saved.pkl
```

### Phase 4: Deploy to Real Robot

1. Domain randomization
2. System identification
3. Deploy!

## Expected Timeline

| Phase | Duration | Hardware | Status |
|-------|----------|----------|--------|
| **Phase 1** | **2-3 days** | **GPU** | **START HERE** ⬅️ |
| Phase 2 | 1-2 days | GPU | After Phase 1 |
| Phase 3 | 2-3 days | GPU | After Phase 2 |
| Phase 4 | 1-2 weeks | Real robot | Final deployment |
| **TOTAL** | **~3 weeks** | - | **Working robot!** 🎉 |

## Troubleshooting

### "ModuleNotFoundError: No module named 'wildrobot_extensions'"

```bash
# Make sure you're in the right directory
cd examples/training_examples/jax_rl

# And importing correctly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### "Dataset not found"

```bash
# Download datasets first
loco-mujoco-download

# Or use environment variable
export LOCOMUJOCO_CONVERTED_DEFAULT_PATH="$HOME/.loco-mujoco-caches"
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

### "Out of memory"

```bash
# Reduce num_envs
python experiment.py \
  --config-name=conf_wildrobot_amp_phase1 \
  num_envs=1024  # Instead of 2048
```

## Summary

**Your Question:**
> Which training to use: jax_amp, gail, or rl_mimic?

**Answer:**
**Use AMP (jax_amp)** ✅

**Why:**
1. ✅ Human-like motion (discriminator learns from mocap)
2. ✅ High-level commands (easy to add goal conditioning)
3. ✅ Fall recovery (curriculum learning)
4. ✅ Best sim-to-real transfer (with noisy IMU sensors)
5. ✅ Proven for character control and robotics

**Start Command:**
```bash
cd examples/training_examples/jax_rl
loco-mujoco-download  # First time only
python experiment.py --config-name=conf_wildrobot_amp_phase1
```

**Estimated Time to Working Robot:** ~3 weeks

Good luck! 🚀 Let me know when you start training and I can help debug!
