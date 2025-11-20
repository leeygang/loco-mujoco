# WildRobot AMP Training Guide

Complete guide to train WildRobot with AMP (Adversarial Motion Priors) for human-like locomotion using AMASS motion capture data.

---

## Table of Contents

1. [Overview](#overview)
2. [Setup AMASS (One-Time, 45-60 min)](#setup-amass-one-time-45-60-min)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Troubleshooting](#troubleshooting)

---

## Overview

**Goal:** Train WildRobot to walk/run with human-like motion using AMP algorithm.

**Dataset:** AMASS motion capture data with SMPL-H retargeting to WildRobot.

**Time:**
- Setup: 45-60 minutes (one-time)
- First training: 10-15 minutes (retargeting) + training time
- Subsequent training: Uses cached data, starts immediately

---

## Setup AMASS (One-Time, 45-60 min)

### Step 1: Automated Setup (5 min)

```bash
cd /Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp
./setup_amass.sh
```

This installs:
- PyTorch CPU (avoids JAX conflicts)
- SMPL dependencies (using `uv sync --group smpl`)
- Creates directories: `~/smpl`, `~/amass`, `~/amass_converted`

### Step 2: Download SMPL-H Models (10-15 min)

**Required for motion retargeting.**

1. Visit: https://mano.is.tue.mpg.de/download.php
2. Register (free for academic/research, requires email verification)
3. Download:
   - ✅ **Extended SMPL+H model** (body model)
   - ✅ **Models & Code** (hand models)
4. Extract both to `~/smpl/`
5. Verify:
   ```bash
   ls ~/smpl/SMPLH_MALE.pkl ~/smpl/SMPLH_FEMALE.pkl
   ```

### Step 3: Download AMASS Datasets (15-30 min)

**Human motion capture data.**

1. Visit: https://amass.is.tue.mpg.de/
2. Register (free for academic/research)
3. Download (select **SMPL-H G** version):
   - ✅ **KIT** (~500 MB) - **Required** - Good variety of locomotion
   - ⭐ **CMU** (~2 GB) - Recommended - Large motion variety
   - ⭐ **BMLrub** (~300 MB) - Recommended - Clean locomotion
4. Extract to `~/amass/`
5. Verify:
   ```bash
   ls ~/amass/KIT/3/
   # Should see: walking_slow08_poses.npz, walking_fast01_poses.npz, etc.
   ```

### Step 4: Generate SMPL-H Neutral Model (5 min)

**Combines body + hand models into one file.**

```bash
cd /Users/ygli/projects/loco-mujoco/loco_mujoco/smpl
chmod u+x install_smplh.sh
./install_smplh.sh
```

This script:
- Creates conda environment
- Generates `SMPLH_NEUTRAL.pkl`
- Cleans up

Verify:
```bash
ls ~/smpl/models/SMPLH_NEUTRAL.pkl
```

### Step 5: Verify Setup (Optional, 2 min)

Test SMPL replay with UnitreeH1:

```bash
cd /Users/ygli/projects/loco-mujoco/examples/replay_datasets
uv run python smpl_example.py
```

If this runs without errors, your SMPL setup is correct!

---

## Training

### Quick Test (First Run - Includes Retargeting)

```bash
cd /Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp

uv run python experiment.py \
  --config-name=conf_wildrobot_amp_amass \
  experiment.total_timesteps=200000 \
  wandb.project=wildrobot-test
```

**Note:** Use `uv run python` to ensure dependencies are properly resolved. You can also use `python` directly if you're already in the uv environment.

**What happens on first run:**

1. **Shape fitting** (~2-3 min):
   - Fits SMPL-H body shape to WildRobot proportions
   - Saves: `~/amass_converted/WildRobot/shape_optimized.pkl`

2. **Motion retargeting** (~1-2 min per sequence):
   - Retargets each AMASS motion to WildRobot
   - Saves: `~/amass_converted/WildRobot/KIT_3_walking_slow08_poses.npz`, etc.

3. **Training** (~2 min for 200k timesteps):
   - Learns human-like walking motion

**Subsequent runs:** Uses cached files, starts training immediately!

### Full Training

```bash
uv run python experiment.py --config-name=conf_wildrobot_amp_amass
```

**Training settings** (see `conf_wildrobot_amp_amass.yaml`):
- **Datasets:** KIT walking (slow/medium/fast) + running
- **Total timesteps:** 3 million (~20-30 min on RTX 3080)
- **Environments:** 2048 parallel
- **Discriminator:** Learns to distinguish robot from human motion
- **Reward:** 50% task (velocity tracking) + 50% style (human-like)

**Progress tracking:**
- Watch WandB: Discriminator outputs, episode returns, validation metrics
- Live logging every 10 updates
- Video recorded at end

### Add More Motions

Edit `conf_wildrobot_amp_amass.yaml`:

```yaml
experiment:
  task_factory:
    params:
      amass_dataset_conf:
        rel_dataset_path:
          - "KIT/3/walking_slow08_poses"
          - "KIT/3/walking_fast02_poses"
          - "KIT/10/running_fast01_poses"
          - "CMU/01/01_01_poses"          # Add CMU motions
          - "BMLrub/0007/walking1_poses"  # Add BMLrub
```

Then re-run training - new sequences will be retargeted and cached.

---

## Evaluation

### Replay Trained Policy

```bash
cd /Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp

uv run python eval.py --path outputs/[timestamp]/AMPJax_saved.pkl
```

This will:
- Load trained policy
- Run 200 steps with 20 parallel environments
- Record video
- Show human-like walking motion

### Test on WildRobot Environment

```python
from loco_mujoco.algorithms import AMPJax
from loco_mujoco import RLFactory

# Load trained agent
agent_conf, agent_state = AMPJax.load_agent("path/to/AMPJax_saved.pkl")

# Create WildRobot environment (for visualization/deployment)
env = RLFactory.make(
    "WildRobot",  # CPU version for visualization
    reward_type="LocomotionReward",
    goal_type="GoalRandomRootVelocity"
)

# Play policy
AMPJax.play_policy(env, agent_conf, agent_state,
                   deterministic=True, n_steps=200, record=True)
```

---

## Troubleshooting

### Motion looks wrong (walking backwards, twisted)

**Solution:** Clean retargeting cache and regenerate

```bash
uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache --all
cd examples/training_examples/training_amp
uv run python experiment.py --config-name=conf_wildrobot_amp_amass
```

### "AMASS path not set" error

**Solution:** Reconfigure paths

```bash
loco-mujoco-set-amass-path --path ~/amass
loco-mujoco-set-smpl-model-path --path ~/smpl
loco-mujoco-set-conv-amass-path --path ~/amass_converted
```

### "SMPLH_NEUTRAL.pkl not found"

**Solution:** Regenerate neutral model

```bash
cd loco_mujoco/smpl
./install_smplh.sh
ls ~/smpl/models/SMPLH_NEUTRAL.pkl  # Verify
```

### "Dataset not found: KIT/3/walking_slow08_poses.npz"

**Solution:** Check AMASS download

```bash
ls ~/amass/KIT/3/
# Should show .npz files
```

If missing, re-download KIT dataset from https://amass.is.tue.mpg.de/

### GPU not detected

```bash
uv run python -c "import jax; print(jax.devices())"
# Should show: [cuda(id=0)] or [gpu(id=0)]
```

If CPU only, install GPU JAX:
```bash
uv pip install jax[cuda12]
```

---

## Configuration

**Config file:** `conf_wildrobot_amp_amass.yaml`

This contains all training hyperparameters:
- AMASS dataset paths
- WildRobot environment settings
- AMP discriminator settings
- PPO training parameters

Edit this file to:
- Add more motion sequences
- Adjust hyperparameters
- Change WandB project name

---

## Directory Structure After Setup

```
~/smpl/
├── models/SMPLH_NEUTRAL.pkl       ← Generated neutral model
├── SMPLH_MALE.pkl                 ← Downloaded
└── SMPLH_FEMALE.pkl               ← Downloaded

~/amass/
├── KIT/                           ← Downloaded AMASS data
│   ├── 3/walking_slow08_poses.npz
│   ├── 3/walking_fast01_poses.npz
│   └── 10/running_fast01_poses.npz
├── CMU/...
└── BMLrub/...

~/amass_converted/
└── WildRobot/
    ├── shape_optimized.pkl        ← Fitted WildRobot shape
    ├── KIT_3_walking_slow08_poses.npz  ← Retargeted motion
    ├── KIT_3_walking_fast01_poses.npz
    └── KIT_10_running_fast01_poses.npz

training_amp/
├── conf_wildrobot_amp_amass.yaml  ← Training config
├── experiment.py                  ← Training script
├── eval.py                        ← Evaluation script
├── setup_amass.sh                 ← Setup script
└── outputs/                       ← Training results
    └── [timestamp]/
        ├── AMPJax_saved.pkl       ← Trained policy
        └── config.yaml            ← Training config snapshot
```

---

## Next Steps

After Phase 1 training (human-like walking):

1. **Phase 2:** Add command conditioning (stop, walk slow/fast, turn)
2. **Phase 3:** Add fall recovery
3. **Phase 4:** Sim-to-real transfer to physical WildRobot

See `WILDROBOT_TRAINING_STRATEGY.md` for the complete 4-phase roadmap.

---

## Quick Reference Commands

```bash
# Setup (one-time)
cd training_amp && ./setup_amass.sh

# Download SMPL-H + AMASS manually, then:
cd /Users/ygli/projects/loco-mujoco/loco_mujoco/smpl
./install_smplh.sh

# Quick test
cd /Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp
uv run python experiment.py --config-name=conf_wildrobot_amp_amass \
  experiment.total_timesteps=200000

# Full training
uv run python experiment.py --config-name=conf_wildrobot_amp_amass

# Evaluate
uv run python eval.py --path outputs/[timestamp]/AMPJax_saved.pkl

# Clean cache (if motion looks wrong)
uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache --all
```

---

**Ready to start?** Run `./setup_amass.sh` to begin! 🚀
