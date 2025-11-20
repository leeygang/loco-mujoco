# Files Moved to training_amp ✅

## New Directory Structure

```
examples/training_examples/training_amp/
├── .gitignore                              # Ignore outputs, cache files
├── README.md                               # Project overview and quick start
├── test_setup.py                           # Verify setup is working
├── experiment.py                           # Main training script
├── eval.py                                 # Evaluation script
├── conf_wildrobot_amp_phase1.yaml         # Phase 1 training config
├── QUICKSTART_AMP_TRAINING.md             # Quick start guide
├── WILDROBOT_TRAINING_STRATEGY.md         # Complete 4-phase strategy
└── wildrobot_extensions/                  # Custom extensions
    ├── __init__.py
    ├── observations.py                    # IMUSensor, AllIMUSensors
    └── README.md                          # Extension documentation
```

## What Was Moved

### From `jax_rl/` to `training_amp/`:

✅ **Core Files:**
- `experiment.py` - Training script
- `eval.py` - Evaluation script

✅ **Configuration:**
- `conf_wildrobot_amp_phase1.yaml` - Phase 1 config

✅ **Documentation:**
- `QUICKSTART_AMP_TRAINING.md` - Quick start
- `WILDROBOT_TRAINING_STRATEGY.md` - Full strategy

✅ **Custom Extensions:**
- `wildrobot_extensions/` - Complete directory
  - `__init__.py`
  - `observations.py`
  - `README.md`

✅ **New Files:**
- `README.md` - Project-specific README
- `test_setup.py` - Setup verification
- `.gitignore` - Ignore patterns

## What Stayed in `jax_rl/`

These files remain in `jax_rl/` for reference and other experiments:

- `conf_wildrobot_noisy_imu.yaml` - PPO with noisy IMU
- `conf_wildrobot_imu.yaml` - PPO with IMU
- `conf_quickcheck.yaml` - Quick test config
- `demo_imu_observations.py` - IMU demos
- `compare_imu_vs_builtin.py` - Sensor comparison
- `verify_sensor_noise.py` - Noise verification
- `test_wildrobot_sensors.py` - Sensor testing
- `train_wildrobot_with_imu.py` - IMU training template
- Other documentation and helper scripts

## Benefits of New Structure

### ✅ Clean Separation
- AMP training work is self-contained
- Easy to find all AMP-related files
- No confusion with other experiments

### ✅ Self-Contained
- All dependencies in one place
- Can copy `training_amp/` to another machine
- Clear entry point (`README.md`)

### ✅ Version Control
- `.gitignore` prevents committing outputs
- Easy to track changes to AMP work
- Can branch/tag AMP progress separately

### ✅ Collaboration
- Share `training_amp/` folder independently
- Clear documentation for collaborators
- No extra files to confuse people

## How to Use

### Navigate to training_amp:
```bash
cd /Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp
```

### Test Setup:
```bash
python test_setup.py
```

### Start Training:
```bash
# Quick test
python experiment.py --config-name=conf_wildrobot_amp_phase1 num_updates=100

# Full training
python experiment.py --config-name=conf_wildrobot_amp_phase1
```

### Evaluate:
```bash
python eval.py --path outputs/[timestamp]/AMPJax_saved.pkl
```

## File Relationships

```
training_amp/
│
├── experiment.py                 # Uses →
│   ├── conf_wildrobot_amp_phase1.yaml
│   └── wildrobot_extensions/
│
├── eval.py                      # Loads →
│   └── outputs/*/AMPJax_saved.pkl
│
├── test_setup.py                # Tests →
│   └── wildrobot_extensions/
│
└── Documentation                # Guides →
    ├── README.md                    (Overview)
    ├── QUICKSTART_AMP_TRAINING.md   (Quick start)
    └── WILDROBOT_TRAINING_STRATEGY.md (Full strategy)
```

## Next Steps

1. ✅ Files organized
2. ⏳ Test setup: `python test_setup.py`
3. ⏳ Download datasets: `loco-mujoco-download`
4. ⏳ Start training: `python experiment.py --config-name=conf_wildrobot_amp_phase1`

## Original Files

The original files in `jax_rl/` remain untouched for:
- Reference
- Other experiments (PPO, GAIL, etc.)
- Sensor testing and validation

---

**Status:** Ready to start AMP training! 🚀

**Location:** `/Users/ygli/projects/loco-mujoco/examples/training_examples/training_amp`
