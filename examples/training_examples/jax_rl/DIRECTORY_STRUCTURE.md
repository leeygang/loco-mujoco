# JAX RL Training Directory - Clean Structure

## 🎯 Main Training Configs (Two-Step Approach)

### Step 1: Stable Forward Walking
- **`conf_step1_stable_walking.yaml`** - Train for reliable forward locomotion (NO RSI)
  - Pure RL approach
  - Default initialization (robot starts upright, facing forward)
  - Balanced rewards (velocity + stability + smoothness)
  - Expected: Mean Forward Vel +0.3 to +0.6 m/s

### Step 2: Human-Like Motion (After Step 1 Works)
- **`conf_step2_humanlike_amp.yaml`** - Add natural motion with AMP
  - Uses WildRobot expert data (generated from Step 1 policy)
  - Handles size differences via style-based discriminator
  - Maintains forward walking while refining motion quality

### Documentation
- **`TWO_STEP_TRAINING_GUIDE.md`** - Complete guide explaining the approach
- **`README.md`** - Original jax_rl documentation

---

## 🛠️ Training Scripts

- **`experiment.py`** - Main training script
  ```bash
  python experiment.py --config-name conf_step1_stable_walking
  ```

- **`eval.py`** - Evaluate trained policies
  ```bash
  python eval.py --path outputs/.../PPOJax_saved.pkl
  ```

- **`generate_wildrobot_dataset.py`** - Generate expert trajectories from trained policy
  ```bash
  python generate_wildrobot_dataset.py \
    --policy_path outputs/.../PPOJax_saved.pkl \
    --num_episodes 200 \
    --output_dir wildrobot_expert_motions/
  ```

---

## 📡 IMU/Sensor Features (Separate Track)

These are for training with IMU sensor observations (separate from main locomotion training):

### Configs
- `conf_wildrobot_imu.yaml` - Train with IMU observations
- `conf_wildrobot_noisy_imu.yaml` - Train with realistic sensor noise

### Scripts
- `compare_imu_vs_builtin.py` - Compare IMU vs built-in observations
- `demo_imu_observations.py` - Demonstrate IMU sensor readings
- `test_wildrobot_sensors.py` - Test sensor implementations
- `train_wildrobot_with_imu.py` - Train with IMU-specific setup
- `verify_sensor_noise.py` - Verify sensor noise levels

### Documentation
- `add_sensor_noise_to_wildrobot.md` - How sensor noise was added
- `IMU_SENSORS_README.md` - IMU sensor architecture
- `SENSOR_NOISE_ADDED.md` - Sensor noise specifications
- `WILDROBOT_IMU_ARCHITECTURE.md` - Detailed IMU implementation

---

## 🗑️ Removed Files (Failed Approaches)

The following configs were removed because they used RSI with human data, which caused backward walking due to orientation mismatches:

- ❌ `conf_rsi_humanlike_combined.yaml` - RSI + reward tuning (walked backward)
- ❌ `conf_rsi_quickcheck.yaml` - Quick RSI test (walked backward)
- ❌ `conf_rsi_quickcheck_fixed.yaml` - Attempted fix (still walked backward)
- ❌ `conf_wildrobot_amp_custom.yaml` - Old AMP config
- ❌ `conf_wildrobot_amp_phase1.yaml` - Old AMP config
- ❌ `conf_quickcheck.yaml` - Old quick test
- ❌ `conf_wildrobot.yaml` - Old base config
- ❌ `conf.yaml` - Old default config
- ❌ `check_rsi_orientation.py` - Diagnostic script (didn't run)
- ❌ `motor_control_example.xml` - Example file
- ❌ `QUICKSTART_AMP_TRAINING.md` - Old guide (superseded)
- ❌ `WILDROBOT_TRAINING_STRATEGY.md` - Old strategy (superseded)

---

## 🚀 Quick Start

**For stable forward walking:**
```bash
python experiment.py --config-name conf_step1_stable_walking
```

**After Step 1 succeeds, generate expert data:**
```bash
python generate_wildrobot_dataset.py \
  --policy_path outputs/YOUR_RUN/PPOJax_saved.pkl \
  --num_episodes 200
```

**Then train for human-like motion:**
```bash
# First update conf_step2_humanlike_amp.yaml to point to your expert data
python experiment.py --config-name conf_step2_humanlike_amp
```

See `TWO_STEP_TRAINING_GUIDE.md` for complete details.
