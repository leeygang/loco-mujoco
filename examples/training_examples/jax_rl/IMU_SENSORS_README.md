# WildRobot IMU Sensors - Quick Start Guide

## Overview

Your WildRobot XML now includes **42 sensors** providing comprehensive state information:

### Physical IMU Sensors (3 total)
- **chest_imu** (BNO085): 9-DOF sensor with gyro, accelerometer, magnetometer
- **left_knee_imu** (ICM45686): 6-DOF sensor with gyro, accelerometer
- **right_knee_imu** (ICM45686): 6-DOF sensor with gyro, accelerometer

### Virtual Sensors from Mimic Sites
- **pelvis_mimic**: 8 sensors (gyro, accel, velocities, orientation)
- **hip_mimic** (L/R): 4 sensors each
- **knee_mimic** (L/R): 4 sensors each
- **foot_mimic** (L/R): 5 sensors each

## New Observation Types

I've created two new observation types in `/Users/ygli/projects/loco-mujoco/loco_mujoco/core/observations/imu.py`:

### 1. `ObservationType.IMUSensor`
Read from specific IMU sensors by name.

```python
# 9-DOF IMU (gyro + accel + mag)
ObservationType.IMUSensor("chest_imu",
    ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"])

# 6-DOF IMU (gyro + accel only)
ObservationType.IMUSensor("left_knee_imu",
    ["left_knee_imu_gyro", "left_knee_imu_accel"])
```

### 2. `ObservationType.AllIMUSensors`
Automatically include ALL IMU sensors in one observation.

```python
# Include all gyro + accel + magnetometer sensors
ObservationType.AllIMUSensors("all_imus", include_magnetometer=True)
```

## Usage Examples

### Quick Test
Run the demonstration script:
```bash
cd examples/training_examples/jax_rl
python demo_imu_observations.py
```

This will run 4 examples showing:
1. Individual IMU sensors
2. All IMU sensors at once
3. MJX (GPU) compatibility
4. Physical IMUs vs virtual sensors comparison

### Training with IMU Sensors

**Option 1: Use the example config**
```bash
cd examples/training_examples/jax_rl
python experiment.py --config-name=conf_wildrobot_imu
```

**Option 2: Modify your existing config**

Add IMU observations to your observation specification:

```yaml
observation_specification:
  - type: "JointPosArray"
    obs_name: "joint_pos"
    xml_names: ["right_hip_pitch", "right_hip_roll", ...]

  # Add IMU sensors
  - type: "IMUSensor"
    obs_name: "chest_imu"
    sensor_names: ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]

  - type: "IMUSensor"
    obs_name: "left_knee_imu"
    sensor_names: ["left_knee_imu_gyro", "left_knee_imu_accel"]

  - type: "IMUSensor"
    obs_name: "right_knee_imu"
    sensor_names: ["right_knee_imu_gyro", "right_knee_imu_accel"]

  - type: "ProjectedGravityVector"
    obs_name: "gravity"
    xml_name: "waist_freejoint"

  - type: "LastAction"
    obs_name: "last_action"
```

### Accessing IMU Data During Training

In your environment code:
```python
# Get observation
obs = env.reset()

# Extract specific IMU readings
chest_imu = env._get_from_obs(obs, "chest_imu")  # (9,) array: [gyro(3), accel(3), mag(3)]
left_knee = env._get_from_obs(obs, "left_knee_imu")  # (6,) array: [gyro(3), accel(3)]
```

## Available Sensors in WildRobot XML

### Physical IMU Sensors
```xml
<!-- Chest IMU (BNO085) - 9-DOF -->
<gyro site="chest_imu" name="chest_imu_gyro" />
<accelerometer site="chest_imu" name="chest_imu_accel" />
<magnetometer site="chest_imu" name="chest_imu_mag" />
<framequat objtype="site" objname="chest_imu" name="chest_imu_quat" />

<!-- Left Knee IMU (ICM45686) - 6-DOF -->
<gyro site="left_knee_imu" name="left_knee_imu_gyro" />
<accelerometer site="left_knee_imu" name="left_knee_imu_accel" />

<!-- Right Knee IMU (ICM45686) - 6-DOF -->
<gyro site="right_knee_imu" name="right_knee_imu_gyro" />
<accelerometer site="right_knee_imu" name="right_knee_imu_accel" />
```

### Pelvis Sensors (from pelvis_mimic site)
```xml
<gyro site="pelvis_mimic" name="pelvis_gyro" />
<accelerometer site="pelvis_mimic" name="pelvis_accel" />
<velocimeter site="pelvis_mimic" name="pelvis_local_linvel" />
<framequat objtype="site" objname="pelvis_mimic" name="pelvis_quat" />
<framezaxis objtype="site" objname="pelvis_mimic" name="pelvis_upvector" />
<framexaxis objtype="site" objname="pelvis_mimic" name="pelvis_forwardvector" />
<framelinvel objtype="site" objname="pelvis_mimic" name="pelvis_global_linvel" />
<frameangvel objtype="site" objname="pelvis_mimic" name="pelvis_global_angvel" />
```

## Key Differences

### Physical IMU Sensors vs Virtual Sensors

**Physical IMU Sensors** (`ObservationType.IMUSensor`):
- ✅ Read from actual `<sensor>` tags in XML
- ✅ Simulate real hardware measurements
- ✅ Can include sensor noise (if configured)
- ✅ Better sim-to-real transfer
- ✅ Match what you'd get from physical robot

**Virtual Sensors** (e.g., `ObservationType.BodyVel`):
- ✅ Computed directly from simulation state
- ✅ Mathematically perfect values
- ✅ No sensor noise (unless explicitly added)
- ✅ Easier to train with initially
- ⚠️ May not transfer well to real robot

## Recommendation

**For best sim-to-real transfer**, use the physical IMU observations:
```python
ObservationType.IMUSensor("chest_imu",
    ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"])
ObservationType.IMUSensor("left_knee_imu",
    ["left_knee_imu_gyro", "left_knee_imu_accel"])
ObservationType.IMUSensor("right_knee_imu",
    ["right_knee_imu_gyro", "right_knee_imu_accel"])
```

This matches your physical robot hardware and will make it easier to deploy the trained policy on the real WildRobot.

## Files Created

1. **`/Users/ygli/projects/loco-mujoco/loco_mujoco/core/observations/imu.py`**
   - New observation types: `IMUSensor`, `AllIMUSensors`

2. **`/Users/ygli/projects/loco-mujoco/examples/training_examples/jax_rl/demo_imu_observations.py`**
   - 4 complete examples showing how to use IMU observations

3. **`/Users/ygli/projects/loco-mujoco/examples/training_examples/jax_rl/conf_wildrobot_imu.yaml`**
   - Ready-to-use training config with IMU sensors

4. **`/Users/ygli/projects/loco-mujoco/examples/training_examples/jax_rl/test_wildrobot_sensors.py`**
   - Verification script showing all 42 sensors

5. **`/Users/ygli/projects/loco-mujoco/loco_mujoco/models/wildrobot/wildrobot.xml`**
   - Updated sensor section with all IMU and mimic site sensors

## Next Steps

1. **Test the demo:**
   ```bash
   cd examples/training_examples/jax_rl
   python demo_imu_observations.py
   ```

2. **Train with IMU sensors:**
   ```bash
   python experiment.py --config-name=conf_wildrobot_imu
   ```

3. **Evaluate trained agent:**
   ```bash
   python eval.py --path outputs/.../PPOJax_saved.pkl
   ```

Enjoy training with your new IMU sensors! 🚀
