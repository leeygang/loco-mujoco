# Adding Realistic Sensor Noise to WildRobot IMUs

## Why Add Sensor Noise?

Currently, WildRobot's IMU sensors read **perfect simulation data** with zero noise. This makes them equivalent to built-in observations like `BodyVel` or `ProjectedGravityVector`.

**Adding realistic sensor noise provides:**
- ✅ Better sim-to-real transfer
- ✅ More robust policies
- ✅ Matches actual hardware behavior
- ✅ Agent learns to handle measurement uncertainty

## Sensor Specifications

### BNO085 (Chest IMU)
**Datasheet specs:**
- Gyroscope noise: ~0.014 deg/s/√Hz → ~0.0002 rad/s
- Accelerometer noise: ~150 μg/√Hz → ~0.0015 m/s²
- Magnetometer noise: ~0.3 μT
- Bandwidth: 100 Hz

### ICM45686 (Knee IMUs)
**Datasheet specs:**
- Gyroscope noise: ~0.003 deg/s/√Hz → ~0.00005 rad/s
- Accelerometer noise: ~80 μg/√Hz → ~0.0008 m/s²
- Bandwidth: 200 Hz

## Updated XML Sensors

Replace the `<sensor>` section in `wildrobot.xml`:

```xml
<sensor>
  <!-- Chest IMU (BNO085) - 9-DOF sensor with realistic noise -->
  <gyro site="chest_imu" name="chest_imu_gyro"
        noise="0.0002" cutoff="100"/>
  <accelerometer site="chest_imu" name="chest_imu_accel"
        noise="0.0015" cutoff="100"/>
  <magnetometer site="chest_imu" name="chest_imu_mag"
        noise="0.3"/>
  <framequat objtype="site" objname="chest_imu" name="chest_imu_quat"/>

  <!-- Left Knee IMU (ICM45686) - 6-DOF sensor with realistic noise -->
  <gyro site="left_knee_imu" name="left_knee_imu_gyro"
        noise="0.00005" cutoff="200"/>
  <accelerometer site="left_knee_imu" name="left_knee_imu_accel"
        noise="0.0008" cutoff="200"/>

  <!-- Right Knee IMU (ICM45686) - 6-DOF sensor with realistic noise -->
  <gyro site="right_knee_imu" name="right_knee_imu_gyro"
        noise="0.00005" cutoff="200"/>
  <accelerometer site="right_knee_imu" name="right_knee_imu_accel"
        noise="0.0008" cutoff="200"/>

  <!-- Pelvis sensors (keep for comparison) -->
  <gyro site="pelvis_mimic" name="pelvis_gyro"/>
  <accelerometer site="pelvis_mimic" name="pelvis_accel"/>
  <velocimeter site="pelvis_mimic" name="pelvis_local_linvel"/>
  <framequat objtype="site" objname="pelvis_mimic" name="pelvis_quat"/>
  <framezaxis objtype="site" objname="pelvis_mimic" name="pelvis_upvector"/>
  <framexaxis objtype="site" objname="pelvis_mimic" name="pelvis_forwardvector"/>
  <framelinvel objtype="site" objname="pelvis_mimic" name="pelvis_global_linvel"/>
  <frameangvel objtype="site" objname="pelvis_mimic" name="pelvis_global_angvel"/>

  <!-- Other mimic site sensors (perfect data for privileged information) -->
  <gyro site="right_hip_mimic" name="right_hip_gyro"/>
  <accelerometer site="right_hip_mimic" name="right_hip_accel"/>
  <framequat objtype="site" objname="right_hip_mimic" name="right_hip_quat"/>
  <frameangvel objtype="site" objname="right_hip_mimic" name="right_hip_angvel"/>

  <!-- ... rest of sensors ... -->
</sensor>
```

## Key Parameters

### `noise` attribute
- Gaussian noise standard deviation added to sensor readings
- Units match sensor output (rad/s for gyro, m/s² for accel)
- Higher values = noisier measurements

### `cutoff` attribute
- Low-pass filter cutoff frequency in Hz
- Simulates sensor bandwidth limitations
- Lower values = more filtering = smoother but delayed readings

## Training Strategy: Privileged Learning

Use this pattern to leverage both noisy and perfect observations:

```python
observation_spec = [
    # Student observations (noisy, matches real robot)
    ObservationType.IMUSensor("chest_imu",
        ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"],
        group="student"),  # Will be used on real robot

    ObservationType.IMUSensor("left_knee_imu",
        ["left_knee_imu_gyro", "left_knee_imu_accel"],
        group="student"),

    # Teacher observations (perfect, simulation only)
    ObservationType.BodyVel("pelvis_vel", "waist",
        group="teacher"),  # Perfect velocity
    ObservationType.ProjectedGravityVector("gravity", "waist_freejoint",
        group="teacher"),  # Perfect orientation

    # Common observations
    ObservationType.JointPosArray("joint_pos", [...]),
    ObservationType.LastAction("last_action"),
]
```

### Teacher-Student Training
1. **Training:** Use both student + teacher observations (noisy + perfect)
2. **Deployment:** Use only student observations (matches real robot)

This gives you:
- Faster learning (teacher provides perfect info)
- Robust deployment (student handles noisy sensors)

## Testing Noise Impact

Compare training with/without noise:

```python
# Test 1: Perfect sensors (current)
python train_wildrobot_with_imu.py --noise=0

# Test 2: Realistic noise (recommended)
python train_wildrobot_with_imu.py --noise=realistic

# Test 3: High noise (stress test)
python train_wildrobot_with_imu.py --noise=high
```

Expected results:
- **No noise:** Fast convergence, brittle policies
- **Realistic noise:** Slower convergence, robust policies ✓
- **High noise:** Much slower, very robust

## Validation on Real Robot

After training with noise:

1. **Collect real IMU data** from WildRobot during walking
2. **Compare statistics:**
   ```python
   # Simulation IMU statistics
   sim_gyro_std = np.std(sim_chest_imu[:3])  # Should match BNO085 spec

   # Real robot IMU statistics
   real_gyro_std = np.std(real_chest_imu[:3])

   # If close → good sim-to-real match!
   print(f"Sim: {sim_gyro_std:.6f}, Real: {real_gyro_std:.6f}")
   ```

3. **Tune noise levels** to match reality

## Advanced: Domain Randomization

Randomize sensor noise during training:

```python
class IMUNoiseRandomizer(DomainRandomizer):
    def randomize(self, env, model, data, backend):
        # Randomize gyro noise (±50% variation)
        for i in range(model.nsensor):
            if model.sensor_type[i] == 3:  # GYRO
                base_noise = 0.0002
                model.sensor_noise[i] = backend.uniform(
                    base_noise * 0.5,
                    base_noise * 1.5
                )
        return model, data

IMUNoiseRandomizer.register()
```

## Comparison: Noisy IMU vs Perfect State

| Observation Type | Noise | Real Robot | Training Speed | Deployment Robustness |
|-----------------|-------|------------|----------------|---------------------|
| IMUSensor (with noise) | ✓ | ✓ | Slower | High ✓ |
| IMUSensor (no noise) | ✗ | ✓ | Fast | Low ✗ |
| BodyVel/ProjectedGravity | ✗ | ✗ | Fast | Low ✗ |

**Recommendation:** Use noisy IMU sensors for production sim-to-real transfer.

## Summary

### Without Noise (Current)
```
IMUSensor ≈ BodyVel/ProjectedGravity
Both read perfect simulation state → No advantage
```

### With Noise (Recommended)
```
IMUSensor = Realistic, noisy measurements
BodyVel = Perfect state (privileged info)

→ Use both in training (teacher-student)
→ Deploy only IMUSensor to real robot
→ Better sim-to-real transfer ✓
```

**Next steps:**
1. Add `noise` and `cutoff` attributes to sensors in XML
2. Train with noisy observations
3. Compare performance with/without noise
4. Deploy to real WildRobot and validate
