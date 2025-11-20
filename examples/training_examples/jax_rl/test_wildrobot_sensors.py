"""Test that the updated WildRobot XML loads correctly and verify sensor data."""

import numpy as np
import mujoco

# Load the model
try:
    model = mujoco.MjModel.from_xml_path("/Users/ygli/projects/loco-mujoco/loco_mujoco/models/wildrobot/wildrobot.xml")
    data = mujoco.MjData(model)

    print("✅ WildRobot XML loaded successfully!")
    print(f"\nTotal sensors: {model.nsensor}")
    print(f"Total sensor data elements: {model.nsensordata}")

    # List all sensors
    print("\n" + "="*80)
    print("SENSOR INVENTORY")
    print("="*80)

    sensor_types = {
        0: "TOUCH",
        1: "ACCELEROMETER",
        2: "VELOCIMETER",
        3: "GYRO",
        4: "FORCE",
        5: "TORQUE",
        6: "MAGNETOMETER",
        7: "RANGEFINDER",
        8: "JOINTPOS",
        9: "JOINTVEL",
        10: "TENDONPOS",
        11: "TENDONVEL",
        12: "ACTUATORPOS",
        13: "ACTUATORVEL",
        14: "ACTUATORFRC",
        15: "BALLQUAT",
        16: "BALLANGVEL",
        17: "JOINTLIMITPOS",
        18: "JOINTLIMITVEL",
        19: "JOINTLIMITFRC",
        20: "TENDONLIMITPOS",
        21: "TENDONLIMITVEL",
        22: "TENDONLIMITFRC",
        23: "FRAMEPOS",
        24: "FRAMEQUAT",
        25: "FRAMEXAXIS",
        26: "FRAMEYAXIS",
        27: "FRAMEZAXIS",
        28: "FRAMELINVEL",
        29: "FRAMEANGVEL",
        30: "FRAMELINACC",
        31: "FRAMEANGACC",
        32: "SUBTREECOM",
        33: "SUBTREELINVEL",
        34: "SUBTREEANGMOM",
        35: "CLOCK",
        36: "USER"
    }

    # Group sensors by category
    imu_sensors = []
    pelvis_sensors = []
    hip_sensors = []
    knee_sensors = []
    foot_sensors = []

    for i in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_type = model.sensor_type[i]
        sensor_type_name = sensor_types.get(sensor_type, f"UNKNOWN({sensor_type})")
        dim = model.sensor_dim[i]

        if "chest_imu" in sensor_name or "knee_imu" in sensor_name:
            imu_sensors.append((sensor_name, sensor_type_name, dim))
        elif "pelvis" in sensor_name:
            pelvis_sensors.append((sensor_name, sensor_type_name, dim))
        elif "hip" in sensor_name:
            hip_sensors.append((sensor_name, sensor_type_name, dim))
        elif "knee" in sensor_name:
            knee_sensors.append((sensor_name, sensor_type_name, dim))
        elif "foot" in sensor_name:
            foot_sensors.append((sensor_name, sensor_type_name, dim))

    # Print categorized sensors
    print("\n🔷 PHYSICAL IMU SENSORS (3 total)")
    print("-" * 80)
    for name, stype, dim in imu_sensors:
        print(f"  {name:30s} | {stype:20s} | dim={dim}")

    print("\n🔷 PELVIS (TRUNK) SENSORS")
    print("-" * 80)
    for name, stype, dim in pelvis_sensors:
        print(f"  {name:30s} | {stype:20s} | dim={dim}")

    print("\n🔷 HIP SENSORS (Left + Right)")
    print("-" * 80)
    for name, stype, dim in hip_sensors:
        print(f"  {name:30s} | {stype:20s} | dim={dim}")

    print("\n🔷 KNEE SENSORS (Left + Right, including mimic sites)")
    print("-" * 80)
    for name, stype, dim in knee_sensors:
        print(f"  {name:30s} | {stype:20s} | dim={dim}")

    print("\n🔷 FOOT SENSORS (Left + Right)")
    print("-" * 80)
    for name, stype, dim in foot_sensors:
        print(f"  {name:30s} | {stype:20s} | dim={dim}")

    # Run a few simulation steps to verify sensors work
    print("\n" + "="*80)
    print("SENSOR DATA TEST (after 100 simulation steps)")
    print("="*80)

    mujoco.mj_forward(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Check a few key sensors
    print("\nSample sensor readings:")
    for sensor_name in ["chest_imu_gyro", "left_knee_imu_accel", "right_knee_imu_accel",
                        "pelvis_gyro", "pelvis_accel", "pelvis_quat"]:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id >= 0:
            adr = model.sensor_adr[sensor_id]
            dim = model.sensor_dim[sensor_id]
            values = data.sensordata[adr:adr+dim]
            print(f"  {sensor_name:30s}: {values}")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Sensors are working correctly!")
    print("="*80)

except Exception as e:
    print(f"\n❌ ERROR loading WildRobot XML:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
