"""
WildRobot Custom Observation Types

This module contains custom observation types specific to WildRobot that extend
the loco-mujoco library without modifying it.

These can be used by importing and registering them in your training scripts.
"""

from __future__ import annotations
from typing import List
import numpy as np
import mujoco

from loco_mujoco.core.observations.base import Observation


class IMUSensor(Observation):
    """
    Observation type for reading from physical IMU sensors.

    This class reads gyroscope, accelerometer, and optionally magnetometer data
    from IMU sensors defined in the MuJoCo XML model.

    Args:
        obs_name: Name of this observation (e.g., "chest_imu", "left_knee_imu").
        sensor_names: List of sensor names to read from. The sensors are read in order
            and concatenated. For example, ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]
            will create a 9-dimensional observation (3 + 3 + 3).
        group: Optional group name for this observation.
        allow_randomization: Whether to allow randomization of this observation.

    Example:
        # Register the observation type
        from wildrobot_extensions.observations import IMUSensor
        IMUSensor.register()

        # Use in observation spec
        observation_spec = [
            # 9-DOF IMU (gyro + accel + mag)
            ObservationType.IMUSensor("chest_imu",
                ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"]),

            # 6-DOF IMU (gyro + accel only)
            ObservationType.IMUSensor("left_knee_imu",
                ["left_knee_imu_gyro", "left_knee_imu_accel"]),
        ]
    """

    def __init__(self, obs_name: str, sensor_names: List[str], **kwargs):
        self.sensor_names = sensor_names
        self.sensor_ids = None
        self.sensor_addrs = None
        self.sensor_dims = None
        self.dim = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize IMU observation from MuJoCo model.

        This method looks up the sensor IDs, addresses, and dimensions from the
        MuJoCo model and sets up the observation indices.
        """
        # Look up sensor information
        self.sensor_ids = []
        self.sensor_addrs = []
        self.sensor_dims = []

        for sensor_name in self.sensor_names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id < 0:
                raise ValueError(f"Sensor '{sensor_name}' not found in model")

            self.sensor_ids.append(sensor_id)
            self.sensor_addrs.append(model.sensor_adr[sensor_id])
            self.sensor_dims.append(model.sensor_dim[sensor_id])

        # Convert to numpy arrays
        self.sensor_ids = np.array(self.sensor_ids)
        self.sensor_addrs = np.array(self.sensor_addrs)
        self.sensor_dims = np.array(self.sensor_dims)

        # Calculate total dimension
        self.dim = int(np.sum(self.sensor_dims))

        # Set observation properties
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim

        # For sensor data, data_type_ind stores the sensor addresses
        self.data_type_ind = self.sensor_addrs

        # Observation indices in the flattened observation vector
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])

        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Returns the data type name in MuJoCo data structure.
        For IMU sensors, we read from sensordata.
        """
        return "sensordata"

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Get all IMU sensor observations from the MuJoCo data structure.

        Args:
            env: The environment.
            model: The MuJoCo model.
            data: The MuJoCo data structure.
            data_ind_cont: The observation index container.
            backend: The backend to use (np or jnp).

        Returns:
            Flattened array of all IMU sensor readings.
        """
        # Get all IMU observations registered in this environment
        imu_obs_list = []
        if hasattr(env, 'observation_specification'):
            for obs in env.observation_specification.values():
                if isinstance(obs, cls):
                    # Read sensor data for this IMU
                    sensor_values = []
                    for addr, dim in zip(obs.sensor_addrs, obs.sensor_dims):
                        sensor_values.append(data.sensordata[addr:addr+dim])
                    imu_obs_list.append(backend.concatenate(sensor_values))

        if len(imu_obs_list) == 0:
            return backend.empty(shape=(0,))

        return backend.concatenate(imu_obs_list)


class AllIMUSensors(Observation):
    """
    Convenience observation type that reads from ALL IMU sensors in the model.

    This automatically discovers all gyro and accelerometer sensors and includes
    them in the observation. Useful for quick experimentation.

    Args:
        obs_name: Name of this observation.
        include_magnetometer: Whether to include magnetometer sensors.
        group: Optional group name for this observation.
        allow_randomization: Whether to allow randomization of this observation.

    Example:
        # Register the observation type
        from wildrobot_extensions.observations import AllIMUSensors
        AllIMUSensors.register()

        # Use in observation spec
        observation_spec = [
            ObservationType.AllIMUSensors("all_imus", include_magnetometer=True)
        ]
    """

    def __init__(self, obs_name: str, include_magnetometer: bool = False, **kwargs):
        self.include_magnetometer = include_magnetometer
        self.sensor_ids = None
        self.sensor_addrs = None
        self.sensor_dims = None
        self.dim = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize by discovering all IMU sensors in the model.
        """
        # Sensor types we want to include
        # MuJoCo sensor types: GYRO=3, ACCELEROMETER=1, MAGNETOMETER=6
        desired_types = [1, 3]  # accel, gyro
        if self.include_magnetometer:
            desired_types.append(6)  # magnetometer

        self.sensor_ids = []
        self.sensor_addrs = []
        self.sensor_dims = []

        # Scan all sensors and find gyro/accel/mag sensors
        for i in range(model.nsensor):
            sensor_type = model.sensor_type[i]
            if sensor_type in desired_types:
                self.sensor_ids.append(i)
                self.sensor_addrs.append(model.sensor_adr[i])
                self.sensor_dims.append(model.sensor_dim[i])

        # Convert to numpy arrays
        self.sensor_ids = np.array(self.sensor_ids)
        self.sensor_addrs = np.array(self.sensor_addrs)
        self.sensor_dims = np.array(self.sensor_dims)

        # Calculate total dimension
        self.dim = int(np.sum(self.sensor_dims))

        # Set observation properties
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim

        # For sensor data, data_type_ind stores the sensor addresses
        self.data_type_ind = self.sensor_addrs

        # Observation indices in the flattened observation vector
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])

        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """Returns 'sensordata' as the data type."""
        return "sensordata"

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Get all IMU sensor observations.

        Args:
            env: The environment.
            model: The MuJoCo model.
            data: The MuJoCo data structure.
            data_ind_cont: The observation index container.
            backend: The backend to use (np or jnp).

        Returns:
            Flattened array of all IMU sensor readings.
        """
        # Read sensor data
        sensor_values = []
        for obs in env.observation_specification.values():
            if isinstance(obs, cls):
                for addr, dim in zip(obs.sensor_addrs, obs.sensor_dims):
                    sensor_values.append(data.sensordata[addr:addr+dim])

        if len(sensor_values) == 0:
            return backend.empty(shape=(0,))

        return backend.concatenate(sensor_values)
