"""
WildRobot Extensions for loco-mujoco

This package contains custom observation types, rewards, and other extensions
specific to WildRobot that are kept separate from the loco-mujoco library.

Usage:
    # Import and register custom observations
    from wildrobot_extensions import IMUSensor, AllIMUSensors

    # Register with loco-mujoco
    IMUSensor.register()
    AllIMUSensors.register()

    # Now you can use them in your observation specs
    from loco_mujoco import RLFactory
    from loco_mujoco.core import ObservationType

    observation_spec = [
        ObservationType.IMUSensor("chest_imu",
            ["chest_imu_gyro", "chest_imu_accel", "chest_imu_mag"])
    ]
"""

from .observations import IMUSensor, AllIMUSensors

__all__ = ['IMUSensor', 'AllIMUSensors']
