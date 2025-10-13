from typing import List, Tuple, Union

import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class WildRobot(BaseRobotHumanoid):
    """
    Minimal environment wrapper for the 11-DOF WildRobot MJCF.

    This environment automatically constructs the observation and action specifications
    from the provided XML:
      - Observations: [FreeJointPosNoXY(root), JointPos for all non-root joints,
        FreeJointVel(root), JointVel for all non-root joints].
      - Actions: one entry per actuator found in the XML (DefaultControl).

    Notes:
      - The robot uses a free joint named 'floating_base' on body 'base' in the provided XML.
      - Sites suitable for mimic/reference tracking default to trunk and feet sites.
    """

    mjx_enabled = False

    def __init__(self, spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed (for MJX variant)
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Build observation spec from XML:
          - Free joint pos/vel (without x,y for position)
          - All joint pos/vel excluding the free root
        """
        # detect root free joint (first free joint in spec)
        free_joint_name = None
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                free_joint_name = j.name
                break
        if free_joint_name is None:
            raise ValueError("WildRobot requires a free joint; none found in spec.")

        # collect all non-root joints
        j_names = [j.name for j in spec.joints if j.name != free_joint_name]

        observation_spec: List[Observation] = []
        observation_spec.append(ObservationType.FreeJointPosNoXY("q_root", xml_name=free_joint_name))
        # individual JointPos/Vel entries (keep names explicit for clarity)
        for name in j_names:
            observation_spec.append(ObservationType.JointPos(f"q_{name}", xml_name=name))
        observation_spec.append(ObservationType.FreeJointVel("dq_root", xml_name=free_joint_name))
        for name in j_names:
            observation_spec.append(ObservationType.JointVel(f"dq_{name}", xml_name=name))

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Action spec: one control entry per actuator, using the actuator names from XML.
        """
        return [a.name for a in spec.actuators]

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return (loco_mujoco.PATH_TO_MODELS / "wildrobot" / "wildrobot.xml").as_posix()

    @info_property
    def root_body_name(self) -> str:
        # root body in the provided XML
        return "base"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        # free joint name in the provided XML
        return "floating_base"

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """
        Minimal set of sites for mimic/reference tracking. Ensure these exist in the XML.
        """
        # trunk IMU and the foot sites exist in wildrobot.xml
        return ["trunk_imu", "left_foot_site", "right_foot_site"]

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        # coarse range; adjust as needed for your robot's size
        return (0.15, 0.7)

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        # Overridden in MjxWildRobot; keep default behavior here
        return spec
