import mujoco
from mujoco import MjSpec

from .wildrobot import WildRobot


class MjxWildRobot(WildRobot):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4,
                                     disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        # Keep default contacts; optionally reduce mesh contacts for speed like ToddlerBot does
        # Here we simply disable Euler damping to match options used elsewhere.
        return spec
