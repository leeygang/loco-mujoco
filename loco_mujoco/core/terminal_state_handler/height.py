from typing import Dict, Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model


from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class HeightBasedTerminalStateHandler(TerminalStateHandler):
    """
    Check if the current state is terminal based on the height of the root.
    """
    def __init__(self, env: Any, **handler_config: Dict[str, Any]):
        """
        Initialize the TerminalStateHandler.

        Args:
            env (Any): The environment instance.
            **handler_config (Any): Configuration dictionary.
        """
        super().__init__(env, **handler_config)

        # Default range comes from environment info; allow overrides via handler_config
        default_min, default_max = self._info_props["root_height_healthy_range"]
        min_h = handler_config.get("min_height", default_min)
        max_h = handler_config.get("max_height", default_max)
        self.root_height_range = (min_h, max_h)

        self.root_free_joint_xml_ind = np.array(
            mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], env._model)
        )

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terminal state handler.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        return data, carry

    def is_absorbing(self,
                     env: Any,
                     obs: np.ndarray,
                     info: Dict[str, Any],
                     data: MjData,
                     carry: Any) -> Union[bool, Any]:
        """
        Check if the current state is terminal. Function for CPU Mujoco.

        Args:
            env (Any): The environment instance.
            obs (np.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (MjData): The Mujoco data structure.
            carry (Any): Additional carry information.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=np)

    def mjx_is_absorbing(self,
                         env: Any,
                         obs: jnp.ndarray,
                         info: Dict[str, Any],
                         data: Data,
                         carry: Any) -> Union[bool, Any]:
        """
        Check if the current state is terminal. Function for Mjx.

        Args:
            env (Any): The environment instance.
            obs (jnp.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (Data): The Mujoco data structure for Mjx.
            carry (Any): Additional carry information.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp)

    def _is_absorbing_compat(self,
                             env: Any,
                             obs: Union[np.ndarray, jnp.ndarray],
                             info: Dict[str, Any],
                             data: Union[MjData, Data],
                             carry: Any,
                             backend: ModuleType) -> Union[bool, Any]:
        """
        Check if the current state is terminal. Compatible with both CPU Mujoco and Mjx.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): Additional carry information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        # MJX Data has shape (n_envs, nq); CPU Data has shape (nq,)
        # For both CPU Mujoco and MJX, within a single env step data.qpos is 1D (nq,).
        # Vectorization over environments is handled by vmap outside this function, so we return a scalar.
        root_pose = data.qpos[self.root_free_joint_xml_ind]
        height = root_pose[2]
        if backend == jnp:
            height_cond = jnp.logical_or(height < self.root_height_range[0],
                                         height > self.root_height_range[1])
        else:
            height_cond = np.logical_or(height < self.root_height_range[0],
                                        height > self.root_height_range[1])
        return height_cond, carry
