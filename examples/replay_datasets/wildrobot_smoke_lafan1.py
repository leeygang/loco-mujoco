import argparse
import jax
import jax.numpy as jnp

from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf


def main(headless: bool = False, steps: int = 500):
    env = ImitationFactory.make(
        "MjxWildRobot",
        lafan1_dataset_conf=LAFAN1DatasetConf(dataset_name="walk1_subject5"),
        headless=headless,
        goal_type="GoalTrajMimic",
        reward_type="MimicReward",
    )

    # MJX backend requires trajectory in JAX format
    if env.th is not None and env.th.is_numpy:
        env.th.to_jax()

    key = jax.random.key(0)
    state = env.mjx_reset(key)

    # Roll with random actions
    action_dim = env.info.action_space.shape[0]
    for _ in range(steps):
        key, sub = jax.random.split(key)
        action = jax.random.normal(sub, shape=(action_dim,))
        state = env.mjx_step(state, action)
        env.mjx_render(state)

    env.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without rendering window")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps)
