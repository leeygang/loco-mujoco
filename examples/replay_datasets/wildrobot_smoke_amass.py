import argparse
import jax

from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf


def main(headless: bool = False, steps: int = 500,
         amass_rel_path: str = "KIT/3/walking_medium07_poses"):
    env = ImitationFactory.make(
        "MjxWildRobot",
        amass_dataset_conf=AMASSDatasetConf(rel_dataset_path=amass_rel_path),
        headless=headless,
        goal_type="GoalTrajMimic",
        reward_type="MimicReward",
    )

    # MJX backend requires trajectory in JAX format
    if env.th is not None and env.th.is_numpy:
        env.th.to_jax()

    key = jax.random.key(0)
    state = env.mjx_reset(key)

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
    parser.add_argument("--amass_rel_path", type=str, default="KIT/3/walking_medium07_poses")
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps, amass_rel_path=args.amass_rel_path)
