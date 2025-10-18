"""
AMASS smoke playback for MjxWildRobot.

Runs a short rollout using a selected AMASS clip. If needed, this will trigger
SMPL shape fitting and motion retargeting, then run a few simulation steps.

Usage:
    python -m loco_mujoco.models.wildrobot.smoke_amass \
        --clip KIT/3/walking_slow08_poses \
        --steps 200
"""
import argparse

from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import AMASSDatasetConf


def main():
    parser = argparse.ArgumentParser(description="AMASS smoke playback for MjxWildRobot")
    parser.add_argument("--clip", type=str, default="KIT/3/walking_slow08_poses",
                        help="Relative AMASS dataset path (e.g., KIT/3/walking_slow08_poses)")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to simulate")
    args = parser.parse_args()

    env = ImitationFactory.make(
        "MjxWildRobot",
        amass_dataset_conf=AMASSDatasetConf(rel_dataset_path=[args.clip]),
    )

    obs, carry = env.mjx_reset(carry=None)
    total_r = 0.0
    for _ in range(args.steps):
        action = env.sample_random_action(carry.key)
        obs, reward, terminated, truncated, info, carry = env.mjx_step(action, carry)
        total_r += float(reward)

    print(f"Ran {args.steps} steps. Total reward: {total_r:.3f}")


if __name__ == "__main__":
    main()
