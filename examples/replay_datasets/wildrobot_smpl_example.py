import numpy as np
from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf


def experiment(seed: int = 0):
    """
    Replay an AMASS SMPL-H trajectory on the WildRobot environment.

    Prerequisites:
    - Install CPU PyTorch first, then `pip install -e '.[smpl]'`
    - Set paths:
      loco-mujoco-set-amass-path --path /path/to/amass
      loco-mujoco-set-smpl-model-path --path /path/to/smpl
      # optional cache for converted AMASS
      # loco-mujoco-set-conv-amass-path --path /path/to/amass_conv
    - Ensure SMPLH_NEUTRAL.PKL is generated (see loco_mujoco/smpl/README.MD)
    """

    np.random.seed(seed)

    # Choose a sample AMASS sequence to replay
    # You can add as many sequences as you want to the list
    amass_sequences = [
        "KIT/3/walking_slow08_poses",
    ]

    # Use the CPU (MuJoCo) environment for interactive rendering
    env = ImitationFactory.make(
        "WildRobot",
        amass_dataset_conf=AMASSDatasetConf(amass_sequences),
        n_substeps=20,
    )

    # Play a few episodes with rendering
    env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


if __name__ == "__main__":
    experiment()
