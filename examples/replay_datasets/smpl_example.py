import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment(seed=0):

    np.random.seed(seed)

    # # example --> you can add as many datasets as you want in the lists!
    env = ImitationFactory.make("UnitreeH1",
                                # if SMPL and AMASS are installed, you can use the following:
                                amass_dataset_conf=AMASSDatasetConf(["KIT/3/walking_slow08_poses"]),
                                n_substeps=20)

    env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


if __name__ == '__main__':
    experiment()
