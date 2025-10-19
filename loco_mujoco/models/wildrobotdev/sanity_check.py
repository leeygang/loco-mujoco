"""
Sanity-check WildRobot mimic sites and SMPL mapping.

Checks:
1) env.sites_for_mimic sites exist in the XML model.
2) YAML mapping in smpl/robot_confs/WildRobot.yaml covers exactly those sites.

Usage:
    python -m loco_mujoco.models.wildrobot.sanity_check [--env WildRobot]
"""
import argparse
from typing import List

import mujoco

from loco_mujoco.environments.humanoids import WildRobot
from loco_mujoco.smpl.retargeting import load_robot_conf_file


def _check_sites_exist(model: mujoco.MjModel, site_names: List[str]) -> None:
    for s in site_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
        if sid == -1:
            raise SystemExit(f"[SanityCheck] Missing site in XML: {s}")


def main():
    parser = argparse.ArgumentParser(description="Sanity-check WildRobot sites and SMPL mapping")
    parser.add_argument("--env", type=str, default="WildRobot", choices=["WildRobot", "MjxWildRobot"],
                        help="Environment name to use for mapping lookup (Mjx prefix is ignored for YAML)")
    args = parser.parse_args()

    # Build a CPU env to access the model/sites (cheaper for a quick check)
    env = WildRobot()
    model = env._model
    sites = env.sites_for_mimic
    print("sites_for_mimic:", sites)

    # 1) Ensure sites exist in the XML
    _check_sites_exist(model, sites)

    # 2) Ensure YAML mapping covers at least the required set
    conf = load_robot_conf_file(args.env)
    mapped_sites = list(conf.site_joint_matches.keys())
    print("YAML site mappings:", mapped_sites)

    missing = set(sites) - set(mapped_sites)
    extra = set(mapped_sites) - set(sites)
    print("Missing in YAML:", sorted(missing))
    if extra:
        print("Note: YAML contains additional default mappings (ignored):", sorted(extra))

    if missing:
        raise SystemExit(f"[SanityCheck] Sites in env.sites_for_mimic not mapped in YAML: {missing}")

    print("[SanityCheck] PASS: Required sites exist and are mapped in YAML.")


if __name__ == "__main__":
    main()
