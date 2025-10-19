"""
Clean retargeted AMASS cache for WildRobot/MjxWildRobot.

By default, removes only the shape file (shape_optimized.pkl) so the next run refits
with the new SMPL-H model. Use --all to also delete all retargeted .npz clips.

Usage examples (with uv):
    uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache               # shape only, both envs
    uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache --all         # shape + all .npz, both envs
    uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache --env MjxWildRobot
    uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache --dry-run
    uv run python -m loco_mujoco.models.wildrobot.clean_amass_cache --cache /custom/AMASS
"""
from __future__ import annotations

import argparse
import os
import glob
from typing import Iterable, List

import yaml

import loco_mujoco


DEFAULT_ENVS = ["WildRobot", "MjxWildRobot"]
SHAPE_FILE = "shape_optimized.pkl"


def _resolve_cache_path(override: str | None) -> str:
    if override:
        return override
    # Read from LOCOMUJOCO_VARIABLES.yaml
    with open(loco_mujoco.PATH_TO_VARIABLES, "r") as f:
        data = yaml.safe_load(f) or {}
    cache = data.get("LOCOMUJOCO_CONVERTED_AMASS_PATH")
    if not cache:
        raise SystemExit(
            "LOCOMUJOCO_CONVERTED_AMASS_PATH not set. Use `loco-mujoco-set-conv-amass-path --path <dir>` first."
        )
    return cache


def _clean_env(cache_root: str, env: str, delete_all: bool, dry_run: bool) -> List[str]:
    """Return list of removed paths (or would-be removed in dry-run)."""
    removed: List[str] = []
    env_dir = os.path.join(cache_root, env)
    if not os.path.isdir(env_dir):
        return removed

    # Always remove the shape file
    shape_path = os.path.join(env_dir, SHAPE_FILE)
    if os.path.exists(shape_path):
        removed.append(shape_path)
        if not dry_run:
            try:
                os.remove(shape_path)
            except OSError:
                pass

    # Optionally remove all retargeted trajectory npz files
    if delete_all:
        for p in glob.glob(os.path.join(env_dir, "*.npz")):
            removed.append(p)
            if not dry_run:
                try:
                    os.remove(p)
                except OSError:
                    pass

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean AMASS cache for WildRobot/MjxWildRobot")
    parser.add_argument(
        "--env",
        action="append",
        dest="envs",
        help="Environment(s) to clean (repeatable). Defaults to both WildRobot and MjxWildRobot.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also delete all retargeted .npz files (not just the shape file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing files.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Override LOCOMUJOCO_CONVERTED_AMASS_PATH (path to converted AMASS cache)",
    )
    args = parser.parse_args()

    cache_root = _resolve_cache_path(args.cache)
    envs: Iterable[str] = args.envs if args.envs else DEFAULT_ENVS

    print(f"Cache root: {cache_root}")
    print(f"Envs: {', '.join(envs)}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'DELETE'} | {'ALL (*.npz + shape)' if args.all else 'SHAPE ONLY'}")

    total: List[str] = []
    for env in envs:
        removed = _clean_env(cache_root, env, delete_all=args.all, dry_run=args.dry_run)
        if removed:
            print(f"[{env}] Removing {len(removed)} file(s):")
            for p in removed:
                print(f"  - {p}")
            total.extend(removed)
        else:
            print(f"[{env}] Nothing to remove.")

    if args.dry_run:
        print(f"DRY-RUN complete. {len(total)} file(s) would be removed.")
    else:
        print(f"Done. Removed {len(total)} file(s).")


if __name__ == "__main__":
    main()
