#!/usr/bin/env python3
"""Inspect site frames and saved SMPL->robot transforms.

Usage: run from repository root.
This script prints site_xmat for the envs UnitreeH1v2 and WildRobot, and
attempts to load the cached shape file (shape_optimized.pkl) for WildRobot
to show the saved pelvis transform.

It tries to be defensive if optional deps (SMPL, jax) or the AMASS cache are
not available.
"""
import os
import sys
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import joblib
except Exception:
    joblib = None

try:
    import mujoco
except Exception:
    mujoco = None

try:
    import jax
except Exception:
    jax = None

try:
    from loco_mujoco.environments import LocoEnv
    from loco_mujoco import PATH_TO_VARIABLES
    from loco_mujoco.smpl.retargeting import OPTIMIZED_SHAPE_FILE_NAME
except Exception as e:
    print('Failed to import loco_mujoco internals:', e)
    sys.exit(1)


def load_vars():
    try:
        data = yaml.safe_load(open(PATH_TO_VARIABLES)) or {}
        return data
    except Exception as e:
        print('Could not read LOCOMUJOCO_VARIABLES.yaml:', e)
        return {}


def make_env(name):
    env_cls = LocoEnv.registered_envs.get(name)
    if env_cls is None:
        raise KeyError(f'Env {name} not registered')
    # prefer the th_params style used by other scripts; fall back to no args
    for ctor_args in (dict(th_params=dict(random_start=False, fixed_start_conf=(0, 0))), {}, None):
        try:
            if ctor_args is None:
                env = env_cls()
            else:
                env = env_cls(**ctor_args)
            # attempt reset to populate site matrices
            try:
                if jax is not None:
                    key = jax.random.key(0)
                    env.reset(key)
                else:
                    # some envs allow None/0
                    env.reset(0)
            except Exception:
                # ignore reset errors; site_xmat may still be available
                pass
            return env
        except Exception:
            continue
    raise RuntimeError(f'Could not construct env {name}')


def print_site_axes(env_name, env):
    print(f'=== site axes for env: {env_name} ===')
    sites = env.sites_for_mimic
    # resolve site ids defensively
    for s in sites:
        try:
            sid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s)
        except Exception:
            try:
                sid = env._model.site_name2id(s)
            except Exception:
                sid = None
        if sid is None or sid < 0:
            print(f'  site: {s} -> NOT FOUND')
            continue
        pos = env._data.site_xpos[sid]
        mat = env._data.site_xmat[sid].reshape(3, 3)
        try:
            e = R.from_matrix(mat).as_euler('xyz', degrees=True)
        except Exception:
            e = ('NA',)
        print(f'Env: {env_name}')
        print(f'  site: {s}')
        print('    pos:', np.array2string(pos, precision=6))
        print('    mat:\n' + np.array2string(mat, precision=9))
        print('    euler (deg, xyz):', e)


def _fmt_axis(v):
    return np.array2string(v, precision=6)


def print_world_axes_for_sites(env_name, env):
    """Print explicit world-frame axes (X, Y, Z) for each mimic site and its parent body.

    This prints the three basis vectors in world coordinates so it's easy to see
    which way +X, +Y and +Z of the site/body point in the viewer.
    """
    print(f'=== world-frame axes for env: {env_name} ===')
    sites = env.sites_for_mimic
    for s in sites:
        # resolve site id
        try:
            sid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s)
        except Exception:
            try:
                sid = env._model.site_name2id(s)
            except Exception:
                sid = None
        if sid is None or sid < 0:
            print(f'  site: {s} -> NOT FOUND')
            continue

        spos = env._data.site_xpos[sid]
        smat = env._data.site_xmat[sid].reshape(3, 3)
        # interpret columns as basis vectors (local X/Y/Z expressed in world coords)
        sx = smat[:, 0]
        sy = smat[:, 1]
        sz = smat[:, 2]
        print(f'Env: {env_name} | site: {s}')
        print('  site pos:', np.array2string(spos, precision=6))
        print('  site +X (world):', _fmt_axis(sx))
        print('  site +Y (world):', _fmt_axis(sy))
        print('  site +Z (world):', _fmt_axis(sz))

        # try to print the parent body axes as well (helpful to understand identity-site vs composed)
        try:
            bodyid = env._model.site[sid].bodyid
        except Exception:
            bodyid = None
        if bodyid is not None:
            try:
                bname = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_BODY, int(bodyid))
            except Exception:
                bname = f'id:{bodyid}'
            try:
                bpos = env._data.xpos[int(bodyid)]
                bmat = env._data.xmat[int(bodyid)].reshape(3, 3)
                bx = bmat[:, 0]
                by = bmat[:, 1]
                bz = bmat[:, 2]
                print(f'  parent body: {bname}')
                print('    body pos:', np.array2string(bpos, precision=6))
                print('    body +X (world):', _fmt_axis(bx))
                print('    body +Y (world):', _fmt_axis(by))
                print('    body +Z (world):', _fmt_axis(bz))
            except Exception as e:
                print('    could not read parent body axes:', e)
        print('')


def inspect_saved_shape(cache_root, env_name):
    if joblib is None:
        print('joblib not available; cannot inspect cached shape file')
        return
    shape_path = os.path.join(cache_root, env_name, OPTIMIZED_SHAPE_FILE_NAME)
    print('\n== inspect', shape_path, '==')
    if not os.path.exists(shape_path):
        print('shape file not found:', shape_path)
        return
    try:
        data = joblib.load(shape_path)
    except Exception as e:
        print('Failed to load shape file:', e)
        return
    # expected layout: (shape_new, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale)
    if len(data) < 4:
        print('Unexpected shape file layout')
        return
    smpl2robot_rot_mat = data[3]
    # create an env to query sites_for_mimic order
    env = make_env(env_name)
    sites = env.sites_for_mimic
    if 'pelvis_mimic' in sites:
        idx = sites.index('pelvis_mimic')
        mat = smpl2robot_rot_mat[idx]
        print('pelvis rot mat:\n', np.array2string(mat, precision=9))
        try:
            print('euler xyz (deg):', np.degrees(R.from_matrix(mat).as_euler('xyz', degrees=False)))
        except Exception as e:
            print('Could not convert to euler:', e)
    else:
        print('pelvis_mimic not in sites_for_mimic')


def main():
    vars = load_vars()
    cache_root = vars.get('LOCOMUJOCO_CONVERTED_AMASS_PATH')
    env_names = ['UnitreeH1v2', 'WildRobot']
    for name in env_names:
        try:
            env = make_env(name)
        except Exception as e:
            print(f'Could not create env {name}:', e)
            continue
        try:
            print_site_axes(name, env)
        except Exception as e:
            print('Error printing site axes for', name, e)
        try:
            print_world_axes_for_sites(name, env)
        except Exception as e:
            print('Error printing world axes for', name, e)
    if cache_root:
        inspect_saved_shape(cache_root, 'WildRobot')
    else:
        print('LOCOMUJOCO_CONVERTED_AMASS_PATH not set; cannot inspect cached shape file')


if __name__ == '__main__':
    main()
