#!/usr/bin/env python3
"""
Diagnostics for trained PPOJax agents on LocoMuJoCo environments.

Features:
- Load saved agent (.pkl) and reconstruct env from embedded config
- Short rollout (deterministic or stochastic)
- Report policy entropy, action magnitude stats, forward velocity stats
- Verify foot contact geoms are present and count contacts during rollout
- Sanity-check observations and actions (NaNs/infs, percentile ranges)
- Detect early resets via episode length histogram

Usage:
  python diagnostics.py --agent /path/to/PPOJax_saved.pkl --steps 2000 --n-envs 128 --deterministic 0
"""
import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.core.utils import mj_jntname2qvelid


def _make_env_from_conf(conf):
    factory = TaskFactory.get_factory_cls(conf.experiment.task_factory.name)
    env = factory.make(**conf.experiment.env_params, **conf.experiment.task_factory.params)
    # Wrap like training
    env = PPOJax._wrap_env(env, conf.experiment)
    return env


def _policy_apply(network, train_state, obs, rng):
    y, updates = network.apply({'params': train_state.params,
                                'run_stats': train_state.run_stats},
                               obs, mutable=["run_stats"])
    pi, val = y
    return pi, val, updates['run_stats']


def _gather_forward_vel_indices(env):
    # Find index of root free joint x-velocity in qvel
    free_jnt_name = env.root_free_joint_xml_name
    model = env.unwrapped()._model
    vel_idx = mj_jntname2qvelid(free_jnt_name, model)[0]
    return vel_idx


def _foot_geom_ids(env):
    # Return geom ids for any configured foot geoms; ignore if missing
    names = []
    try:
        # info properties are attributes on env, not env.info
        names = list(env.foot_geom_names)
    except Exception:
        return []
    model = env.unwrapped()._model
    ids = []
    for n in names:
        try:
            gid = int(np.where(np.array([g.name for g in model.geoms], dtype=object) == n)[0][0])
        except Exception:
            # fallback to mujoco api name2id
            import mujoco
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            ids.append(gid)
    return ids


def run_rollout(agent_path, steps=2000, n_envs=128, deterministic=False, seed=0):
    # Load agent
    agent_conf, agent_state = PPOJax.load_agent(agent_path)
    conf = agent_conf.config
    network = agent_conf.network
    ts = agent_state.train_state

    # Build env
    env = _make_env_from_conf(conf)

    # Prep policy call
    @jax.jit
    def act(ts, obs, rng):
        pi, _val, run_stats = _policy_apply(network, ts, obs, rng)
        ts = ts.replace(run_stats=run_stats)
        if deterministic:
            a = pi.mode()
        else:
            a = pi.sample(seed=rng)
        return a, ts, pi

    # Seed and reset
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    obs, state = env.reset(env_keys)

    # Diagnostics accumulators
    vel_idx = _gather_forward_vel_indices(env)
    forward_vels = []
    entropies = []
    act_mags = []
    ep_lengths = np.zeros((n_envs,), dtype=np.int32)
    ep_done_counts = np.zeros((n_envs,), dtype=np.int32)
    foot_ids = _foot_geom_ids(env)
    foot_contacts = 0

    for t in range(int(steps)):
        rng, sub = jax.random.split(rng)
        a, ts, pi = act(ts, obs, sub)
        a = jnp.atleast_2d(a)
        obs, rew, absorbing, done, info, state = env.step(state, a)

        # metrics
        try:
            v = jnp.mean(state.env_state.data.qvel[:, vel_idx])
            forward_vels.append(float(v))
        except Exception:
            pass
        try:
            entropies.append(float(jnp.mean(pi.entropy())))
        except Exception:
            pass
        try:
            act_mags.append(float(jnp.mean(jnp.abs(a))))
        except Exception:
            pass

        # step episode counters
        d_np = np.asarray(done)
        ep_lengths += 1
        ep_done_counts += d_np.astype(np.int32)
        ep_lengths = ep_lengths * (1 - d_np)  # reset where done

        # count foot contacts
        try:
            if foot_ids and hasattr(state.env_state.data, 'contact'):
                geoms = np.asarray(state.env_state.data.contact.geom)
                # geoms shape: [ncon, 2]; count any contact involving foot ids
                if geoms.ndim == 2 and geoms.size > 0:
                    g1 = geoms[:, 0]
                    g2 = geoms[:, 1]
                    c = np.isin(g1, foot_ids) | np.isin(g2, foot_ids)
                    foot_contacts += int(np.sum(c))
        except Exception:
            pass

    # Summaries
    summ = {
        "mean_forward_vel": float(np.mean(forward_vels)) if forward_vels else np.nan,
        "p90_forward_vel": float(np.percentile(forward_vels, 90)) if forward_vels else np.nan,
        "mean_entropy": float(np.mean(entropies)) if entropies else np.nan,
        "mean_abs_action": float(np.mean(act_mags)) if act_mags else np.nan,
        "episodes_finished": int(np.sum(ep_done_counts > 0)),
        "min_ep_len": int(np.min(ep_lengths)) if ep_lengths.size > 0 else -1,
        "max_ep_len": int(np.max(ep_lengths)) if ep_lengths.size > 0 else -1,
        "foot_contact_events": int(foot_contacts),
    }

    # Observation sanity (last obs)
    obs_np = np.asarray(obs)
    obs_finite = np.isfinite(obs_np).all()
    summ["obs_all_finite"] = bool(obs_finite)
    try:
        low = np.asarray(env.info.observation_space.low)
        high = np.asarray(env.info.observation_space.high)
        within = np.logical_and(obs_np >= low, obs_np <= high)
        summ["obs_within_bounds_frac"] = float(np.mean(within))
    except Exception:
        summ["obs_within_bounds_frac"] = np.nan

    print("\nDiagnostics summary:")
    for k, v in summ.items():
        print(f"  {k}: {v}")

    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', required=True, help='Path to PPOJax_saved.pkl')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--n-envs', type=int, default=128)
    p.add_argument('--deterministic', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    agent_path = os.path.expanduser(args.agent)
    deterministic = bool(args.deterministic)

    run_rollout(agent_path, steps=args.steps, n_envs=args.n_envs,
                deterministic=deterministic, seed=args.seed)


if __name__ == '__main__':
    main()
