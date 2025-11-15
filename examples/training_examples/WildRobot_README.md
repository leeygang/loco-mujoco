# Training WildRobot with JAX RL (PPO)

This guide explains how to train a **WildRobot** locomotion policy using the
JAX PPO pipeline in `examples/training_examples`.

It assumes:
- You have installed LocoMuJoCo in editable mode.
- You are using **uv** for dependency management (pip alternatives are also shown).

---

## 1. Environment and dependencies

### 1.1. Base installation (uv)

From the repository root:

```bash
uv sync
```

This uses `pyproject.toml` to create/sync the base environment with:
- MuJoCo / MJX
- JAX, Flax, Orbax
- PPOJax and utilities

If you prefer `pip` instead of `uv`, you can do:

```bash
pip install -e .
```

### 1.2. Optional NVIDIA CUDA support

If you want to train WildRobot on GPU with CUDA JAX, and you have configured
`gpu` as a dependency group / extra in `pyproject.toml`, run:

```bash
uv sync --group gpu
```

or, with pip extras (if configured):

```bash
pip install -e ".[gpu]"
```

Then verify that JAX sees your GPU:

```bash
python -c "import jax; print(jax.devices())"
```

You should see at least one `GpuDevice` in the printed list.

> Note: Make sure your CUDA / driver stack matches the JAX CUDA wheels. See the
> JAX installation docs if you encounter version or compatibility errors.

SMPL/AMASS are **not required** for pure RL training; WildRobot is used here as a
standard RL environment.

---

## 2. WildRobot RL configuration

The WildRobot-specific PPO configuration is in:

- `examples/training_examples/jax_rl/conf_wildrobot.yaml`

Key settings:

- **Factory**: `RLFactory` (via `TaskFactory`)
- **Environment**: `MjxWildRobot` (MJX, GPU-capable variant of WildRobot)
- **Reward**: `LocomotionReward`
- **Goal**: `GoalRandomRootVelocity` (encourages forward motion)
- **PPO hyperparameters** (defaults you can tune):
  - `num_envs: 1024`
  - `num_steps: 50`
  - `total_timesteps: 5e7`
  - `hidden_layers: [512, 256]`
  - `lr: 1e-4`

You can edit this YAML if you want to change reward weights, horizon, number of
parallel environments, or PPO hyperparameters.

---

## 3. Training entrypoint

WildRobot RL training uses the **generic JAX RL experiment script** with a
WildRobot-specific config:

- Script: `examples/training_examples/jax_rl/experiment.py`
- Config: `examples/training_examples/jax_rl/conf_wildrobot.yaml`

You do **not** need a separate Python wrapper; Hydra will load the
WildRobot config directly via the command line.

---

## 4. How to launch training

From the repository root, run the experiment script with the
WildRobot config name:

```bash
cd examples/training_examples/jax_rl
uv run experiment.py --config-name conf_wildrobot
```

If you are using `pip` instead of `uv`:

```bash
cd examples/training_examples/jax_rl
python experiment.py --config-name conf_wildrobot
```

What happens during training:

- A `MjxWildRobot` environment is created via `RLFactory`.
- PPOJax builds and JIT-compiles the training function.
- Training runs for `total_timesteps` as defined in `conf_wildrobot.yaml`.
- Metrics are logged to Weights & Biases (WandB) under the project name
  specified in the config (default: `wildrobot-rl`).
- At the end of training, the agent state is saved and a rollout is recorded
  as a video using `PPOJax.play_policy`.

Make sure you are logged into WandB before starting training:

```bash
wandb login
```

If you want to temporarily disable WandB, you can either:
- Set `experiment.debug: true` in `conf_wildrobot.yaml`, or
- Edit the base `jax_rl/experiment.py` to skip WandB init/logging.

---

## 5. Common tweaks

You can customize WildRobot RL behavior by editing
`examples/training_examples/jax_rl/conf_wildrobot.yaml`.

### 5.1. Training scale and speed

- `experiment.num_envs`: number of parallel MJX environments
  - Example: reduce to `512` or `256` if 1024 is too heavy.
- `experiment.num_steps`: rollout length per PPO update.
- `experiment.total_timesteps`: total number of environment interaction steps.

### 5.2. Reward shaping

Under `experiment.env_params.reward_params` you can adjust:

- `tracking_w_exp_xy`, `tracking_w_exp_yaw`: how strongly to track target
  velocities.
- `air_time_coeff`, `air_time_max`: encourage or penalize foot air-time.
- `joint_acc_coeff`, `joint_torque_coeff`, `energy_coeff`: regularization
  terms for smoother, lower-energy motion.
- `joint_position_limit_coeff`: penalty for hitting joint limits.
- `action_rate_coeff`, `symmetry_air_coeff`: additional smoothness/symmetry
  regularizers.

These weights will influence gait stability, energy usage, and style.

### 5.3. PPO hyperparameters

You can also tune:

- `hidden_layers`: network size (e.g., `[256, 256]` for a smaller model).
- `lr`: learning rate (e.g., `3e-4` for faster learning but potentially less
  stable).
- `clip_eps`, `num_minibatches`, `update_epochs`: PPO-specific stability knobs.

---

## 6. Sanity checks

Before launching long runs, it is useful to check:

1. **Device setup**

   ```bash
   python -c "import jax; print(jax.devices())"
   ```

   Confirm that the expected CPU/GPU devices are listed.

2. **Short debug run**

   In `conf_wildrobot.yaml`, you can temporarily reduce:

   - `total_timesteps` (e.g., `1e6`)
   - `num_envs` (e.g., `128`)

   and/or set `experiment.debug: true` to verify everything runs end-to-end
   before committing to a long training run.

3. **WandB logging**

   Check that metrics (e.g., mean episode return, episode length) appear in
   your WandB project and that the logged video at the end plays back a
   reasonable WildRobot motion.

With this setup, you have a clean starting point for training WildRobot with
pure RL. Once this is working, you can extend to imitation-based methods
(`jax_rl_mimic`, `jax_amp`, `jax_gail`) using WildRobot and your AMASS
pipelines.
