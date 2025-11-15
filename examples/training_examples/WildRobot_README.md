# Training WildRobot with JAX RL (PPO) using uv

This guide explains how to train a **WildRobot** locomotion policy with the JAX PPO pipeline in
`examples/training_examples` using **uv** exclusively (fast resolver, reproducible lockfiles).

Assumptions:
* You are in the project root.
* You will use only `uv` commands (no `pip` alternatives).
* GPU training is optional and enabled via the `gpu` dependency group.

---

## 1. Environment and dependencies

### 1.1 Base installation

```bash
uv sync
```

Resolves the base environment (MuJoCo, MJX, JAX, Flax, PPO utilities).

### 1.2 Optional GPU (CUDA JAX)

```bash
uv sync --group gpu
```

Verify devices:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Expect at least one `GpuDevice` if CUDA is active.

### 1.3 Optional dev / SMPL groups (not needed for pure RL)

```bash
uv sync --group dev
uv sync --group smpl
```

SMPL/AMASS assets are not required for standard WildRobot PPO training.

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

Hydra-driven experiment script + WildRobot config:
* Script: `examples/training_examples/jax_rl/experiment.py`
* Config: `examples/training_examples/jax_rl/conf_wildrobot.yaml`

No wrapper needed.

---

## 4. Launch training

From the project root:

```bash
uv run examples/training_examples/jax_rl/experiment.py --config-name conf_wildrobot
```

Authenticate WandB (optional but recommended for live charts):

```bash
export WANDB_API_KEY=YOUR_API_KEY
export WANDB_MODE=online    # or offline
uv run wandb login
```

During startup the script prints:
* `[Hydra] output_dir: ...`
* `[WandB] run.dir: ...` -> contains `files/wandb-history.jsonl` (live metrics file).

Pipeline overview:
1. `MjxWildRobot` env via `RLFactory`.
2. PPOJax builds & JIT-compiles fused training loop.
3. Host callbacks stream metrics if enabled.
4. Agent saved (`PPOJax_saved.pkl`).
5. Final rollout recorded as a video artifact.

---

## 5. Common tweaks & monitoring

Edit `examples/training_examples/jax_rl/conf_wildrobot.yaml`.

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

### 5.3 PPO hyperparameters

Tune:
* `hidden_layers` (model capacity)
* `lr` (learning rate)
* `clip_eps`, `num_minibatches`, `update_epochs` (stability trade-offs)

### 5.4 Live progress & WandB streaming

Flags:
* `live_wandb: true` – enable in-loop streaming via `jax.debug.callback`.
* `live_wandb_interval: 5` – log every N PPO updates.
* `debug: true` – verbose per-episode prints (noisy; default false).

Examples:
```bash
# Log every update with a smaller test run
uv run examples/training_examples/jax_rl/experiment.py \
  --config-name conf_wildrobot \
  experiment.live_wandb_interval=1 \
  experiment.total_timesteps=1000000

# Disable live streaming (only final aggregation)
uv run examples/training_examples/jax_rl/experiment.py \
  --config-name conf_wildrobot \
  experiment.live_wandb=false
```

### 5.5 Tail logs locally

Use printed `run.dir` path:
```bash
tail -f RUN_DIR/files/wandb-history.jsonl | grep -E 'Mean Episode Return|Mean Episode Length'
```

Pretty-print (requires jq):
```bash
tail -f RUN_DIR/files/wandb-history.jsonl | jq '{_step, "Mean Episode Return", "Mean Episode Length"}'
```

Offline mode & later sync:
```bash
export WANDB_MODE=offline
uv run examples/training_examples/jax_rl/experiment.py --config-name conf_wildrobot
uv run wandb sync RUN_DIR
```

---

## 6. Sanity checks

1. Devices:
```bash
uv run python -c "import jax; print(jax.devices())"
```

2. Quick test run:
```bash
uv run examples/training_examples/jax_rl/experiment.py \
  --config-name conf_wildrobot \
  experiment.total_timesteps=1000000 \
  experiment.num_envs=128 \
  experiment.live_wandb_interval=1
```

3. Streaming confirmation: open W&B URL or tail history file.

4. Final video: verify forward locomotion stability.

After PPO convergence, explore imitation variants (`jax_rl_mimic`, `jax_amp`, `jax_gail`).

---

Happy training! Adjust `live_wandb_interval` upward to reduce logging overhead on massive runs.
