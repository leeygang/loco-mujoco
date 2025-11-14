# WildRobot – Replaying AMASS Datasets

This guide explains how to replay AMASS/SMPL-H motion on the **WildRobot** environment using the scripts in `examples/replay_datasets`.

It covers:

- Installing the right dependencies
- Pointing LocoMuJoCo to your AMASS and SMPL models
- Cleaning and regenerating WildRobot AMASS caches
- Running the `wildrobot_smpl_example.py` replay script
- Optional sanity checks and diagnostics

---

## 1. Prerequisites

### 1.1. Base installation (uv)

This project is set up to work well with **uv** for dependency management.

From the repository root, create/sync the base environment:

```bash
uv sync
```

This installs LocoMuJoCo with CPU JAX and MuJoCo according to `pyproject.toml`.

If you prefer plain `pip`, you can instead do:

```bash
pip install -e .
```

### 1.2. Optional: NVIDIA CUDA GPU support

If you have an NVIDIA GPU and a CUDA 12 runtime, you can install the GPU dependencies defined in `pyproject.toml`.

Using **uv** (recommended):

```bash
uv sync --group gpu
```

This uses the `[dependency-groups].gpu` section in `pyproject.toml`.

Using **pip** extras (if you are not using uv):

```bash
pip install -e ".[gpu]"
```

Both options pull in `jax[cuda12]` and enable GPU-accelerated MJX training and simulation.

> Note: Make sure your CUDA / driver stack matches the JAX CUDA wheels. See the JAX installation docs if you encounter version errors.

### 1.3. SMPL-H and AMASS datasets

WildRobot replay uses SMPL-H body models and AMASS motion capture data. You need to:

1. **Obtain SMPL-H models** (license required):
   - Follow the instructions at https://github.com/vchoutas/smplx to download `SMPLH_NEUTRAL.pkl`.
   - Place the SMPL-H files under a directory of your choice, e.g. `~/data/smpl_models`.

2. **Obtain AMASS** (license required):
   - Download AMASS sequences you want to replay (e.g. the KIT dataset) and place them under a root directory, e.g. `~/data/amass`.

3. **Install SMPL extras and point LocoMuJoCo to these paths.**

   First, install the SMPL optional dependencies. With **uv**:

   ```bash
   uv sync --group smpl
   ```

   or with **pip**:

   ```bash
   pip install -e ".[smpl]"
   ```

   Then, point LocoMuJoCo to your datasets using the provided CLI helpers:

   ```bash
   loco-mujoco-set-smpl-model-path
   loco-mujoco-set-amass-path
   loco-mujoco-set-conv-amass-path
   ```

   These commands will prompt for the path, or you can pass it as an argument (see `loco_mujoco/utils.py` for details). The converted AMASS path is where LocoMuJoCo will store cached, retargeted trajectories and fitted shape files.

---

## 2. WildRobot AMASS cache layout

When you first run a WildRobot AMASS replay script, LocoMuJoCo will:

1. Fit a SMPL-H body shape for WildRobot and save it as:

   ```
   <LOCOMUJOCO_CONVERTED_AMASS_PATH>/WildRobot/shape_optimized.pkl
   ```

2. Retarget selected AMASS sequences to WildRobot and cache them as `.npz` files in the same folder, for example:

   ```
   <LOCOMUJOCO_CONVERTED_AMASS_PATH>/WildRobot/KIT_3_walking_slow08_poses.npz
   ```

The exact filenames depend on the AMASS sequences you configure.

If you later change the WildRobot model, the SMPL retargeting configuration, or the mimic site setup, you should **clear the cache** so these artifacts are recomputed.

---

## 3. Cleaning WildRobot AMASS cache

There is a small helper script dedicated to cleaning the AMASS cache for WildRobot and MjxWildRobot. From the repo root:

```bash
python -m loco_mujoco.models.wildrobot.clean_amass_cache
```

By default this will remove the WildRobot/MjxWildRobot `shape_optimized.pkl` file(s). You can also remove all retargeted `.npz` files by passing the appropriate flag (see the script header / `--help`):

```bash
python -m loco_mujoco.models.wildrobot.clean_amass_cache --all
```

This is useful when:

- You modify `wildrobot.xml` (e.g. site positions/axes).
- You change the set or ordering of mimic sites in the WildRobot configuration.
- You update the SMPL retargeting logic.

After cleaning, the next replay run will refit `shape_optimized.pkl` and regenerate the cached trajectories.

---

## 4. Running the WildRobot replay example

The main example script is `examples/replay_datasets/wildrobot_smpl_example.py`.

From the repository root:

```bash
cd examples/replay_datasets
python wildrobot_smpl_example.py
```

What this script does:

- Selects one or more AMASS sequences (by default something like `KIT/3/walking_slow08_poses`).
- Uses `ImitationFactory.make("WildRobot", ...)` to construct a WildRobot environment with the appropriate trajectory handler.
- Ensures a WildRobot shape file exists at `<LOCOMUJOCO_CONVERTED_AMASS_PATH>/WildRobot/shape_optimized.pkl` (fitting it if needed).
- Loads the retargeted AMASS trajectories from the converted AMASS cache.
- Calls `env.play_trajectory(...)` to replay the motion in MuJoCo with rendering.

You can edit `amass_sequences` inside `wildrobot_smpl_example.py` to point to different AMASS clips as long as they are present under your `LOCOMUJOCO_AMASS_PATH`.

---

## 5. Sanity checks and diagnostics

### 5.1. Axis / site inspection

To check that WildRobot’s mimic sites and pelvis frame are consistent with the retargeting assumptions, you can use the `inspect_axis.py` script.

From the repository root:

```bash
cd examples/replay_datasets
python inspect_axis.py
```

This will print, for both Unitree and WildRobot:

- Site positions and orientation matrices in the world frame.
- The cached pelvis transform stored in `shape_optimized.pkl`.

You can redirect this output to a file for later inspection, for example:

```bash
python inspect_axis.py > output.txt
```

### 5.2. General replay sanity

If the robot motion looks incorrect (e.g. walking backwards, twisted axes):

1. **Clean the AMASS cache** for WildRobot/MjxWildRobot:

   ```bash
   python -m loco_mujoco.models.wildrobot.clean_amass_cache --all
   ```

2. Re-run the replay example:

   ```bash
   cd examples/replay_datasets
   python wildrobot_smpl_example.py
   ```

3. If issues persist, check `inspect_axis.py` output and verify that:
   - The pelvis mimic site frame in `wildrobot.xml` is identity-aligned as expected.
   - The number and ordering of mimic sites in the environment configuration matches the SMPL retargeter’s expectations.

---

## 6. Tips and troubleshooting

- **Performance:**
  - For faster iteration on a laptop, you can temporarily limit the number of steps per episode in `env.play_trajectory` or reduce the number of episodes.
  - On GPU, ensure you installed the `[gpu]` extra and that JAX sees your CUDA device. You can test this quickly:

    ```bash
    python -c "import jax; print(jax.devices())"
    ```

- **Path issues:**
  - If the script complains about missing AMASS or SMPL paths, re-run:

    ```bash
    loco-mujoco-set-smpl-model-path
    loco-mujoco-set-amass-path
    loco-mujoco-set-conv-amass-path
    ```

- **Cache mismatches:**
  - Any time you change the WildRobot model, SMPL retargeting config, or mimic sites, clean the cache and rerun the example.

---

With these steps, you should be able to reliably replay AMASS motions on WildRobot, regenerate shape and trajectory caches when needed, and debug most alignment issues using the provided tools.
