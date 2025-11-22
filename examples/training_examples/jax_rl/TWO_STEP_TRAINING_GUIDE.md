# Two-Step Training Approach for Human-Like WildRobot Walking

## Problem Summary
- **RSI with human data causes backward walking** due to orientation mismatch
- **Size difference** (humans 3.4× taller) makes direct pose copying impossible
- **Need**: First get stable walking, THEN add human-like style

---

## STEP 1: Stable Forward Walking (NO Imitation)

### Goal
Get WildRobot walking forward reliably, ignoring human-like motion for now.

### Config
`conf_step1_stable_walking.yaml`

### Key Points
✅ **RLFactory** - No trajectory loading
✅ **DefaultInitialStateHandler** - Robot starts upright, facing +X
✅ **Balanced rewards** - Velocity tracking + stability + smoothness
✅ **Expected results**:
   - Mean Forward Vel: +0.3 to +0.6 m/s (POSITIVE!)
   - Episode Length: 200-400 steps (stable walking)
   - Episode Return: 300-600

### Run
```bash
python experiment.py --config-name conf_step1_stable_walking
```

### Success Criteria
- ✅ Consistent forward walking (Mean Forward Vel > 0.3 m/s)
- ✅ Stable gait (Episode Length > 200 steps)
- ✅ No falling (Root Height stable ~0.43m)

---

## STEP 2: Add Human-Like Motion (AFTER Step 1 Works)

### Goal
Make the walking motion look more natural/human-like while maintaining forward progress.

### Approach: AMP (Adversarial Motion Priors)
**Why AMP, not DeepMimic?**
- ❌ DeepMimic: Exact pose tracking (impossible with size difference)
- ✅ AMP: Learns abstract "style" features (works across different morphologies)

### How AMP Handles Size Differences

AMP uses a **discriminator** that learns to distinguish:
- **Expert motion** (WildRobot reference from Step 1)
- **Policy motion** (current WildRobot behavior)

The discriminator looks at **kinematic features**, not absolute positions:
- Joint velocities (relative motion)
- Joint accelerations (smoothness)
- Foot contact patterns (gait rhythm)
- Body orientation changes (balance)

**These features are size-invariant!**

### Data Pipeline

**Using WildRobot Expert Data (from Step 1)**
1. Take the trained policy from Step 1
2. Generate 200 episodes of WildRobot walking
3. Save as expert demonstrations
4. Use for AMP discriminator training

**Generate expert data:**
```bash
cd examples/training_examples/jax_rl
python generate_wildrobot_dataset.py \
    --policy_path outputs/2025-11-22/11-19-29/PPOJax_saved.pkl \
    --num_episodes 200 \
    --output_dir wildrobot_expert_motions
```

**Convert HDF5 to npz format:**
```bash
cd examples/training_examples/jax_rl
python convert_expert_data.py \
    --input wildrobot_expert_motions/wildrobot_expert_dataset.h5 \
    --output wildrobot_expert_traj.npz
```

### Config
Located in **`examples/training_examples/jax_amp/conf_wildrobot_step2.yaml`**

Uses AMPJax algorithm with:
- `proportion_env_reward: 0.5` → 50% task reward + 50% discriminator reward
- Same LocomotionReward as Step 1
- WildRobot expert dataset from Step 1

### Run Training

**Quick verification (2 minutes):**
```bash
cd examples/training_examples/jax_amp
uv run experiment.py --config-name conf_wildrobot_step2 \
  'experiment.total_timesteps=2e5' \
  'experiment.num_envs=512' \
  'experiment.live_wandb_interval=1' \
  'wandb.project=wildrobot-step2-quickcheck'
```

**Full training (20-25 minutes):**
```bash
cd examples/training_examples/jax_amp
uv run experiment.py --config-name conf_wildrobot_step2
```

### Expected Results
- Forward walking maintained (from Step 1)
- More natural foot placement
- Smoother transitions
- Human-like gait rhythm

---

## Why This Approach Works

### 1. Separates Concerns
- **Step 1**: Learn basic locomotion (direction, stability)
- **Step 2**: Refine motion style (human-like appearance)

### 2. Avoids RSI Bug
- No `TrajectoryInitialStateHandler` with human data
- Robot always starts facing +X (forward)
- Velocity measured correctly in local frame

### 3. Handles Size Differences
- AMP learns **style**, not **scale**
- Discriminator sees:
  - Joint angular velocities ✅ (same for any size)
  - Foot contact timing ✅ (gait pattern)
  - Body orientation ✅ (balance strategy)
- Discriminator ignores:
  - Absolute positions ❌ (different scales)
  - Step length ❌ (size-dependent)
  - Body height ❌ (morphology-specific)

### 4. Proven Approach
- Used in "AMP: Adversarial Motion Priors" (Peng et al., 2021)
- Successfully transfers motion from:
  - Humans → simulated humanoids
  - Dogs → quadrupeds
  - Different sized characters

---

## Alternative: Style Feature Matching

If AMP is too complex, you can manually define style features:

```yaml
reward_type: StyleFeatureReward
reward_params:
  # Match gait rhythm (not size-dependent)
  foot_contact_frequency_weight: 1.0
  target_frequency: 1.8  # Hz, from human walking

  # Match smoothness (not size-dependent)
  joint_acceleration_smoothness_weight: 0.5

  # Match symmetry (not size-dependent)
  left_right_symmetry_weight: 0.3

  # Still optimize for forward velocity
  velocity_tracking_weight: 2.0
```

This requires custom reward implementation but gives you control.

---

## Next Steps

1. **Run Step 1** (stable walking)
2. **Verify forward motion** (Mean Forward Vel > 0)
3. **Generate WildRobot expert data** (from trained policy)
4. **Run Step 2** (add human-like style with AMP)
5. **Evaluate** (check if motion looks more natural)

---

## References

- **AMP Paper**: https://arxiv.org/abs/2104.02180
- **LocoMuJoCo AMP example**: `examples/training_examples/jax_rl_amp/`
- **Expert data generation**: `generate_wildrobot_dataset.py` (provided)
