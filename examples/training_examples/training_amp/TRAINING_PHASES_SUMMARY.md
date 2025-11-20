# Complete Training Path for Command-Following WildRobot

Your requirements:
1. ✅ Start from standing
2. ✅ Follow commands (stop, walk slow/fast, turn left/right)
3. ✅ Fall recovery (stand up and continue)

## Current Status

**✅ Phase 1 Complete** (or in progress): Basic human-like motion learning

Your current config (`conf_wildrobot_amp_amass.yaml`) trains the robot to mimic human walking/running/turning. This is the foundation!

## Path Forward: 3-Phase Approach

### **Phase 1: Basic Motion Learning** (Current - 20-30 min)

**Config:** `conf_wildrobot_amp_amass.yaml`

**What it does:**
- Learns human-like walking, running, turning from AMASS mocap
- Discriminator ensures motion looks natural
- Foundation for all future training

**Command:**
```bash
uv run python experiment.py --config-name=conf_wildrobot_amp_amass
```

**When to move on:** After training completes successfully

---

### **Phase 2: Command Following** (Next - 30-40 min)

**Config:** `conf_wildrobot_amp_phase2_commands.yaml` ✅ Created

**What changes:**
1. **Add standing/idle motions** to dataset
2. **Switch to command-based goal** (GoalRandomRootVelocity or custom GoalDiscreteCommand)
3. **Robot learns to follow velocity commands**

**Setup:**

1. **Check for standing motions in AMASS:**
   ```bash
   ls ~/amass/KIT/3/ | grep -i stand
   ls ~/amass/KIT/3/ | grep -i idle
   ```

2. **Update config with available standing motions:**
   Edit `conf_wildrobot_amp_phase2_commands.yaml`:
   ```yaml
   rel_dataset_path:
     - "KIT/3/standing01_poses"  # Adjust based on what you have
     - "KIT/3/walking_slow08_poses"
     ...
   ```

3. **Train Phase 2:**
   ```bash
   uv run python experiment.py --config-name=conf_wildrobot_amp_phase2_commands
   ```

**What you get:**
- ✅ Robot responds to velocity commands
- ✅ Smooth transitions between speeds
- ✅ Can stop (stand still)
- ✅ Still looks human-like

**Optional: Discrete Commands**

For true discrete commands (not continuous velocities), implement custom goal:
- See: `PHASE2_COMMAND_FOLLOWING.md`
- Create `wildrobot_extensions/goals.py`
- Use `GoalDiscreteCommand` goal type

---

### **Phase 3: Fall Recovery** (Final - 1-2 hours)

**Goal:** Robot stands up after falling and continues executing command

**Approach: Curriculum Learning**

Train in stages with progressively harder initial poses:

**Stage 0: Standing** (already done in Phase 2)
```yaml
# No special initial state - starts standing
```

**Stage 1: Crouch Start**
```yaml
initial_state_type: RandomInitialPose
initial_state_params:
  pose_range:
    pelvis_tz: [0.3, 0.5]  # Lower starting height
```

**Stage 2: Sitting Start**
```yaml
initial_state_params:
  pose_range:
    pelvis_tz: [0.1, 0.3]  # Even lower
    knee_angle: [1.5, 2.0]  # Bent knees
```

**Stage 3: Lying Down**
```yaml
initial_state_params:
  pose_range:
    pelvis_tz: [0.0, 0.1]  # On ground
    pelvis_rotation: [-0.5, 0.5]  # Random orientation
```

**Add Get-Up Motions to Dataset:**

You'll need lying→standing motions from AMASS:
```yaml
rel_dataset_path:
  # Standing up motions
  - "KIT/3/lying_to_standing01_poses"
  - "KIT/317/lying_to_standing02_poses"

  # Regular motions
  - "KIT/3/walking_slow08_poses"
  ...
```

**Training Command:**
```bash
# Continue from Phase 2
uv run python experiment.py \
  --config-name=conf_wildrobot_amp_phase3_recovery \
  experiment.load_checkpoint=outputs/phase2/AMPJax_saved.pkl
```

---

## Quick Start: Simplified 2-Phase Approach

If you want faster results, combine Phase 2+3:

### **Phase 1: Motion Learning** (20-30 min)

```bash
uv run python experiment.py --config-name=conf_wildrobot_amp_amass
```

### **Phase 2: Commands + Basic Recovery** (30-40 min)

1. **Add to dataset:**
   - Standing motions
   - Get-up motions (if available)

2. **Use existing Phase 2 config:**
   ```bash
   uv run python experiment.py --config-name=conf_wildrobot_amp_phase2_commands
   ```

3. **Test command following:**
   ```python
   # After training
   from loco_mujoco.algorithms import AMPJax

   agent_conf, agent_state = AMPJax.load_agent("outputs/.../AMPJax_saved.pkl")

   # Test different velocities
   # (0, 0, 0) = stop
   # (0.5, 0, 0) = walk slow
   # (2.0, 0, 0) = walk fast
   # (0.5, 0, 0.5) = turn left
   ```

---

## Recommended Path

**For your requirements, I recommend:**

1. **✅ Complete Phase 1** (current training)
   - Get solid human-like motion foundation
   - ~20-30 minutes

2. **✅ Move to Phase 2** (command following)
   - Add standing motions
   - Switch to velocity-based goals
   - ~30-40 minutes

3. **✅ Add fall recovery later** (Phase 3)
   - Implement curriculum learning
   - Add get-up motions
   - ~1-2 hours

**Total time:** 2-3 hours for complete system

---

## Key Files

- `conf_wildrobot_amp_amass.yaml` - Phase 1 (current)
- `conf_wildrobot_amp_phase2_commands.yaml` - Phase 2 (created)
- `PHASE2_COMMAND_FOLLOWING.md` - Discrete command implementation
- `WILDROBOT_TRAINING_STRATEGY.md` - Full 4-phase strategy

---

## What to Do Now

**Option A: Complete Phase 1 first (Recommended)**
```bash
# Finish current training
uv run python experiment.py --config-name=conf_wildrobot_amp_amass

# Then move to Phase 2
```

**Option B: Skip to Phase 2 immediately**
```bash
# Check for standing motions
ls ~/amass/KIT/3/ | grep stand

# Update Phase 2 config with correct motion files
# Then train
uv run python experiment.py --config-name=conf_wildrobot_amp_phase2_commands
```

**My recommendation:** Complete Phase 1, verify it works, then build Phase 2 on top of it. This gives you:
1. Solid foundation of human-like motion
2. Clear debugging (know Phase 1 works)
3. Can use Phase 1 checkpoint as starting point for Phase 2
