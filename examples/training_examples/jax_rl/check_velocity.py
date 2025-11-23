"""Quick diagnostic to check if expert data has expected velocities."""
import sys
import numpy as np
from loco_mujoco.trajectory import Trajectory

if len(sys.argv) < 2:
    print("Usage: python check_velocity.py <trajectory.npz>")
    sys.exit(1)

traj_path = sys.argv[1]
print(f"Analyzing: {traj_path}")

traj = Trajectory.load(traj_path, backend=np)

root_pos = traj.data.xpos[:, 0, :]
root_vel = np.diff(root_pos, axis=0) / 0.02
forward_vel = root_vel[:, 0]

mean_vel = np.mean(forward_vel)
median_vel = np.median(forward_vel)

print(f"\nForward velocity:")
print(f"  Mean:   {mean_vel:.3f} m/s")
print(f"  Median: {median_vel:.3f} m/s")
print(f"  Std:    {np.std(forward_vel):.3f} m/s")
print(f"  Range:  [{np.min(forward_vel):.3f}, {np.max(forward_vel):.3f}] m/s")

if mean_vel < 0.6:
    print(f"\n❌ ISSUE: Velocity too low (expected 0.6-1.2 m/s)")
    print("   Policy likely couldn't handle higher velocity goals")
    print("   → Need to retrain Step 1 with higher targets")
elif mean_vel > 1.3:
    print(f"\n⚠️  WARNING: Velocity higher than expected")
else:
    print(f"\n✅ Velocity in expected range")
