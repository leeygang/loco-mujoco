#!/bin/bash
# WildRobot AMASS Setup Script
# Run this to set up AMASS retargeting for WildRobot

set -e

echo "=========================================="
echo "WildRobot AMASS Setup"
echo "=========================================="
echo ""

# Navigate to repo root
cd "$(dirname "$0")/../../.."

# Step 1: Install PyTorch CPU
echo "Step 1/5: Installing PyTorch CPU..."
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Step 2: Install SMPL dependencies
echo ""
echo "Step 2/5: Installing SMPL dependencies..."
echo "⚠️  By continuing, you agree to the SMPL Software Copyright License"
echo "   https://github.com/vchoutas/smplx"
read -p "Press Enter to continue or Ctrl+C to cancel..."
uv sync --group smpl

# Step 3: Create directories
echo ""
echo "Step 3/5: Creating directories..."
mkdir -p ~/smpl
mkdir -p ~/amass
mkdir -p ~/amass_converted

# Step 4: Configure paths
echo ""
echo "Step 4/5: Configuring LocoMuJoCo paths..."
loco-mujoco-set-smpl-model-path --path ~/smpl
loco-mujoco-set-amass-path --path ~/amass
loco-mujoco-set-conv-amass-path --path ~/amass_converted

echo ""
echo "=========================================="
echo "Automated setup complete!"
echo "=========================================="
echo ""
echo "Next steps (MANUAL):"
echo ""
echo "1. Download SMPL-H models:"
echo "   → Visit: https://mano.is.tue.mpg.de/download.php"
echo "   → Download: Extended SMPL+H model + Models & Code"
echo "   → Extract to: ~/smpl/"
echo ""
echo "2. Download AMASS datasets:"
echo "   → Visit: https://amass.is.tue.mpg.de/"
echo "   → Download: KIT, CMU, BMLrub (SMPL-H G version)"
echo "   → Extract to: ~/amass/"
echo ""
echo "3. Generate SMPL-H neutral model:"
echo "   → cd loco_mujoco/smpl"
echo "   → chmod u+x install_smplh.sh"
echo "   → ./install_smplh.sh"
echo ""
echo "4. Verify setup:"
echo "   → ls ~/smpl/models/SMPLH_NEUTRAL.pkl"
echo "   → ls ~/amass/KIT/3/"
echo ""
echo "5. Start training:"
echo "   → cd examples/training_examples/training_amp"
echo "   → python experiment.py --config-name=conf_wildrobot_amp_amass"
echo ""
echo "📖 See SETUP_AMASS_FOR_WILDROBOT.md for detailed instructions"
echo ""
