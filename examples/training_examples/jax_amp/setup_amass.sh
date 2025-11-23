#!/bin/bash
# Setup script for WildRobot AMP training with AMASS data
# Run this on your Linux training machine

set -e  # Exit on error

echo "=========================================="
echo "WildRobot AMP Training Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "experiment.py" ]; then
    echo -e "${RED}Error: Please run this script from the jax_amp directory${NC}"
    echo "Expected: examples/training_examples/jax_amp/"
    exit 1
fi

echo "✓ Running from correct directory"
echo ""

# Step 1: Check Python dependencies
echo "=========================================="
echo "Step 1: Checking Python environment"
echo "=========================================="

if command -v uv &> /dev/null; then
    echo "✓ uv is installed"
    echo ""
    echo "Installing SMPL dependencies..."
    uv sync --group smpl
    echo -e "${GREEN}✓ SMPL dependencies installed${NC}"
else
    echo -e "${YELLOW}! uv not found, using pip${NC}"
    pip install -e ".[smpl]"
fi

echo ""

# Step 2: Check for AMASS and SMPL paths
echo "=========================================="
echo "Step 2: Checking AMASS and SMPL paths"
echo "=========================================="

# Check if paths are set
AMASS_PATH=$(python -c "from loco_mujoco.utils import get_amass_path; print(get_amass_path())" 2>/dev/null || echo "NOT_SET")
SMPL_PATH=$(python -c "from loco_mujoco.utils import get_smpl_model_path; print(get_smpl_model_path())" 2>/dev/null || echo "NOT_SET")

if [ "$AMASS_PATH" = "NOT_SET" ] || [ "$AMASS_PATH" = "None" ]; then
    echo -e "${YELLOW}! AMASS path not set${NC}"
    echo ""
    echo "Please set your AMASS path:"
    echo "  loco-mujoco-set-amass-path --path /path/to/amass"
    echo ""
    echo "Download AMASS from: https://amass.is.tue.mpg.de/"
    echo "(You need at least the KIT subset)"
    echo ""
    NEEDS_SETUP=1
else
    echo -e "${GREEN}✓ AMASS path: $AMASS_PATH${NC}"
fi

if [ "$SMPL_PATH" = "NOT_SET" ] || [ "$SMPL_PATH" = "None" ]; then
    echo -e "${YELLOW}! SMPL path not set${NC}"
    echo ""
    echo "Please set your SMPL model path:"
    echo "  loco-mujoco-set-smpl-model-path --path /path/to/smpl"
    echo ""
    echo "Download SMPL models from: https://smpl.is.tue.mpg.de/"
    echo "(You need SMPL+H models)"
    echo ""
    NEEDS_SETUP=1
else
    echo -e "${GREEN}✓ SMPL path: $SMPL_PATH${NC}"
fi

echo ""

# Step 3: Set converted AMASS cache path (optional but recommended)
echo "=========================================="
echo "Step 3: Setting cache directory"
echo "=========================================="

CONV_AMASS_PATH=$(python -c "from loco_mujoco.utils import get_converted_amass_path; print(get_converted_amass_path())" 2>/dev/null || echo "NOT_SET")

if [ "$CONV_AMASS_PATH" = "NOT_SET" ] || [ "$CONV_AMASS_PATH" = "None" ]; then
    echo "Setting default cache directory..."
    DEFAULT_CACHE="$HOME/.loco_mujoco/converted_amass"
    loco-mujoco-set-conv-amass-path --path "$DEFAULT_CACHE"
    echo -e "${GREEN}✓ Cache directory: $DEFAULT_CACHE${NC}"
else
    echo -e "${GREEN}✓ Cache directory: $CONV_AMASS_PATH${NC}"
fi

echo ""

# Step 4: Check GPU
echo "=========================================="
echo "Step 4: Checking GPU availability"
echo "=========================================="

python -c "
import jax
devices = jax.devices()
print(f'Found {len(devices)} device(s):')
for i, d in enumerate(devices):
    print(f'  [{i}] {d.device_kind}: {d}')

if devices[0].device_kind == 'gpu':
    print('\n✓ GPU detected - ready for training!')
else:
    print('\n⚠ No GPU detected - training will be slow on CPU')
"

echo ""

# Step 5: Verify config exists
echo "=========================================="
echo "Step 5: Verifying configuration"
echo "=========================================="

if [ -f "conf_wildrobot_amp_amass.yaml" ]; then
    echo -e "${GREEN}✓ Config file found: conf_wildrobot_amp_amass.yaml${NC}"
else
    echo -e "${RED}✗ Config file not found!${NC}"
    echo "Please copy conf_wildrobot_amp_amass.yaml to this directory"
    exit 1
fi

echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""

if [ -n "$NEEDS_SETUP" ]; then
    echo -e "${YELLOW}⚠ Action Required:${NC}"
    echo "  1. Download AMASS dataset (KIT subset minimum)"
    echo "  2. Download SMPL+H models"
    echo "  3. Run the setup commands shown above"
    echo ""
    echo "After setup, run this script again to verify"
else
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Ready to train! Run:"
    echo -e "${GREEN}  python experiment.py --config-name conf_wildrobot_amp_amass${NC}"
    echo ""
    echo "Expected first-run time:"
    echo "  - Shape fitting + retargeting: ~5 minutes"
    echo "  - Training (50M steps): ~30 minutes on RTX 3080"
    echo ""
    echo "Subsequent runs will skip shape fitting (uses cache)"
fi

echo ""
echo "=========================================="
