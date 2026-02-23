#!/bin/bash
# Quick Setup Script for CPMP MCP
# CPMP: Cyclic Peptide Membrane Permeability Prediction using Deep Learning (MAT)
# Predicts PAMPA, Caco-2, RRCK, and MDCK membrane permeability
# Source: https://github.com/panda1103/CPMP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up CPMP MCP ==="

# Step 1: Create Python environment
echo "[1/7] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install core dependencies via conda/mamba
echo "[2/7] Installing pandas, scikit-learn, matplotlib, seaborn, rdkit..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c conda-forge pandas=2.2.3 scikit-learn=1.6.1 matplotlib seaborn rdkit=2024.3.2 -y) || \
./env/bin/pip install pandas==2.2.3 scikit-learn==1.6.1 matplotlib seaborn

# Step 3: Install PyTorch with CUDA support
echo "[3/7] Installing PyTorch 2.5.1 with CUDA 12.4..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c pytorch -c nvidia pytorch=2.5.1 pytorch-cuda=12.4 -y) || \
./env/bin/pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 4: Install additional Python packages
echo "[4/7] Installing numpy, scipy, fastmcp, loguru..."
./env/bin/pip install numpy scipy
./env/bin/pip install fastmcp loguru

# Step 5: Ensure fastmcp is properly installed
echo "[5/7] Finalizing fastmcp installation..."
./env/bin/pip install --ignore-installed fastmcp

# Step 6: Create model symlinks
echo "[6/7] Creating model checkpoint symlinks..."
MODEL_DIR="repo/CPMP/model_checkpoints"
SAVED_DIR="repo/CPMP/saved_model"
mkdir -p "$MODEL_DIR"

for assay in pampa caco2 rrck mdck; do
    if [ -f "$SAVED_DIR/${assay}.best_wegiht.pth" ]; then
        ln -sf "../saved_model/${assay}.best_wegiht.pth" "$MODEL_DIR/${assay}_uff_ig_true_final.pt"
        echo "   Created: ${assay}_uff_ig_true_final.pt -> ${assay}.best_wegiht.pth"
    else
        echo "   WARNING: $SAVED_DIR/${assay}.best_wegiht.pth not found"
    fi
done

# Step 7: Verify installation
echo "[7/7] Verifying installation..."
./env/bin/python -c "
import sys
sys.path.insert(0, 'repo/CPMP')
from model.transformer import make_model
from featurization.data_utils import load_data_from_smiles
print('   Python imports: OK')
" || echo "   WARNING: Import verification failed"

echo ""
echo "=== CPMP MCP Setup Complete ==="
echo "To run the MCP server: ./env/bin/python src/server.py"
