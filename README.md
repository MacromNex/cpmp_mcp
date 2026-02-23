# CPMP MCP

> Cyclic Peptide Membrane Permeability prediction toolkit - MCP tools for cyclic peptide computational analysis

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

CPMP MCP provides comprehensive computational tools for predicting membrane permeability of cyclic peptides across four different biological assays. The toolkit uses a Molecular Attention Transformer (MAT) neural network to analyze cyclic peptide structures and predict their permeability properties, making it invaluable for drug discovery and development workflows.

### Features
- **Multi-assay permeability prediction** (PAMPA, Caco-2, RRCK, MDCK)
- **Molecular property calculation** using RDKit
- **Data preprocessing and featurization** for machine learning
- **Batch processing for virtual screening** with job management
- **Comprehensive analysis and visualization** capabilities

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   └── server.py           # MCP server (15 tools)
├── scripts/
│   ├── predict_all_assays.py      # Multi-assay prediction
│   ├── predict_single_assay.py    # Single assay analysis
│   ├── preprocess_data.py          # Data preprocessing
│   ├── batch_analysis.py           # Comprehensive analysis
│   └── lib/                        # Shared utilities (23 functions)
├── examples/
│   └── data/               # Demo data
│       ├── sample_cyclic_peptides.csv    # Sample SMILES with permeability
│       └── sequences/      # Assay-specific sample datasets
├── configs/                # Configuration files
│   ├── predict_all_assays_config.json
│   ├── predict_single_assay_config.json
│   ├── preprocess_data_config.json
│   ├── batch_analysis_config.json
│   └── default_config.json
└── repo/                   # Original CPMP repository
    └── CPMP/               # Core algorithms and pre-trained models
```

---

## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

### Manual Setup (Advanced)

For manual installation or customization, follow these steps.

#### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- RDKit (installed automatically)
- 8GB+ RAM for molecular featurization
- GPU optional for acceleration

#### Create Environment
Please follow the information in `reports/step3_environment.md` for detailed setup. An example workflow is shown below.

```bash
# Navigate to the MCP directory
cd /path/to/cpmp_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install core scientific packages
mamba install -c conda-forge \
    pandas=2.2.3 \
    scikit-learn=1.6.1 \
    matplotlib \
    seaborn \
    rdkit=2024.3.2 \
    -y

# Install PyTorch
mamba install -c pytorch -c nvidia \
    pytorch=2.5.1 \
    pytorch-cuda=12.4 \
    -y

# Install additional utilities
pip install numpy scipy

# Install MCP dependencies
pip install fastmcp loguru
```

---

## Model Setup

The CPMP toolkit requires pre-trained model weights to make predictions. The model files are stored with a specific naming convention from the original repository.

### Model File Locations

- **Source**: `repo/CPMP/saved_model/` contains the original model weights
- **Symlinks**: `repo/CPMP/model_checkpoints/` contains symlinks used by the scripts

### Model File Naming

The original CPMP repository uses the naming convention `{assay}.best_wegiht.pth` (note: the typo "wegiht" is intentional and comes from the original repository). The scripts expect files at `model_checkpoints/{assay}_uff_ig_true_final.pt`.

### Expected Files

| Saved Model | Symlink |
|-------------|---------|
| `saved_model/pampa.best_wegiht.pth` | `model_checkpoints/pampa_uff_ig_true_final.pt` |
| `saved_model/caco2.best_wegiht.pth` | `model_checkpoints/caco2_uff_ig_true_final.pt` |
| `saved_model/rrck.best_wegiht.pth` | `model_checkpoints/rrck_uff_ig_true_final.pt` |
| `saved_model/mdck.best_wegiht.pth` | `model_checkpoints/mdck_uff_ig_true_final.pt` |

### Creating Symlinks Manually

If the `quick_setup.sh` script didn't create the symlinks, you can create them manually:

```bash
cd /path/to/cpmp_mcp
mkdir -p repo/CPMP/model_checkpoints

for assay in pampa caco2 rrck mdck; do
    ln -sf "../saved_model/${assay}.best_wegiht.pth" \
           "repo/CPMP/model_checkpoints/${assay}_uff_ig_true_final.pt"
done
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/predict_all_assays.py` | Predict permeability across all 4 assays | See below |
| `scripts/predict_single_assay.py` | Focused analysis for specific assay | See below |
| `scripts/preprocess_data.py` | Process and split datasets for training | See below |
| `scripts/batch_analysis.py` | Comprehensive analysis with visualizations | See below |

### Script Examples

#### Predict All Assays

```bash
# Activate environment
mamba activate ./env

# Run multi-assay prediction
python scripts/predict_all_assays.py \
  --input examples/data/sample_cyclic_peptides.csv \
  --output results/all_predictions.csv \
  --device cpu
```

**Parameters:**
- `--input, -i`: Input CSV with 'smiles' column (required)
- `--output, -o`: Output file path (optional)
- `--device`: Computing device - cpu or cuda (default: cpu)
- `--batch-size`: Batch size for processing (default: 32)

#### Single Assay Analysis

```bash
python scripts/predict_single_assay.py \
  --assay caco2 \
  --input examples/data/sample_cyclic_peptides.csv \
  --with-labels \
  --device cpu
```

**Parameters:**
- `--assay`: Assay name - pampa, caco2, rrck, or mdck (required)
- `--input, -i`: Input CSV file (required)
- `--with-labels`: Include binary classification (default: True)
- `--output, -o`: Output file path (optional)

#### Data Preprocessing

```bash
python scripts/preprocess_data.py \
  --dataset examples/data/sample_cyclic_peptides.csv \
  --output-dir results/preprocessing \
  --skip-featurization
```

**Parameters:**
- `--dataset`: Input dataset CSV file (required)
- `--output-dir`: Output directory for splits (required)
- `--skip-featurization`: Skip molecular featurization for speed
- `--train-size`: Training fraction (default: 0.7)
- `--val-size`: Validation fraction (default: 0.1)
- `--test-size`: Test fraction (default: 0.2)

#### Comprehensive Analysis

```bash
python scripts/batch_analysis.py \
  --input examples/data/sample_cyclic_peptides.csv \
  --output-dir results/analysis \
  --calculate-properties \
  --create-visualizations
```

**Parameters:**
- `--input, -i`: Input CSV file (required)
- `--output-dir`: Analysis output directory (required)
- `--calculate-properties`: Calculate molecular properties
- `--create-visualizations`: Generate analysis plots

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name cycpep-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/path/to/cpmp_mcp/env/bin/python",
      "args": ["/path/to/cpmp_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from cycpep-tools?
```

#### Property Calculation (Fast)
```
Calculate molecular properties for this cyclic peptide:
CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC1=O
```

#### Single Assay Prediction (Fast)
```
Predict Caco-2 permeability for the cyclic peptides in @examples/data/sample_cyclic_peptides.csv
```

#### All Assays Prediction (Submit API)
```
Submit a job to predict permeability across all 4 assays for @examples/data/sample_cyclic_peptides.csv
```

#### Check Job Status
```
Check the status of job abc12345
```

#### Batch Processing
```
Submit a comprehensive analysis job for @examples/data/sample_cyclic_peptides.csv with molecular properties and visualizations
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sample_cyclic_peptides.csv` | Main sample dataset (19 molecules) |
| `@examples/data/sequences/pampa_sample.csv` | PAMPA-specific samples |
| `@configs/predict_all_assays_config.json` | Multi-assay configuration |
| `@results/` | Output directory for analysis |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/path/to/cpmp_mcp/env/bin/python",
      "args": ["/path/to/cpmp_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Validate cyclic peptide SMILES: "CC(=O)NC1CCCC1C(=O)O"
> Predict single assay permeability for caco2 assay
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `validate_cyclic_peptide_smiles` | Validate SMILES structure | `smiles_list` |
| `preprocess_cyclic_peptide_data` | Preprocess and split datasets | `input_file`, `train_size`, `skip_featurization` |
| `predict_single_assay_permeability` | Predict specific assay | `assay`, `input_file`, `with_labels` |
| `predict_all_assays_permeability` | Predict all 4 assays | `input_file`, `device`, `batch_size` |
| `analyze_cyclic_peptide_batch` | Comprehensive analysis | `input_file`, `calculate_properties`, `create_visualizations` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes or large datasets):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_preprocess_data` | Background preprocessing | `input_file`, `output_dir`, `job_name` |
| `submit_single_assay_prediction` | Background single assay | `assay`, `input_file`, `job_name` |
| `submit_all_assays_prediction` | Background all assays | `input_file`, `job_name` |
| `submit_batch_analysis` | Background analysis | `input_file`, `output_dir`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

### Utility Tools

| Tool | Description |
|------|-------------|
| `get_server_info` | Server status and capabilities |

---

## Examples

### Example 1: Quick Validation and Property Calculation

**Goal:** Validate and calculate properties for a single cyclic peptide

**Using Script:**
```bash
python scripts/predict_all_assays.py \
  --input examples/data/sample_cyclic_peptides.csv \
  --output results/properties.csv
```

**Using MCP (in Claude Code):**
```
First validate this SMILES: "CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC1=O"

Then predict all assay permeabilities for it.
```

**Expected Output:**
- SMILES validation: Valid cyclic peptide
- PAMPA: -5.85 log cm/s (permeable)
- Caco-2: -5.69 log cm/s (permeable)
- RRCK: -5.77 log cm/s (permeable)
- MDCK: -6.25 log cm/s (borderline)

### Example 2: Single Assay Focused Analysis

**Goal:** Detailed Caco-2 analysis for intestinal absorption prediction

**Using Script:**
```bash
python scripts/predict_single_assay.py \
  --assay caco2 \
  --input examples/data/sample_cyclic_peptides.csv \
  --with-labels \
  --output results/caco2_analysis.csv
```

**Using MCP (in Claude Code):**
```
Analyze @examples/data/sample_cyclic_peptides.csv for Caco-2 permeability.
Include binary classification and provide detailed metrics.
```

**Expected Output:**
- Continuous permeability values (log cm/s)
- Binary classification (permeable/non-permeable)
- Probability scores for classification
- Model performance metrics (if ground truth available)

### Example 3: Large Dataset Virtual Screening Pipeline

**Goal:** Screen a library of cyclic peptides for oral bioavailability

**Using MCP (in Claude Code):**
```
I want to process @examples/data/sample_cyclic_peptides.csv through a complete screening pipeline:

1. First preprocess the data with 70/10/20 train/val/test splits
2. Submit comprehensive analysis with molecular properties and visualizations
3. Predict permeability across all 4 assays
4. Check jobs and show results when complete
```

**Expected Workflow:**
1. Data preprocessing: train/val/test splits saved
2. Molecular properties: MW, LogP, TPSA, HBD, HBA calculated
3. Visualizations: correlation heatmaps, distribution plots
4. All assay predictions: PAMPA, Caco-2, RRCK, MDCK results
5. Analysis report: Statistical summary and recommendations

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `sample_cyclic_peptides.csv` | Main dataset (19 molecules) | All prediction tools |
| `sequences/pampa_sample.csv` | PAMPA-specific samples | Single assay tools |
| `sequences/mdck_sample.csv` | MDCK-specific samples | Single assay tools |
| `sequences/rrck_sample.csv` | RRCK-specific samples | Single assay tools |

### Sample Data Properties
- **Molecular weight range**: 400-1200 g/mol
- **Permeability range**: -7.5 to -4.5 log cm/s
- **Ring sizes**: 6-12 amino acids
- **Chemical diversity**: Various side chains and modifications

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `predict_all_assays_config.json` | Multi-assay prediction | Device, batch sizes, model paths |
| `predict_single_assay_config.json` | Single assay analysis | Assay settings, thresholds |
| `preprocess_data_config.json` | Data preprocessing | Split ratios, filtering criteria |
| `batch_analysis_config.json` | Comprehensive analysis | Property lists, visualization settings |
| `default_config.json` | Global defaults | Paths, molecular properties |

### Config Example

```json
{
  "device": "cpu",
  "batch_size": 32,
  "assays": {
    "caco2": {
      "name": "Caco-2",
      "full_name": "Human colon adenocarcinoma cell line",
      "threshold": -6.0,
      "model_path": "model_checkpoints/caco2_uff_ig_true_final.pt"
    }
  },
  "molecular_properties": [
    "molecular_weight", "logp", "tpsa", "hbd", "hba",
    "rotatable_bonds", "aromatic_rings"
  ]
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install fastmcp loguru
mamba install -c conda-forge rdkit=2024.3.2 -y
```

**Problem:** RDKit import errors
```bash
# Install RDKit from conda-forge (not pip)
mamba install -c conda-forge rdkit -y

# Verify installation
python -c "from rdkit import Chem; print('RDKit version:', Chem.__version__)"
```

**Problem:** Import errors
```bash
# Verify CPMP modules
cd /path/to/cpmp_mcp
python -c "
import sys
sys.path.append('repo/CPMP')
from featurization.data_utils import load_data_from_df
print('CPMP modules working')
"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove cycpep-tools
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Invalid SMILES error
```
Always validate SMILES first using validate_cyclic_peptide_smiles.
For cyclic peptides, ensure ring closure is properly specified.
Example valid: "CC(C)C[C@@H]1NC(=O)...NC1=O"
```

**Problem:** Tools not working
```bash
# Test server directly
mamba activate ./env
python -c "
from src.server import mcp
from src.jobs.manager import job_manager
print('Server modules loaded successfully')
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
tail -20 jobs/<job_id>/job.log
```

**Problem:** Job failed due to missing models
```
Download pre-trained model weights and place in repo/CPMP/saved_model/
Required: pampa.best_wegiht.pth, caco2.best_wegiht.pth, etc.
```

**Problem:** Model checkpoints not found (symlinks missing)
```bash
# Recreate model symlinks
cd /path/to/cpmp_mcp
mkdir -p repo/CPMP/model_checkpoints

for assay in pampa caco2 rrck mdck; do
    if [ -f "repo/CPMP/saved_model/${assay}.best_wegiht.pth" ]; then
        ln -sf "../saved_model/${assay}.best_wegiht.pth" \
               "repo/CPMP/model_checkpoints/${assay}_uff_ig_true_final.pt"
        echo "Created symlink for $assay"
    else
        echo "WARNING: Model file for $assay not found"
    fi
done
```

**Problem:** Out of memory during featurization
```bash
# Use smaller batch size
python scripts/predict_all_assays.py --batch-size 16

# Use CPU instead of GPU
python scripts/predict_all_assays.py --device cpu

# Process smaller datasets (<100 molecules)
```

### Performance Issues

**Problem:** Slow prediction times
```bash
# Use GPU if available
python scripts/predict_all_assays.py --device cuda

# Increase batch size for throughput
python scripts/predict_all_assays.py --batch-size 64

# Skip featurization for preprocessing-only
python scripts/preprocess_data.py --skip-featurization
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test individual scripts
python scripts/predict_all_assays.py --help
python scripts/preprocess_data.py --help

# Test MCP server
fastmcp dev src/server.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
mamba activate ./env
fastmcp dev src/server.py

# Or with explicit environment
mamba run --prefix ./env fastmcp dev src/server.py
```

### Performance Benchmarks

| Operation | Dataset Size | CPU Time | GPU Time | Memory |
|-----------|--------------|----------|----------|---------|
| Validation | 100 SMILES | 5 sec | N/A | 100MB |
| Preprocessing | 100 molecules | 15 min | N/A | 1GB |
| Single assay | 100 molecules | 8 min | 3 min | 2GB |
| All assays | 100 molecules | 25 min | 10 min | 4GB |
| Batch analysis | 100 molecules | 12 min | 5 min | 3GB |

---

## License

MIT License - see original CPMP repository for details

## Credits

Based on [CPMP (Cyclic Peptide Membrane Permeability)](https://github.com/original-repo-url) - Molecular Attention Transformer for cyclic peptide permeability prediction

---

*🧬 Generated with [Claude Code](https://claude.com/claude-code) - Advanced Cyclic Peptide Analysis Toolkit*