# CPMP Examples and Use Cases

This directory contains practical examples and use cases for the CPMP (Cyclic Peptide Membrane Permeability) prediction toolkit. The CPMP model uses Molecular Attention Transformer (MAT) architecture to predict membrane permeability across four different assay types.

## 🧬 About CPMP

CPMP predicts cyclic peptide membrane permeability for four key assays:
- **PAMPA** (Parallel Artificial Membrane Permeability Assay): Passive diffusion through artificial lipid membrane
- **Caco-2** (Human colon adenocarcinoma cell line): Intestinal absorption and oral bioavailability
- **RRCK** (Ralph Russ Canine Kidney cells): Blood-brain barrier permeability
- **MDCK** (Madin-Darby Canine Kidney cells): General membrane permeability

### Model Performance
- **PAMPA**: R² = 0.67
- **Caco-2**: R² = 0.75 (best performance)
- **RRCK**: R² = 0.62
- **MDCK**: R² = 0.73

## 📁 Directory Structure

```
examples/
├── README.md                           # This file
├── data/                              # Sample datasets
│   ├── sample_cyclic_peptides.csv     # Main sample dataset (19 molecules)
│   ├── sequences/                      # Sample sequences by assay
│   │   ├── pampa_sample.csv           # PAMPA sample data
│   │   ├── rrck_sample.csv            # RRCK sample data
│   │   └── mdck_sample.csv            # MDCK sample data
│   ├── structures/                     # For 3D structure files
│   └── models/                         # For saved model outputs
├── use_case_1_predict_all_assays.py   # Universal multi-assay prediction
├── use_case_2_predict_single_assay.py # Single assay with binary classification
├── use_case_3_train_model.py          # Model training for specific assay
├── use_case_4_data_preprocessing.py   # Data preprocessing and featurization
└── use_case_5_batch_analysis.py       # Comprehensive analysis and visualization
```

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**: Make sure you have the CPMP conda environment activated:
```bash
# Activate the environment
mamba activate /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/cpmp_mcp/env

# Or if using conda
conda activate /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/cpmp_mcp/env
```

2. **Pre-trained Models**: The examples expect pre-trained models in `../repo/CPMP/saved_model/`:
   - `pampa.best_wegiht.pth`
   - `caco2.best_wegiht.pth`
   - `rrck.best_wegiht.pth`
   - `mdck.best_wegiht.pth`

### Basic Prediction Example

```bash
# Predict permeability for all assays (fastest start)
python use_case_1_predict_all_assays.py

# Predict for a specific assay with binary classification
python use_case_2_predict_single_assay.py --assay caco2 --with-labels
```

## 📋 Use Cases Overview

### 1. Universal Multi-Assay Prediction (`use_case_1_predict_all_assays.py`)

**Purpose**: Predict membrane permeability across all four assay types simultaneously.

**Features**:
- Processes SMILES strings to predict permeability values
- Outputs continuous permeability predictions (log P units)
- Includes performance statistics for each assay
- Fastest way to get comprehensive permeability predictions

**Example Usage**:
```bash
# Basic usage with default sample data
python use_case_1_predict_all_assays.py

# Custom input file and output
python use_case_1_predict_all_assays.py \
    --input data/sample_cyclic_peptides.csv \
    --output my_predictions.csv \
    --device cpu

# Using GPU (if available)
python use_case_1_predict_all_assays.py --device cuda:0
```

**Output**: CSV file with predictions for all four assays:
```
smiles,pampa,caco2,rrck,mdck
CC(C)C[C@@H]1NC(=O)...,-5.85,-5.69,-5.77,-6.25
```

### 2. Single Assay with Binary Classification (`use_case_2_predict_single_assay.py`)

**Purpose**: Focused prediction for a specific assay with binary permeability classification.

**Features**:
- Single assay prediction with detailed analysis
- Binary classification (permeable/non-permeable) using -6.0 threshold
- Performance evaluation when experimental labels are available
- Detailed assay information and statistics

**Example Usage**:
```bash
# Predict Caco-2 permeability
python use_case_2_predict_single_assay.py --assay caco2

# Include experimental data evaluation
python use_case_2_predict_single_assay.py \
    --assay pampa \
    --input data/sequences/pampa_sample.csv \
    --with-labels \
    --device cpu

# All available assays
for assay in pampa caco2 rrck mdck; do
    python use_case_2_predict_single_assay.py --assay $assay
done
```

**Output**: CSV with continuous and binary predictions:
```
smiles,caco2_permeability,caco2_permeable,caco2_probability
CC(C)C[C@@H]1NC(=O)...,-5.69,1,0.76
```

### 3. Model Training (`use_case_3_train_model.py`)

**Purpose**: Train CPMP models for specific assays using preprocessed data.

**Features**:
- Complete training pipeline for MAT-based CPMP models
- Configurable hyperparameters and training settings
- Model checkpointing and performance monitoring
- Training history logging and final evaluation

**Requirements**: Preprocessed data (see Use Case 4) or download from [Zenodo](https://zenodo.org/records/14638776)

**Example Usage**:
```bash
# Train PAMPA model with default settings
python use_case_3_train_model.py --assay pampa

# Custom training with specific parameters
python use_case_3_train_model.py \
    --assay caco2 \
    --data-dir /path/to/preprocessed/data \
    --epochs 400 \
    --learning-rate 0.001 \
    --output-dir my_training_output \
    --merge-train-val

# Training all assays
for assay in pampa caco2 rrck mdck; do
    python use_case_3_train_model.py --assay $assay --epochs 200
done
```

**Output**:
- Trained model: `{assay}.best_weight.pth`
- Training log: `{assay}_training.csv`
- Test predictions: `{assay}_test_predictions.csv`

### 4. Data Preprocessing (`use_case_4_data_preprocessing.py`)

**Purpose**: Preprocess raw cyclic peptide data for CPMP model training.

**Features**:
- Data filtering and cleaning
- Train/validation/test splitting
- Molecular featurization using RDKit and UFF force field
- 3D conformer generation and optimization
- Data format conversion for PyTorch training

**Example Usage**:
```bash
# Preprocess custom dataset
python use_case_4_data_preprocessing.py \
    --assay caco2 \
    --input my_raw_data.csv \
    --output-dir preprocessed_caco2

# Quick splitting without featurization (faster)
python use_case_4_data_preprocessing.py \
    --assay pampa \
    --skip-featurization \
    --test-size 0.15 \
    --val-size 0.15

# Full preprocessing pipeline
python use_case_4_data_preprocessing.py \
    --assay rrck \
    --min-value -9.0 \
    --force-field uff \
    --random-seed 42
```

**Note**: Molecular featurization can take several hours for large datasets.

### 5. Batch Analysis and Visualization (`use_case_5_batch_analysis.py`)

**Purpose**: Comprehensive analysis and visualization of permeability predictions.

**Features**:
- Statistical analysis across multiple assays
- Correlation analysis between assays
- Molecular property calculations (MW, LogP, TPSA, etc.)
- Automated visualization generation
- Detailed analysis reports

**Example Usage**:
```bash
# Basic batch analysis
python use_case_5_batch_analysis.py

# Comprehensive analysis with molecular properties
python use_case_5_batch_analysis.py \
    --input data/sample_cyclic_peptides.csv \
    --output-dir comprehensive_analysis \
    --calculate-properties \
    --max-molecules 500

# Large dataset analysis
python use_case_5_batch_analysis.py \
    --input large_dataset.csv \
    --device cuda:0 \
    --max-molecules 2000
```

**Output**:
- `complete_results.csv`: All predictions and properties
- `analysis_report.md`: Comprehensive analysis report
- Visualizations: distributions, correlations, property relationships

## 🔬 Understanding the Data

### Input Format

All scripts expect CSV files with a `smiles` column containing SMILES strings of cyclic peptides:

```csv
smiles
CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC1=O
C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C
```

### Output Interpretation

**Permeability Values**:
- Units: log₁₀(cm/s)
- Higher values = more permeable
- Typical ranges: -8.0 to -4.0

**Classification Thresholds**:
- **High permeability**: > -6.0 log cm/s
- **Moderate permeability**: -6.0 to -7.0 log cm/s
- **Low permeability**: < -7.0 log cm/s

**Assay-Specific Applications**:
- **PAMPA**: Drug screening, passive transport assessment
- **Caco-2**: Oral drug development, intestinal absorption prediction
- **RRCK**: CNS drug development, blood-brain barrier evaluation
- **MDCK**: General permeability screening, renal clearance

## ⚙️ Technical Details

### Molecular Featurization

The CPMP model uses:
- **Force Field**: UFF (Universal Force Field) for 3D optimization
- **Atom Features**: 28-dimensional one-hot encoded formal charges
- **Graph Representation**: Adjacency matrices and distance matrices
- **Conformer Generation**: RDKit ETKDG algorithm

### Model Architecture

- **Base Model**: Molecular Attention Transformer (MAT)
- **Hidden Dimension**: 64
- **Transformer Layers**: 6
- **Attention Heads**: 64
- **Dense Layers**: 2
- **Aggregation**: Dummy node pooling

### Performance Tips

1. **GPU Usage**: Use `--device cuda:0` for faster inference when available
2. **Batch Size**: Larger molecules may require smaller batch sizes
3. **Memory**: Featurization is memory-intensive; monitor RAM usage
4. **Time**: Molecular featurization is the slowest step (~1-5 min per molecule)

## 🧪 Sample Datasets

### Main Sample (`data/sample_cyclic_peptides.csv`)
- **Size**: 19 cyclic peptides
- **Source**: Caco-2 test set
- **Purpose**: Quick testing and demonstrations

### Assay-Specific Samples (`data/sequences/`)
- **pampa_sample.csv**: PAMPA permeability data (9 molecules)
- **rrck_sample.csv**: RRCK permeability data (9 molecules)
- **mdck_sample.csv**: MDCK permeability data (9 molecules)

Each sample includes experimental permeability values for validation.

## 🐛 Troubleshooting

### Common Issues

1. **Model Files Not Found**:
   ```
   Error: Model file pampa.best_wegiht.pth not found
   ```
   **Solution**: Ensure pre-trained models are in `../repo/CPMP/saved_model/`

2. **Memory Errors During Featurization**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU (`--device cpu`) or reduce batch size

3. **RDKit Import Errors**:
   ```
   ImportError: cannot import name 'Chem' from 'rdkit'
   ```
   **Solution**: Verify conda environment is properly activated

4. **Slow Featurization**:
   **Solution**: Use `--skip-featurization` for data splitting only, or reduce dataset size

### Getting Help

- Check that your conda environment is activated
- Ensure all dependencies are installed (see main README)
- Use `--help` flag with any script for detailed options
- Review the error messages and stack traces

## 📚 Further Reading

- **CPMP Paper**: [Original research publication]
- **MAT Framework**: https://github.com/ardigen/MAT
- **RDKit Documentation**: https://www.rdkit.org/docs/
- **Zenodo Dataset**: https://zenodo.org/records/14638776

## 🤝 Contributing

To add new use cases or improve existing ones:

1. Follow the existing naming convention: `use_case_X_description.py`
2. Include comprehensive docstrings and help text
3. Add error handling and progress indicators
4. Update this README with usage examples
5. Test with the provided sample datasets

---

*Generated by CPMP MCP Tool - Cyclic Peptide Membrane Permeability Prediction Toolkit*