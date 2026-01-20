#!/usr/bin/env python3
"""
Script: predict_single_assay.py
Description: Predict cyclic peptide membrane permeability for a specific assay with detailed analysis

Original Use Case: repo/CPMP/examples/use_case_2_predict_single_assay.py
Dependencies Removed: Simplified model loading, inlined analysis functions

Usage:
    python scripts/predict_single_assay.py --assay <assay_name> --input <input_file> --output <output_file>

Example:
    python scripts/predict_single_assay.py --assay caco2 --input examples/data/sample_cyclic_peptides.csv --output results/caco2_predictions.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import sys
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# Essential scientific packages
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",
    "featurization": {
        "force_field": "uff",
        "ignore_interfrag_interactions": True,
        "add_dummy_node": True,
        "one_hot_formal_charge": False
    },
    "model": {
        "d_atom": 28,
        "d_model": 64,
        "N": 6,
        "h": 64,
        "N_dense": 2,
        "lambda_attention": 0.1,
        "lambda_distance": 0.6,
        "leaky_relu_slope": 0.16,
        "dense_output_nonlinearity": "relu",
        "distance_matrix_kernel": "exp",
        "aggregation_type": "mean",
        "dropout": 0.1
    }
}

# Assay information and model paths
ASSAY_INFO = {
    'pampa': {
        'name': 'PAMPA',
        'full_name': 'Parallel Artificial Membrane Permeability Assay',
        'description': 'Measures passive diffusion through artificial lipid membrane',
        'batch_size': 32,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/pampa_uff_ig_true_final.pt',
        'applications': 'Drug screening, passive permeability assessment'
    },
    'caco2': {
        'name': 'Caco-2',
        'full_name': 'Human colon adenocarcinoma cell line',
        'description': 'Models intestinal absorption and oral bioavailability',
        'batch_size': 64,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/caco2_uff_ig_true_final.pt',
        'applications': 'Oral drug development, intestinal permeability'
    },
    'rrck': {
        'name': 'RRCK',
        'full_name': 'Ralph Russ Canine Kidney cells',
        'description': 'Evaluates blood-brain barrier permeability',
        'batch_size': 64,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/rrck_uff_ig_true_final.pt',
        'applications': 'CNS drug development, BBB penetration'
    },
    'mdck': {
        'name': 'MDCK',
        'full_name': 'Madin-Darby Canine Kidney cells',
        'description': 'Models general membrane permeability',
        'batch_size': 64,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/mdck_uff_ig_true_final.pt',
        'applications': 'General permeability screening, renal clearance'
    }
}

# Model checkpoint paths - relative to MCP root
SCRIPT_DIR = Path(__file__).parent
MCP_ROOT = SCRIPT_DIR.parent
REPO_PATH = MCP_ROOT / "repo" / "CPMP"

# ==============================================================================
# Lazy Repo Loading (minimize startup time)
# ==============================================================================
def get_repo_modules():
    """Lazy load repo modules to minimize startup time."""
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))

    from featurization.data_utils import load_data_from_df, construct_loader
    from model.transformer import make_model
    return load_data_from_df, construct_loader, make_model

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def parse_input_file(input_file: Union[str, Path]) -> pd.DataFrame:
    """Parse input CSV file with SMILES data."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must have a 'smiles' column")

    # Add dummy y column if not present (required by data_utils)
    if 'y' not in df.columns:
        df['y'] = -10.0

    return df

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def sigmoid(x):
    """Sigmoid function for probability calculation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float):
    """Calculate binary classification metrics."""
    y_binary_true = (y_true > threshold).astype(int)
    y_binary_pred = (y_pred > threshold).astype(int)

    # Calculate probabilities using sigmoid of distance from threshold
    y_prob = sigmoid((y_pred - threshold) * 5)  # Scale factor for reasonable probabilities

    metrics = {
        'accuracy': accuracy_score(y_binary_true, y_binary_pred),
        'auc_roc': roc_auc_score(y_binary_true, y_prob) if len(np.unique(y_binary_true)) > 1 else np.nan,
        'confusion_matrix': confusion_matrix(y_binary_true, y_binary_pred).tolist(),
        'permeable_count': int(y_binary_pred.sum()),
        'permeable_fraction': float(y_binary_pred.mean())
    }

    return metrics, y_binary_pred, y_prob

def create_temp_csv(df: pd.DataFrame, temp_dir: Path) -> Path:
    """Create temporary CSV file for data_utils loading."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_input.csv"
    df.to_csv(temp_file, index=False)
    return temp_file

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_single_assay(
    assay: str,
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    with_labels: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict cyclic peptide membrane permeability for a specific assay.

    Args:
        assay: Assay name (pampa, caco2, rrck, mdck)
        input_file: Path to input CSV file with 'smiles' column
        output_file: Path to save predictions CSV (optional)
        with_labels: Include binary classification and probabilities
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - predictions: DataFrame with predictions
            - metrics: Binary classification metrics (if with_labels=True)
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_single_assay("caco2", "input.csv", "output.csv")
        >>> print(result['metrics']['accuracy'])
    """
    # Validate assay
    if assay.lower() not in ASSAY_INFO:
        raise ValueError(f"Unknown assay '{assay}'. Available: {list(ASSAY_INFO.keys())}")

    assay = assay.lower()
    assay_info = ASSAY_INFO[assay]

    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    device = torch.device(config["device"])

    print(f"🧪 Predicting {assay_info['name']} membrane permeability")
    print(f"   📋 Assay: {assay_info['full_name']}")
    print(f"   📄 Description: {assay_info['description']}")
    print(f"   📁 Input: {input_file}")
    print(f"   💻 Device: {device}")

    # Load and validate input data
    df = parse_input_file(input_file)
    n_molecules = len(df)
    print(f"   🧬 Loaded {n_molecules} molecules")

    # Validate SMILES
    valid_smiles = df['smiles'].apply(validate_smiles)
    if not valid_smiles.all():
        invalid_count = (~valid_smiles).sum()
        print(f"   ⚠️ Warning: {invalid_count} invalid SMILES found")
        df = df[valid_smiles].reset_index(drop=True)
        print(f"   ✅ Processing {len(df)} valid molecules")

    # Lazy load repo modules
    load_data_from_df, construct_loader, make_model = get_repo_modules()

    # Create temporary file for data loading
    temp_dir = MCP_ROOT / "temp"
    temp_csv = create_temp_csv(df, temp_dir)

    try:
        # Featurize data
        print(f"   🔬 Featurizing molecules...")
        X_data, y_data = load_data_from_df(
            str(temp_csv),
            ff=config["featurization"]["force_field"],
            ignoreInterfragInteractions=config["featurization"]["ignore_interfrag_interactions"],
            add_dummy_node=config["featurization"]["add_dummy_node"],
            one_hot_formal_charge=config["featurization"]["one_hot_formal_charge"]
        )

        # Create data loader
        loader = construct_loader(
            X_data, y_data,
            batch_size=assay_info["batch_size"]
        )

        # Load model
        model_path = REPO_PATH / assay_info["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"   🤖 Loading model: {model_path}")
        model = make_model(**config["model"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Predict
        print(f"   🎯 Predicting permeability...")
        predictions = []
        with torch.no_grad():
            for batch in loader:
                X_batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch[0]]
                outputs = model(X_batch)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions).flatten()

        # Create results DataFrame
        results = df[['smiles']].copy()
        results[f'{assay}_permeability'] = predictions

        # Add binary classification if requested
        metrics = None
        if with_labels:
            threshold = assay_info["threshold"]

            # Calculate binary classification
            binary_pred = (predictions > threshold).astype(int)
            prob_pred = sigmoid((predictions - threshold) * 5)

            results[f'{assay}_permeable'] = binary_pred
            results[f'{assay}_probability'] = prob_pred

            # Calculate metrics if we have true labels
            if 'y' in df.columns and not (df['y'] == -10.0).all():
                metrics, _, _ = calculate_binary_metrics(df['y'].values, predictions, threshold)
                print(f"   📊 Accuracy: {metrics['accuracy']:.3f}")
                if not np.isnan(metrics['auc_roc']):
                    print(f"   📊 AUC-ROC: {metrics['auc_roc']:.3f}")
            else:
                # Summary metrics for predictions
                permeable_count = binary_pred.sum()
                permeable_fraction = binary_pred.mean()
                print(f"   📊 Permeable molecules: {permeable_count}/{len(df)} ({permeable_fraction:.1%})")

                metrics = {
                    'permeable_count': int(permeable_count),
                    'permeable_fraction': float(permeable_fraction),
                    'threshold': threshold
                }

        # Print summary statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        print(f"   📊 Permeability: {mean_pred:.2f} ± {std_pred:.2f} log cm/s")
        print(f"   📊 Range: [{min_pred:.2f}, {max_pred:.2f}]")

    finally:
        # Cleanup temp file
        if temp_csv.exists():
            temp_csv.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"   💾 Saved predictions: {output_path}")

    print(f"   ✅ Prediction complete")

    return {
        "predictions": results,
        "metrics": metrics,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "n_molecules": len(results),
            "assay": assay,
            "assay_info": assay_info,
            "config": config
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--assay', '-a', required=True,
                       choices=list(ASSAY_INFO.keys()),
                       help='Assay to predict (pampa, caco2, rrck, mdck)')
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with SMILES column')
    parser.add_argument('--output', '-o',
                       help='Output CSV file for predictions')
    parser.add_argument('--config', '-c',
                       help='Configuration file (JSON)')
    parser.add_argument('--device', default='cpu',
                       help='Device for inference (cpu/cuda)')
    parser.add_argument('--with-labels', action='store_true', default=True,
                       help='Include binary classification and probabilities')
    parser.add_argument('--no-labels', dest='with_labels', action='store_false',
                       help='Skip binary classification (continuous values only)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI args
    if config is None:
        config = {}

    if args.device != 'cpu':
        config['device'] = args.device

    # Run prediction
    try:
        result = run_predict_single_assay(
            assay=args.assay,
            input_file=args.input,
            output_file=args.output,
            with_labels=args.with_labels,
            config=config
        )

        if result['output_file']:
            print(f"\n🎉 Success! Predictions saved to: {result['output_file']}")
        else:
            print(f"\n🎉 Success! Predictions completed for {result['metadata']['n_molecules']} molecules")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()