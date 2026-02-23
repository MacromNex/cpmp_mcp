#!/usr/bin/env python3
"""
Script: predict_all_assays.py
Description: Predict cyclic peptide membrane permeability across all assays

Original Use Case: repo/CPMP/examples/use_case_1_predict_all_assays.py
Dependencies Removed: Simplified model loading, inlined configurations

Usage:
    python scripts/predict_all_assays.py --input <input_file> --output <output_file>

Example:
    python scripts/predict_all_assays.py --input examples/data/sample_cyclic_peptides.csv --output results/predictions.csv
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

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",
    "batch_size": 32,
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
    },
    "assays": {
        "pampa": {"batch_size": 32, "threshold": -6.0},
        "caco2": {"batch_size": 64, "threshold": -6.0},
        "rrck": {"batch_size": 64, "threshold": -6.0},
        "mdck": {"batch_size": 64, "threshold": -6.0}
    }
}

# Model checkpoint paths - relative to MCP root
SCRIPT_DIR = Path(__file__).parent
MCP_ROOT = SCRIPT_DIR.parent
REPO_PATH = MCP_ROOT / "repo" / "CPMP"

MODEL_PATHS = {
    "pampa": "model_checkpoints/pampa_uff_ig_true_final.pt",
    "caco2": "model_checkpoints/caco2_uff_ig_true_final.pt",
    "rrck": "model_checkpoints/rrck_uff_ig_true_final.pt",
    "mdck": "model_checkpoints/mdck_uff_ig_true_final.pt"
}

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
    """Parse input CSV file with SMILES data.

    Expected format: CSV with 'smiles' column, optional 'y' column for labels.
    """
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

def create_temp_csv(df: pd.DataFrame, temp_dir: Path) -> Path:
    """Create temporary CSV file for data_utils loading."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_input.csv"
    df.to_csv(temp_file, index=False)
    return temp_file

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_all_assays(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict cyclic peptide membrane permeability across all assays.

    Args:
        input_file: Path to input CSV file with 'smiles' column
        output_file: Path to save predictions CSV (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - predictions: DataFrame with predictions for all assays
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_all_assays("input.csv", "output.csv")
        >>> print(result['predictions'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    device = torch.device(config["device"])

    print(f"🧪 Predicting membrane permeability for cyclic peptides")
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

        # Predict for each assay
        predictions = {}
        all_predictions = df[['smiles']].copy()

        for assay_name in config["assays"].keys():
            print(f"   🎯 Predicting {assay_name.upper()}...")

            # Create data loader
            loader = construct_loader(
                X_data, y_data,
                batch_size=config["assays"][assay_name]["batch_size"]
            )

            # Load model
            model_path = REPO_PATH / MODEL_PATHS[assay_name]
            if not model_path.exists():
                print(f"   ⚠️ Model not found: {model_path}")
                continue

            model = make_model(**config["model"])
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Predict
            assay_predictions = []
            with torch.no_grad():
                for batch in loader:
                    adjacency, features, distances, labels = batch
                    batch_mask = torch.sum(torch.abs(features), dim=-1) != 0
                    adjacency = adjacency.to(device)
                    features = features.to(device)
                    distances = distances.to(device)
                    batch_mask = batch_mask.to(device)
                    outputs = model(features, batch_mask, adjacency, distances, None)
                    assay_predictions.extend(outputs.view(-1).cpu().numpy())

            # Store predictions
            predictions[assay_name] = np.array(assay_predictions).flatten()
            all_predictions[f'{assay_name}_permeability'] = predictions[assay_name]

            # Add binary classification
            threshold = config["assays"][assay_name]["threshold"]
            all_predictions[f'{assay_name}_permeable'] = (predictions[assay_name] > threshold).astype(int)

            mean_pred = np.mean(predictions[assay_name])
            std_pred = np.std(predictions[assay_name])
            print(f"     📊 Mean: {mean_pred:.2f} ± {std_pred:.2f} log cm/s")

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
        all_predictions.to_csv(output_path, index=False)
        print(f"   💾 Saved predictions: {output_path}")

    print(f"   ✅ Prediction complete")

    return {
        "predictions": all_predictions,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "n_molecules": len(all_predictions),
            "assays": list(config["assays"].keys()),
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
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with SMILES column')
    parser.add_argument('--output', '-o',
                       help='Output CSV file for predictions')
    parser.add_argument('--config', '-c',
                       help='Configuration file (JSON)')
    parser.add_argument('--device', default='cpu',
                       help='Device for inference (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')

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

    if args.batch_size != 32:
        if 'assays' not in config:
            config['assays'] = {}
        for assay in DEFAULT_CONFIG['assays']:
            if assay not in config['assays']:
                config['assays'][assay] = {}
            config['assays'][assay]['batch_size'] = args.batch_size

    # Run prediction
    try:
        result = run_predict_all_assays(
            input_file=args.input,
            output_file=args.output,
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