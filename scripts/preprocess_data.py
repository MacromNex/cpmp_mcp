#!/usr/bin/env python3
"""
Script: preprocess_data.py
Description: Preprocess and split cyclic peptide datasets for model training

Original Use Case: examples/use_case_4_data_preprocessing.py
Dependencies Removed: Simplified to core preprocessing without heavy featurization

Usage:
    python scripts/preprocess_data.py --assay <assay_name> --output-dir <output_dir>

Example:
    python scripts/preprocess_data.py --assay caco2 --output-dir results/preprocessing
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import sys
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import json

# Essential scientific packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "data_filter": {
        "min_y_value": -10.0,
        "remove_duplicates": True,
        "validate_smiles": True
    },
    "train_split": {
        "train_size": 0.7,
        "val_size": 0.1,
        "test_size": 0.2,
        "random_state": 42,
        "stratify": False  # Set to True for classification
    },
    "featurization": {
        "force_field": "uff",
        "ignore_interfrag_interactions": True,
        "skip_featurization": True  # Default to skip for speed
    }
}

# Dataset paths - relative to MCP root
SCRIPT_DIR = Path(__file__).parent
MCP_ROOT = SCRIPT_DIR.parent
REPO_PATH = MCP_ROOT / "repo" / "CPMP"

DATASET_PATHS = {
    "caco2": "data/CycPeptMPDB_Peptide_Assay_Caco2.csv",
    "pampa": "data/CycPeptMPDB_Peptide_Assay_PAMPA.csv",
    "rrck": "data/CycPeptMPDB_Peptide_Assay_RRCK.csv",
    "mdck": "data/CycPeptMPDB_Peptide_Assay_MDCK.csv"
}

# ==============================================================================
# Lazy Repo Loading (minimize startup time)
# ==============================================================================
def get_repo_modules():
    """Lazy load repo modules for featurization (if needed)."""
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))

    from featurization.data_utils import load_data_from_df
    return load_data_from_df

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def load_dataset(dataset_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset CSV file."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Standardize column names
    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles'})
    if 'Y' in df.columns:
        df = df.rename(columns={'Y': 'y'})

    # Ensure required columns exist
    if 'smiles' not in df.columns:
        raise ValueError("Dataset must have 'smiles' or 'SMILES' column")
    if 'y' not in df.columns:
        raise ValueError("Dataset must have 'y' or 'Y' column")

    return df

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def filter_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter dataset based on quality criteria."""
    filter_config = config["data_filter"]
    stats = {"original": len(df)}

    print(f"   📊 Original dataset: {len(df)} molecules")

    # Filter by y value
    if "min_y_value" in filter_config:
        min_y = filter_config["min_y_value"]
        df = df[df['y'] > min_y]
        stats["after_y_filter"] = len(df)
        print(f"   📊 After y > {min_y} filter: {len(df)} molecules")

    # Validate SMILES
    if filter_config.get("validate_smiles", True):
        valid_smiles = df['smiles'].apply(validate_smiles)
        invalid_count = (~valid_smiles).sum()
        if invalid_count > 0:
            print(f"   ⚠️ Found {invalid_count} invalid SMILES")
            df = df[valid_smiles]
            stats["after_smiles_validation"] = len(df)
            print(f"   📊 After SMILES validation: {len(df)} molecules")

    # Remove duplicates
    if filter_config.get("remove_duplicates", True):
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['smiles']).reset_index(drop=True)
        duplicates_removed = before_dedup - len(df)
        if duplicates_removed > 0:
            print(f"   🔄 Removed {duplicates_removed} duplicate SMILES")
        stats["after_deduplication"] = len(df)
        print(f"   📊 After deduplication: {len(df)} molecules")

    return df, stats

def split_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/validation/test sets."""
    split_config = config["train_split"]

    train_size = split_config["train_size"]
    val_size = split_config["val_size"]
    test_size = split_config["test_size"]
    random_state = split_config["random_state"]

    # Validate split sizes
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )

    return train_df, val_df, test_df

def calculate_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate basic statistics for a dataset."""
    y_values = df['y'].values

    return {
        "count": len(df),
        "mean": float(np.mean(y_values)),
        "std": float(np.std(y_values)),
        "min": float(np.min(y_values)),
        "max": float(np.max(y_values)),
        "median": float(np.median(y_values)),
        "q25": float(np.percentile(y_values, 25)),
        "q75": float(np.percentile(y_values, 75))
    }

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_preprocess_data(
    assay: Optional[str] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "./results/preprocessing",
    skip_featurization: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Preprocess and split cyclic peptide dataset for model training.

    Args:
        assay: Assay name (caco2, pampa, rrck, mdck) - uses default dataset
        dataset_path: Path to custom dataset CSV file
        output_dir: Directory to save processed files
        skip_featurization: Skip featurization for faster processing
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - splits: Dict with train/val/test DataFrames
            - statistics: Statistics for each split
            - output_files: Paths to saved files
            - metadata: Processing metadata

    Example:
        >>> result = run_preprocess_data(assay="caco2", output_dir="results")
        >>> print(result['statistics']['train']['count'])
    """
    # Validate input
    if assay is None and dataset_path is None:
        raise ValueError("Either 'assay' or 'dataset_path' must be provided")

    if assay and assay.lower() not in DATASET_PATHS:
        raise ValueError(f"Unknown assay '{assay}'. Available: {list(DATASET_PATHS.keys())}")

    # Setup
    if assay:
        assay = assay.lower()
        dataset_path = REPO_PATH / DATASET_PATHS[assay]
        dataset_name = assay
    else:
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.stem

    output_dir = Path(output_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    config["featurization"]["skip_featurization"] = skip_featurization

    print(f"🔬 Preprocessing cyclic peptide dataset")
    print(f"   📋 Dataset: {dataset_name}")
    print(f"   📁 Input: {dataset_path}")
    print(f"   📁 Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"   📖 Loading dataset...")
    df = load_dataset(dataset_path)

    # Filter dataset
    print(f"   🔍 Filtering dataset...")
    df_clean, filter_stats = filter_dataset(df, config)

    # Split dataset
    print(f"   ✂️ Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df_clean, config)

    print(f"   📊 Dataset splits:")
    print(f"      Train: {len(train_df)} molecules ({len(train_df)/len(df_clean):.1%})")
    print(f"      Val:   {len(val_df)} molecules ({len(val_df)/len(df_clean):.1%})")
    print(f"      Test:  {len(test_df)} molecules ({len(test_df)/len(df_clean):.1%})")

    # Calculate statistics
    statistics = {
        "train": calculate_statistics(train_df),
        "val": calculate_statistics(val_df),
        "test": calculate_statistics(test_df),
        "combined": calculate_statistics(df_clean)
    }

    print(f"   📊 Permeability statistics (log cm/s):")
    print(f"      Combined: {statistics['combined']['mean']:.2f} ± {statistics['combined']['std']:.2f}")
    print(f"      Range: [{statistics['combined']['min']:.2f}, {statistics['combined']['max']:.2f}]")

    # Save processed files
    print(f"   💾 Saving processed files...")
    output_files = {}

    # Save clean dataset
    clean_file = output_dir / f"{dataset_name}_clean.csv"
    df_clean.to_csv(clean_file, index=False)
    output_files["clean"] = str(clean_file)

    # Save splits
    split_files = {
        "train": output_dir / f"{dataset_name}_train.csv",
        "val": output_dir / f"{dataset_name}_val.csv",
        "test": output_dir / f"{dataset_name}_test.csv"
    }

    train_df.to_csv(split_files["train"], index=False)
    val_df.to_csv(split_files["val"], index=False)
    test_df.to_csv(split_files["test"], index=False)

    for split_name, file_path in split_files.items():
        output_files[split_name] = str(file_path)

    # Save metadata
    metadata_file = output_dir / f"{dataset_name}_metadata.json"
    metadata = {
        "dataset_name": dataset_name,
        "original_path": str(dataset_path),
        "processing_date": pd.Timestamp.now().isoformat(),
        "filter_stats": filter_stats,
        "statistics": statistics,
        "config": config,
        "output_files": output_files
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = str(metadata_file)

    print(f"   💾 Files saved:")
    for file_type, file_path in output_files.items():
        file_size = Path(file_path).stat().st_size
        print(f"      {file_type}: {Path(file_path).name} ({file_size:,} bytes)")

    # Featurization (if requested and not skipped)
    if not config["featurization"]["skip_featurization"]:
        print(f"   🔬 Featurizing data (this may take a while)...")
        try:
            load_data_from_df = get_repo_modules()

            # Featurize each split
            for split_name, split_file in split_files.items():
                print(f"      Featurizing {split_name}...")
                X_data, y_data = load_data_from_df(
                    str(split_file),
                    ff=config["featurization"]["force_field"],
                    ignoreInterfragInteractions=config["featurization"]["ignore_interfrag_interactions"]
                )

                # Save featurized data
                feature_file = output_dir / f"{dataset_name}_{split_name}_features.pkl"
                import pickle
                with open(feature_file, 'wb') as f:
                    pickle.dump((X_data, y_data), f)

                output_files[f"{split_name}_features"] = str(feature_file)
                print(f"         Saved: {feature_file.name}")

        except Exception as e:
            print(f"   ⚠️ Featurization failed: {e}")
            print(f"      Continuing without featurization...")

    print(f"   ✅ Preprocessing complete")

    return {
        "splits": {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "clean": df_clean
        },
        "statistics": statistics,
        "output_files": output_files,
        "metadata": metadata
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--assay', '-a',
                           choices=list(DATASET_PATHS.keys()),
                           help='Assay dataset to preprocess')
    input_group.add_argument('--dataset', '-d',
                           help='Path to custom dataset CSV file')

    parser.add_argument('--output-dir', '-o', default='./results/preprocessing',
                       help='Output directory for processed files')
    parser.add_argument('--config', '-c',
                       help='Configuration file (JSON)')
    parser.add_argument('--skip-featurization', action='store_true', default=True,
                       help='Skip featurization for faster processing')
    parser.add_argument('--with-featurization', dest='skip_featurization',
                       action='store_false',
                       help='Include featurization (slow but complete)')

    # Data filtering options
    parser.add_argument('--min-y', type=float,
                       help='Minimum y value for filtering')
    parser.add_argument('--train-size', type=float,
                       help='Training set proportion (e.g., 0.7)')
    parser.add_argument('--val-size', type=float,
                       help='Validation set proportion (e.g., 0.1)')
    parser.add_argument('--test-size', type=float,
                       help='Test set proportion (e.g., 0.2)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI args
    if config is None:
        config = {}

    # Override data filter settings
    if args.min_y is not None:
        if "data_filter" not in config:
            config["data_filter"] = {}
        config["data_filter"]["min_y_value"] = args.min_y

    # Override split settings
    if any([args.train_size, args.val_size, args.test_size]):
        if "train_split" not in config:
            config["train_split"] = {}

        if args.train_size:
            config["train_split"]["train_size"] = args.train_size
        if args.val_size:
            config["train_split"]["val_size"] = args.val_size
        if args.test_size:
            config["train_split"]["test_size"] = args.test_size

    # Run preprocessing
    try:
        result = run_preprocess_data(
            assay=args.assay,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            skip_featurization=args.skip_featurization,
            config=config
        )

        print(f"\n🎉 Success! Processed files saved to: {args.output_dir}")
        print(f"   📁 Files: {len(result['output_files'])} files created")
        print(f"   🧬 Molecules: {result['statistics']['combined']['count']} total")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()