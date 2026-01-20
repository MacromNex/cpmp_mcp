#!/usr/bin/env python3
"""
Use Case 4: Preprocess Cyclic Peptide Data for CPMP Training

This script demonstrates how to preprocess raw cyclic peptide membrane permeability
data for training CPMP models. It handles data filtering, train/val/test splitting,
molecular featurization, and data format conversion.

Based on: repo/CPMP/data/*/process_data.py
Author: CPMP MCP Tool
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add CPMP to Python path
SCRIPT_DIR = Path(__file__).parent
CPMP_DIR = SCRIPT_DIR.parent / "repo" / "CPMP"
sys.path.insert(0, str(CPMP_DIR))

# Import CPMP modules
from featurization.data_utils import load_data_from_df


ASSAY_INFO = {
    'pampa': {
        'name': 'PAMPA (Parallel Artificial Membrane Permeability Assay)',
        'column_name': 'PAMPA',
        'dataset_file': 'CycPeptMPDB_Peptide_Assay_PAMPA.csv',
        'min_value': -10.0,
        'description': 'Parallel Artificial Membrane Permeability Assay data'
    },
    'caco2': {
        'name': 'Caco-2 (Human colon adenocarcinoma cell line)',
        'column_name': 'Caco2',
        'dataset_file': 'CycPeptMPDB_Peptide_Assay_Caco2.csv',
        'min_value': -10.0,
        'description': 'Caco-2 permeability assay data'
    },
    'rrck': {
        'name': 'RRCK (Ralph Russ Canine Kidney cells)',
        'column_name': 'RRCK',
        'dataset_file': 'CycPeptMPDB_Peptide_Assay_RRCK.csv',
        'min_value': -10.0,
        'description': 'RRCK permeability assay data'
    },
    'mdck': {
        'name': 'MDCK (Madin-Darby Canine Kidney cells)',
        'column_name': 'MDCK',
        'dataset_file': 'CycPeptMPDB_Peptide_Assay_MDCK.csv',
        'min_value': -10.0,
        'description': 'MDCK permeability assay data'
    }
}


def filter_and_clean_data(df, assay, min_value=-10.0):
    """
    Filter and clean the dataset.

    Args:
        df: Raw dataframe with SMILES and permeability data
        assay: Assay type
        min_value: Minimum permeability value to include

    Returns:
        Cleaned dataframe
    """
    print(f"   📊 Original dataset size: {len(df)} molecules")

    # Filter by minimum permeability value
    df_filtered = df[df['y'] > min_value]
    print(f"   📊 After filtering (y > {min_value}): {len(df_filtered)} molecules")

    # Remove duplicates
    initial_size = len(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=['smiles'])
    if len(df_filtered) < initial_size:
        print(f"   📊 Removed {initial_size - len(df_filtered)} duplicate SMILES")

    # Remove missing values
    df_filtered = df_filtered.dropna()
    print(f"   📊 Final dataset size: {len(df_filtered)} molecules")

    # Show statistics
    print(f"   📈 Permeability range: [{df_filtered['y'].min():.2f}, {df_filtered['y'].max():.2f}]")
    print(f"   📈 Mean ± std: {df_filtered['y'].mean():.2f} ± {df_filtered['y'].std():.2f}")

    return df_filtered


def split_dataset(df, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Split dataset into train/validation/test sets.

    Args:
        df: Dataframe to split
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=None
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)  # Adjust validation size
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_fraction, random_state=random_seed + 1
    )

    print(f"   📚 Dataset split:")
    print(f"      Training: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"      Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"      Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def featurize_dataset(csv_file, output_name, force_field='uff'):
    """
    Featurize molecules from CSV file using CPMP featurization.

    Args:
        csv_file: Path to CSV file with 'smiles' and 'y' columns
        output_name: Name prefix for output
        force_field: Force field for 3D optimization

    Returns:
        Tuple of (X, y) - featurized features and labels
    """
    print(f"   🧪 Featurizing {output_name}...")
    print(f"      Force field: {force_field}")
    print(f"      Input file: {csv_file}")

    # Change to CPMP directory for relative imports to work
    original_dir = os.getcwd()
    os.chdir(str(CPMP_DIR))

    try:
        X, y = load_data_from_df(
            str(csv_file),
            ff=force_field,
            ignoreInterfragInteractions=True,
            one_hot_formal_charge=True,
            use_data_saving=True
        )
        print(f"      ✅ Featurization complete: {len(X)} molecules")
        return X, y

    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess cyclic peptide data for CPMP model training"
    )
    parser.add_argument(
        "--assay", "-a",
        choices=list(ASSAY_INFO.keys()),
        required=True,
        help="Assay type to preprocess"
    )
    parser.add_argument(
        "--input", "-i",
        help="Input CSV file (if not using default dataset)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="preprocessed_data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--min-value", "-m",
        type=float,
        help="Minimum permeability value to include (default: assay-specific)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of data for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--force-field", "-ff",
        default="uff",
        choices=["uff", "mmff"],
        help="Force field for molecular optimization (default: uff)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)"
    )
    parser.add_argument(
        "--skip-featurization",
        action="store_true",
        help="Skip molecular featurization (only do data splitting)"
    )

    args = parser.parse_args()

    assay_info = ASSAY_INFO[args.assay]
    min_value = args.min_value or assay_info['min_value']

    print(f"🧬 CPMP Data Preprocessing - {assay_info['name']}")
    print("=" * 70)
    print(f"🎯 Assay: {args.assay.upper()}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🚪 Minimum permeability: {min_value}")
    print(f"📊 Data split: {args.test_size:.1%} test, {args.val_size:.1%} val")
    print("=" * 70)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Load input data
        if args.input:
            input_file = Path(args.input)
            if not input_file.exists():
                print(f"❌ Error: Input file {input_file} not found!")
                return 1

            print(f"\n📚 Loading custom dataset: {input_file}")
            df = pd.read_csv(input_file, low_memory=False)

            # Ensure proper column names
            if 'SMILES' in df.columns:
                df = df.rename(columns={'SMILES': 'smiles'})
            if assay_info['column_name'] in df.columns:
                df = df.rename(columns={assay_info['column_name']: 'y'})

            # Select required columns
            df = df[['smiles', 'y']]

        else:
            # Use default dataset
            default_dataset = CPMP_DIR / "data" / f"{args.assay}_uff_ig_true" / assay_info['dataset_file']
            if not default_dataset.exists():
                print(f"❌ Error: Default dataset {default_dataset} not found!")
                print(f"   Please provide --input with your dataset file")
                return 1

            print(f"\n📚 Loading default dataset: {default_dataset}")
            df = pd.read_csv(default_dataset, low_memory=False)
            df = df[['SMILES', assay_info['column_name']]].copy()
            df.columns = ['smiles', 'y']

        # Clean and filter data
        print(f"\n🧹 Cleaning and filtering data...")
        df_clean = filter_and_clean_data(df, args.assay, min_value)

        # Save cleaned dataset
        clean_file = output_dir / f"{args.assay}_clean.csv"
        df_clean.to_csv(clean_file, index=False)
        print(f"   💾 Cleaned dataset saved: {clean_file}")

        # Split dataset
        print(f"\n📊 Splitting dataset...")
        train_df, val_df, test_df = split_dataset(
            df_clean, args.test_size, args.val_size, args.random_seed
        )

        # Save split datasets
        train_file = output_dir / f"{args.assay}_train.csv"
        val_file = output_dir / f"{args.assay}_val.csv"
        test_file = output_dir / f"{args.assay}_test.csv"

        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"   💾 Split datasets saved:")
        print(f"      Training: {train_file}")
        print(f"      Validation: {val_file}")
        print(f"      Test: {test_file}")

        # Molecular featurization
        if not args.skip_featurization:
            print(f"\n🧪 Starting molecular featurization...")
            print(f"   ⏰ This may take several hours for large datasets")

            # Featurize each split
            for split_name, split_file in [("train", train_file), ("val", val_file), ("test", test_file)]:
                print(f"\n   🔄 Processing {split_name} set...")
                X, y = featurize_dataset(split_file, split_name, args.force_field)

                # Convert to DataFrame/Series and save as pickle
                X_df = pd.DataFrame(X)
                y_series = pd.Series(y)

                X_pkl_file = output_dir / f"X_{split_name}.pkl"
                y_pkl_file = output_dir / f"y_{split_name}.pkl"

                X_df.to_pickle(X_pkl_file)
                y_series.to_pickle(y_pkl_file)

                print(f"      💾 Saved: {X_pkl_file}")
                print(f"      💾 Saved: {y_pkl_file}")

            print(f"\n✅ Molecular featurization complete!")

        else:
            print(f"\n⏭️  Skipping molecular featurization")
            print(f"   To featurize later, run without --skip-featurization")

        # Summary
        print(f"\n📋 Data preprocessing summary:")
        print(f"   🎯 Assay: {assay_info['name']}")
        print(f"   📊 Total molecules: {len(df_clean)}")
        print(f"   📚 Training: {len(train_df)} molecules")
        print(f"   🔍 Validation: {len(val_df)} molecules")
        print(f"   🧪 Test: {len(test_df)} molecules")
        print(f"   📁 Output directory: {output_dir}")

        if not args.skip_featurization:
            print(f"\n🚀 Ready for model training!")
            print(f"   Use: python use_case_3_train_model.py --assay {args.assay} --data-dir {output_dir}")

        return 0

    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())