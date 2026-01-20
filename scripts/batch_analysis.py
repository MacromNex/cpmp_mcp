#!/usr/bin/env python3
"""
Script: batch_analysis.py
Description: Comprehensive batch analysis and visualization of cyclic peptide permeability predictions

Original Use Case: examples/use_case_5_batch_analysis.py
Dependencies Removed: Simplified visualization functions, inlined molecular property calculations

Usage:
    python scripts/batch_analysis.py --input <input_file> --output-dir <output_dir>

Example:
    python scripts/batch_analysis.py --input examples/data/sample_cyclic_peptides.csv --output-dir results/analysis
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
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats

# Optional imports for enhanced visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",
    "calculate_properties": True,
    "create_visualizations": True,
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
    "visualization": {
        "dpi": 300,
        "figsize": [12, 8],
        "style": "whitegrid" if HAS_SEABORN else "default"
    }
}

# Assay information
ASSAY_INFO = {
    'pampa': {
        'name': 'PAMPA',
        'full_name': 'Parallel Artificial Membrane Permeability Assay',
        'description': 'Measures passive diffusion through artificial lipid membrane',
        'batch_size': 32,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/pampa_uff_ig_true_final.pt',
        'color': '#1f77b4'
    },
    'caco2': {
        'name': 'Caco-2',
        'full_name': 'Human colon adenocarcinoma cell line',
        'description': 'Models intestinal absorption and oral bioavailability',
        'batch_size': 64,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/caco2_uff_ig_true_final.pt',
        'color': '#ff7f0e'
    },
    'rrck': {
        'name': 'RRCK',
        'full_name': 'Ralph Russ Canine Kidney cells',
        'description': 'Evaluates blood-brain barrier permeability',
        'batch_size': 64,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/rrck_uff_ig_true_final.pt',
        'color': '#2ca02c'
    },
    'mdck': {
        'name': 'MDCK',
        'full_name': 'Madin-Darby Canine Kidney cells',
        'description': 'Models general membrane permeability',
        'batch_size': 64,
        'threshold': -6.0,
        'model_path': 'model_checkpoints/mdck_uff_ig_true_final.pt',
        'color': '#d62728'
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
# Inlined Utility Functions
# ==============================================================================
def parse_input_file(input_file: Union[str, Path]) -> pd.DataFrame:
    """Parse input CSV file with SMILES data."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must have a 'smiles' column")

    # Add dummy y column if not present
    if 'y' not in df.columns:
        df['y'] = -10.0

    return df

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_molecular_properties(smiles_list: List[str]) -> pd.DataFrame:
    """Calculate basic molecular properties using RDKit."""
    properties = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            props = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'Rings': Descriptors.RingCount(mol)
            }
        else:
            props = {prop: np.nan for prop in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Rings']}

        properties.append(props)

    return pd.DataFrame(properties)

def sigmoid(x):
    """Sigmoid function for probability calculation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def create_temp_csv(df: pd.DataFrame, temp_dir: Path) -> Path:
    """Create temporary CSV file for data_utils loading."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_input.csv"
    df.to_csv(temp_file, index=False)
    return temp_file

def predict_all_assays(df: pd.DataFrame, config: Dict[str, Any], device: torch.device) -> Dict[str, np.ndarray]:
    """Predict permeability for all assays."""
    # Lazy load repo modules
    load_data_from_df, construct_loader, make_model = get_repo_modules()

    # Create temporary file for data loading
    temp_dir = MCP_ROOT / "temp"
    temp_csv = create_temp_csv(df, temp_dir)

    predictions = {}

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
        for assay_name, assay_info in ASSAY_INFO.items():
            print(f"   🎯 Predicting {assay_name.upper()}...")

            # Create data loader
            loader = construct_loader(
                X_data, y_data,
                batch_size=assay_info["batch_size"]
            )

            # Load model
            model_path = REPO_PATH / assay_info["model_path"]
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
                    X_batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch[0]]
                    outputs = model(X_batch)
                    assay_predictions.extend(outputs.cpu().numpy())

            predictions[assay_name] = np.array(assay_predictions).flatten()

    finally:
        # Cleanup temp file
        if temp_csv.exists():
            temp_csv.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    return predictions

def perform_statistical_analysis(df: pd.DataFrame, output_dir: Path):
    """Perform statistical analysis of predictions."""
    assays = [col for col in df.columns if any(col.startswith(assay) for assay in ASSAY_INFO) and col.endswith('_permeability')]

    if not assays:
        return {}

    # Calculate summary statistics
    summary_stats = df[assays].describe()
    summary_file = output_dir / "statistical_summary.csv"
    summary_stats.to_csv(summary_file)

    # Calculate correlations
    correlation_matrix = df[assays].corr()
    corr_file = output_dir / "correlation_matrix.csv"
    correlation_matrix.to_csv(corr_file)

    # Calculate binary classification stats
    binary_stats = {}
    for assay in ASSAY_INFO.keys():
        perm_col = f"{assay}_permeable"
        if perm_col in df.columns:
            permeable_count = df[perm_col].sum()
            total_count = len(df)
            binary_stats[assay] = {
                "permeable_count": int(permeable_count),
                "permeable_fraction": float(permeable_count / total_count),
                "total_count": total_count
            }

    return {
        "summary_stats": summary_stats,
        "correlation_matrix": correlation_matrix,
        "binary_stats": binary_stats
    }

def create_visualizations(df: pd.DataFrame, output_dir: Path, config: Dict[str, Any]):
    """Create analysis visualizations."""
    if not config.get("create_visualizations", True):
        return []

    viz_config = config["visualization"]
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_style(viz_config["style"])

    created_plots = []

    # Get assay columns
    assay_cols = [col for col in df.columns if any(col.startswith(assay) for assay in ASSAY_INFO) and col.endswith('_permeability')]
    property_cols = [col for col in df.columns if col in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Rings']]

    if not assay_cols:
        return created_plots

    try:
        # 1. Correlation heatmap
        if len(assay_cols) > 1:
            fig, ax = plt.subplots(figsize=viz_config["figsize"])
            correlation_matrix = df[assay_cols].corr()

            if HAS_SEABORN:
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                          square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            else:
                im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(assay_cols)))
                ax.set_yticks(range(len(assay_cols)))
                ax.set_xticklabels([col.replace('_permeability', '') for col in assay_cols], rotation=45)
                ax.set_yticklabels([col.replace('_permeability', '') for col in assay_cols])
                plt.colorbar(im, ax=ax, label='Correlation')

                # Add correlation values
                for i in range(len(assay_cols)):
                    for j in range(len(assay_cols)):
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")

            plt.title('Assay Correlation Heatmap')
            plt.tight_layout()
            heatmap_file = output_dir / "correlation_heatmap.png"
            plt.savefig(heatmap_file, dpi=viz_config["dpi"], bbox_inches='tight')
            plt.close()
            created_plots.append(str(heatmap_file))

        # 2. Permeability distributions
        if assay_cols:
            n_assays = len(assay_cols)
            fig, axes = plt.subplots(2, 2, figsize=viz_config["figsize"])
            axes = axes.flatten()

            for i, col in enumerate(assay_cols[:4]):  # Max 4 assays
                ax = axes[i]
                assay_name = col.replace('_permeability', '')
                data = df[col].dropna()

                ax.hist(data, bins=20, alpha=0.7, color=ASSAY_INFO.get(assay_name, {}).get('color', 'blue'))
                ax.axvline(ASSAY_INFO.get(assay_name, {}).get('threshold', -6), color='red',
                          linestyle='--', label=f'Threshold ({ASSAY_INFO.get(assay_name, {}).get("threshold", -6)})')
                ax.set_xlabel('Permeability (log cm/s)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{assay_name.upper()} Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for j in range(n_assays, 4):
                axes[j].set_visible(False)

            plt.tight_layout()
            dist_file = output_dir / "permeability_distributions.png"
            plt.savefig(dist_file, dpi=viz_config["dpi"], bbox_inches='tight')
            plt.close()
            created_plots.append(str(dist_file))

        # 3. Pairwise scatter plots
        if len(assay_cols) > 1:
            n_assays = min(len(assay_cols), 4)
            fig, axes = plt.subplots(n_assays-1, n_assays-1, figsize=viz_config["figsize"])
            if n_assays == 2:
                axes = [[axes]]
            elif n_assays == 3:
                axes = [axes]

            for i in range(n_assays-1):
                for j in range(n_assays-1):
                    if j > i:
                        axes[i][j].set_visible(False)
                        continue

                    ax = axes[i][j] if n_assays > 2 else axes[0][0] if n_assays == 2 else axes
                    x_col = assay_cols[j]
                    y_col = assay_cols[i+1]

                    ax.scatter(df[x_col], df[y_col], alpha=0.6)
                    ax.set_xlabel(x_col.replace('_permeability', ''))
                    ax.set_ylabel(y_col.replace('_permeability', ''))

                    # Add correlation coefficient
                    corr = df[x_col].corr(df[y_col])
                    ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            scatter_file = output_dir / "pairwise_scatterplots.png"
            plt.savefig(scatter_file, dpi=viz_config["dpi"], bbox_inches='tight')
            plt.close()
            created_plots.append(str(scatter_file))

        # 4. Property correlations (if molecular properties calculated)
        if property_cols and assay_cols:
            combined_cols = assay_cols[:2] + property_cols[:5]  # Limit to prevent overcrowding
            if len(combined_cols) > 2:
                fig, ax = plt.subplots(figsize=viz_config["figsize"])
                correlation_matrix = df[combined_cols].corr()

                if HAS_SEABORN:
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                              square=True, ax=ax, cbar_kws={'label': 'Correlation'})
                else:
                    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(combined_cols)))
                    ax.set_yticks(range(len(combined_cols)))
                    ax.set_xticklabels([col.replace('_permeability', '') for col in combined_cols], rotation=45)
                    ax.set_yticklabels([col.replace('_permeability', '') for col in combined_cols])
                    plt.colorbar(im, ax=ax, label='Correlation')

                plt.title('Permeability-Property Correlations')
                plt.tight_layout()
                prop_corr_file = output_dir / "property_correlations.png"
                plt.savefig(prop_corr_file, dpi=viz_config["dpi"], bbox_inches='tight')
                plt.close()
                created_plots.append(str(prop_corr_file))

    except Exception as e:
        print(f"   ⚠️ Visualization error: {e}")

    return created_plots

def generate_analysis_report(df: pd.DataFrame, statistics: Dict, output_dir: Path):
    """Generate comprehensive analysis report."""
    report_file = output_dir / "analysis_report.md"

    assay_cols = [col for col in df.columns if any(col.startswith(assay) for assay in ASSAY_INFO) and col.endswith('_permeability')]

    with open(report_file, 'w') as f:
        f.write("# Cyclic Peptide Permeability Analysis Report\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Molecules**: {len(df)}\n\n")

        # Dataset Overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Molecules analyzed**: {len(df)}\n")
        f.write(f"- **Assays predicted**: {len(assay_cols)}\n")

        if 'MW' in df.columns:
            property_cols = [col for col in df.columns if col in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA']]
            f.write(f"- **Molecular properties**: {len(property_cols)}\n")

        # Permeability Summary
        f.write("\n## Permeability Summary\n\n")
        f.write("| Assay | Mean (log cm/s) | Std | Range | Permeable (%) |\n")
        f.write("|-------|----------------|-----|-------|---------------|\n")

        for assay in ASSAY_INFO.keys():
            perm_col = f"{assay}_permeability"
            binary_col = f"{assay}_permeable"

            if perm_col in df.columns:
                data = df[perm_col].dropna()
                mean_val = data.mean()
                std_val = data.std()
                min_val = data.min()
                max_val = data.max()

                permeable_pct = ""
                if binary_col in df.columns:
                    perm_count = df[binary_col].sum()
                    perm_pct = (perm_count / len(df)) * 100
                    permeable_pct = f"{perm_count}/{len(df)} ({perm_pct:.1f}%)"

                f.write(f"| {assay.upper()} | {mean_val:.2f} | {std_val:.2f} | [{min_val:.2f}, {max_val:.2f}] | {permeable_pct} |\n")

        # Molecular Properties (if available)
        if 'MW' in df.columns:
            f.write("\n## Molecular Properties\n\n")
            f.write("| Property | Mean | Std | Range |\n")
            f.write("|----------|------|-----|-------|\n")

            prop_names = {
                'MW': 'Molecular Weight',
                'LogP': 'LogP',
                'HBD': 'H-Bond Donors',
                'HBA': 'H-Bond Acceptors',
                'TPSA': 'Topological PSA',
                'RotBonds': 'Rotatable Bonds',
                'Rings': 'Ring Count'
            }

            for prop, name in prop_names.items():
                if prop in df.columns:
                    data = df[prop].dropna()
                    if len(data) > 0:
                        mean_val = data.mean()
                        std_val = data.std()
                        min_val = data.min()
                        max_val = data.max()
                        f.write(f"| {name} | {mean_val:.2f} | {std_val:.2f} | [{min_val:.2f}, {max_val:.2f}] |\n")

        # Correlations
        if len(assay_cols) > 1 and 'correlation_matrix' in statistics:
            f.write("\n## Assay Correlations\n\n")
            corr_matrix = statistics['correlation_matrix']
            f.write("| Assay Pair | Correlation |\n")
            f.write("|------------|-------------|\n")

            for i, assay1 in enumerate(assay_cols):
                for j, assay2 in enumerate(assay_cols):
                    if i < j:
                        corr = corr_matrix.loc[assay1, assay2]
                        assay1_name = assay1.replace('_permeability', '').upper()
                        assay2_name = assay2.replace('_permeability', '').upper()
                        f.write(f"| {assay1_name} - {assay2_name} | {corr:.3f} |\n")

        f.write(f"\n---\n*Report generated by batch_analysis.py*\n")

    return str(report_file)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_analysis(
    input_file: Union[str, Path],
    output_dir: Union[str, Path] = "./results/analysis",
    calculate_properties: bool = True,
    create_visualizations: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Comprehensive batch analysis of cyclic peptide permeability predictions.

    Args:
        input_file: Path to input CSV file with 'smiles' column
        output_dir: Directory to save analysis files
        calculate_properties: Calculate molecular properties
        create_visualizations: Create analysis plots
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: DataFrame with all predictions and properties
            - statistics: Analysis statistics
            - output_files: Paths to created files
            - metadata: Analysis metadata

    Example:
        >>> result = run_batch_analysis("input.csv", "results/analysis")
        >>> print(result['statistics']['binary_stats'])
    """
    # Setup
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    config["calculate_properties"] = calculate_properties
    config["create_visualizations"] = create_visualizations
    device = torch.device(config["device"])

    print(f"📊 Comprehensive cyclic peptide analysis")
    print(f"   📁 Input: {input_file}")
    print(f"   📁 Output: {output_dir}")
    print(f"   💻 Device: {device}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Start with base results
    results = df[['smiles']].copy()

    # Predict permeability for all assays
    print(f"   🎯 Predicting membrane permeability...")
    predictions = predict_all_assays(df, config, device)

    # Add predictions to results
    for assay_name, pred_values in predictions.items():
        results[f'{assay_name}_permeability'] = pred_values

        # Add binary classification
        threshold = ASSAY_INFO[assay_name]["threshold"]
        results[f'{assay_name}_permeable'] = (pred_values > threshold).astype(int)
        results[f'{assay_name}_probability'] = sigmoid((pred_values - threshold) * 5)

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        permeable_count = (pred_values > threshold).sum()
        print(f"     {assay_name.upper()}: {mean_pred:.2f} ± {std_pred:.2f}, {permeable_count}/{len(pred_values)} permeable")

    # Calculate molecular properties if requested
    if calculate_properties:
        print(f"   🧪 Calculating molecular properties...")
        try:
            properties = calculate_molecular_properties(df['smiles'].tolist())
            for col in properties.columns:
                results[col] = properties[col]
            print(f"     Calculated {len(properties.columns)} properties")
        except Exception as e:
            print(f"     ⚠️ Property calculation failed: {e}")

    # Perform statistical analysis
    print(f"   📊 Performing statistical analysis...")
    statistics = perform_statistical_analysis(results, output_dir)

    # Create visualizations if requested
    created_plots = []
    if create_visualizations:
        print(f"   📈 Creating visualizations...")
        created_plots = create_visualizations(results, output_dir, config)
        print(f"     Created {len(created_plots)} plots")

    # Save complete results
    results_file = output_dir / "complete_results.csv"
    results.to_csv(results_file, index=False)

    # Generate analysis report
    print(f"   📝 Generating analysis report...")
    report_file = generate_analysis_report(results, statistics, output_dir)

    # Collect all output files
    output_files = {
        "complete_results": str(results_file),
        "analysis_report": report_file,
        "plots": created_plots
    }

    # Add statistical files
    for stat_file in ["statistical_summary.csv", "correlation_matrix.csv"]:
        file_path = output_dir / stat_file
        if file_path.exists():
            output_files[stat_file.replace('.csv', '')] = str(file_path)

    print(f"   💾 Analysis files created:")
    for file_type, file_path in output_files.items():
        if isinstance(file_path, list):
            print(f"      {file_type}: {len(file_path)} files")
        else:
            file_size = Path(file_path).stat().st_size
            print(f"      {file_type}: {Path(file_path).name} ({file_size:,} bytes)")

    print(f"   ✅ Analysis complete")

    return {
        "results": results,
        "statistics": statistics,
        "output_files": output_files,
        "metadata": {
            "input_file": str(input_file),
            "n_molecules": len(results),
            "n_assays": len(predictions),
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
    parser.add_argument('--output-dir', '-o', default='./results/analysis',
                       help='Output directory for analysis files')
    parser.add_argument('--config', '-c',
                       help='Configuration file (JSON)')
    parser.add_argument('--device', default='cpu',
                       help='Device for inference (cpu/cuda)')
    parser.add_argument('--calculate-properties', action='store_true', default=True,
                       help='Calculate molecular properties')
    parser.add_argument('--no-properties', dest='calculate_properties',
                       action='store_false',
                       help='Skip molecular property calculation')
    parser.add_argument('--create-visualizations', action='store_true', default=True,
                       help='Create analysis visualizations')
    parser.add_argument('--no-plots', dest='create_visualizations',
                       action='store_false',
                       help='Skip visualization creation')

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

    # Run analysis
    try:
        result = run_batch_analysis(
            input_file=args.input,
            output_dir=args.output_dir,
            calculate_properties=args.calculate_properties,
            create_visualizations=args.create_visualizations,
            config=config
        )

        print(f"\n🎉 Success! Analysis completed")
        print(f"   📁 Results: {args.output_dir}")
        print(f"   📊 Files: {sum(1 for files in result['output_files'].values() for file in (files if isinstance(files, list) else [files]))}")
        print(f"   🧬 Molecules: {result['metadata']['n_molecules']}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()