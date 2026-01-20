#!/usr/bin/env python3
"""
Use Case 5: Batch Analysis and Visualization of Cyclic Peptide Permeability Predictions

This script demonstrates how to perform comprehensive batch analysis of cyclic peptide
membrane permeability predictions, including statistical analysis, visualization,
and comparative studies across multiple assays.

Based on: Combined functionality for analysis and visualization
Author: CPMP MCP Tool
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from scipy import stats

# Add CPMP to Python path
SCRIPT_DIR = Path(__file__).parent
CPMP_DIR = SCRIPT_DIR.parent / "repo" / "CPMP"
sys.path.insert(0, str(CPMP_DIR))

# Import CPMP modules
from featurization.data_utils import load_data_from_df, construct_loader
from model.transformer import make_model


ASSAY_INFO = {
    'pampa': {
        'name': 'PAMPA',
        'full_name': 'Parallel Artificial Membrane Permeability Assay',
        'description': 'Measures passive diffusion through artificial lipid membrane',
        'r2': 0.67,
        'batch_size': 32,
        'threshold': -6.0,
        'color': '#1f77b4',
        'applications': 'Drug screening, passive permeability assessment'
    },
    'caco2': {
        'name': 'Caco-2',
        'full_name': 'Human colon adenocarcinoma cell line',
        'description': 'Models intestinal absorption and oral bioavailability',
        'r2': 0.75,
        'batch_size': 64,
        'threshold': -6.0,
        'color': '#ff7f0e',
        'applications': 'Oral drug development, intestinal permeability'
    },
    'rrck': {
        'name': 'RRCK',
        'full_name': 'Ralph Russ Canine Kidney cells',
        'description': 'Evaluates blood-brain barrier permeability',
        'r2': 0.62,
        'batch_size': 64,
        'threshold': -6.0,
        'color': '#2ca02c',
        'applications': 'CNS drug development, BBB penetration'
    },
    'mdck': {
        'name': 'MDCK',
        'full_name': 'Madin-Darby Canine Kidney cells',
        'description': 'Models general membrane permeability',
        'r2': 0.73,
        'batch_size': 64,
        'threshold': -6.0,
        'color': '#d62728',
        'applications': 'General permeability screening, renal clearance'
    }
}


def predict_all_assays(data_loader, device="cpu"):
    """
    Predict permeability for all assays.

    Args:
        data_loader: PyTorch data loader with featurized molecules
        device: Device to run inference on

    Returns:
        Dictionary with predictions for each assay
    """
    # Model hyperparameters
    model_params = {
        'd_atom': 28,
        'd_model': 64,
        'N': 6,
        'h': 64,
        'N_dense': 2,
        'lambda_attention': 0.1,
        'lambda_distance': 0.6,
        'leaky_relu_slope': 0.16,
        'dense_output_nonlinearity': 'relu',
        'distance_matrix_kernel': 'exp',
        'dropout': 0.1,
        'aggregation_type': 'dummy_node'
    }

    predictions = {}

    for assay in ASSAY_INFO.keys():
        print(f"   🔬 Running {assay.upper()} predictions...")

        # Load model
        model = make_model(**model_params)
        model_path = CPMP_DIR / "saved_model" / f"{assay}.best_wegiht.pth"

        if not model_path.exists():
            print(f"      ⚠️ Model file {model_path} not found, skipping {assay}")
            continue

        # Load trained weights
        if device == "cpu":
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        else:
            checkpoint = torch.load(model_path, weights_only=True)

        model_state_dict = model.state_dict()
        for name, param in checkpoint.items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

        model = model.to(device)
        model.eval()

        # Run predictions
        assay_predictions = []
        with torch.no_grad():
            for batch in data_loader:
                adjacency_matrix, node_features, distance_matrix, _ = batch
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

                # Move to device
                adjacency_matrix = adjacency_matrix.to(device)
                node_features = node_features.to(device)
                distance_matrix = distance_matrix.to(device)
                batch_mask = batch_mask.to(device)

                # Forward pass
                output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
                assay_predictions.extend(output.view(-1).cpu().numpy().tolist())

        predictions[assay] = np.array(assay_predictions)

    return predictions


def calculate_molecular_properties(smiles_list):
    """
    Calculate basic molecular properties using RDKit.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        DataFrame with molecular properties
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError:
        print("   ⚠️ RDKit not available for molecular property calculation")
        return pd.DataFrame()

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


def perform_statistical_analysis(df, output_dir):
    """
    Perform statistical analysis of predictions.

    Args:
        df: DataFrame with predictions
        output_dir: Output directory for results
    """
    print(f"\n📊 Performing statistical analysis...")

    assays = [col for col in df.columns if col in ASSAY_INFO]

    # Basic statistics
    stats_summary = df[assays].describe()
    print(f"   📈 Descriptive statistics:")
    print(stats_summary.round(3))

    # Correlation analysis
    print(f"\n   🔗 Correlation between assays:")
    correlation_matrix = df[assays].corr()
    print(correlation_matrix.round(3))

    # Binary classification statistics
    print(f"\n   🚪 Permeability classification (threshold: -6.0):")
    for assay in assays:
        threshold = ASSAY_INFO[assay]['threshold']
        permeable = (df[assay] > threshold).sum()
        total = len(df)
        print(f"      {assay.upper()}: {permeable}/{total} ({permeable/total*100:.1f}%) permeable")

    # Save statistics
    stats_file = output_dir / "statistical_summary.csv"
    stats_summary.to_csv(stats_file)

    correlation_file = output_dir / "correlation_matrix.csv"
    correlation_matrix.to_csv(correlation_file)

    print(f"   💾 Statistics saved to {stats_file}")
    print(f"   💾 Correlation matrix saved to {correlation_file}")

    return stats_summary, correlation_matrix


def create_visualizations(df, output_dir):
    """
    Create visualization plots.

    Args:
        df: DataFrame with predictions and properties
        output_dir: Output directory for plots
    """
    print(f"\n📊 Creating visualizations...")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    assays = [col for col in df.columns if col in ASSAY_INFO]

    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, assay in enumerate(assays):
        if i < len(axes):
            ax = axes[i]
            ax.hist(df[assay], bins=30, alpha=0.7, color=ASSAY_INFO[assay]['color'],
                   edgecolor='black', linewidth=0.5)
            ax.axvline(ASSAY_INFO[assay]['threshold'], color='red', linestyle='--',
                      label=f'Threshold ({ASSAY_INFO[assay]["threshold"]})')
            ax.set_xlabel('Log Permeability (cm/s)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{ASSAY_INFO[assay]["name"]} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_plot = output_dir / "permeability_distributions.png"
    plt.savefig(dist_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[assays].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Assay Correlation Matrix')
    plt.tight_layout()
    corr_plot = output_dir / "correlation_heatmap.png"
    plt.savefig(corr_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Pairwise scatter plots
    if len(assays) >= 2:
        fig, axes = plt.subplots(len(assays)-1, len(assays)-1, figsize=(12, 10))
        if len(assays) == 2:
            axes = [axes]

        for i in range(len(assays)-1):
            for j in range(len(assays)-1):
                if j >= i:
                    continue
                ax = axes[i][j] if len(assays) > 2 else axes[j]
                x_assay = assays[j]
                y_assay = assays[i+1]

                ax.scatter(df[x_assay], df[y_assay], alpha=0.6, s=20)
                ax.set_xlabel(f'{ASSAY_INFO[x_assay]["name"]} (log cm/s)')
                ax.set_ylabel(f'{ASSAY_INFO[y_assay]["name"]} (log cm/s)')

                # Add threshold lines
                ax.axvline(ASSAY_INFO[x_assay]['threshold'], color='red', linestyle='--', alpha=0.5)
                ax.axhline(ASSAY_INFO[y_assay]['threshold'], color='red', linestyle='--', alpha=0.5)

                # Calculate and display correlation
                corr = df[x_assay].corr(df[y_assay])
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_plot = output_dir / "pairwise_scatterplots.png"
        plt.savefig(scatter_plot, dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Molecular property correlations (if available)
    property_cols = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Rings']
    available_props = [col for col in property_cols if col in df.columns and not df[col].isna().all()]

    if available_props and assays:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, assay in enumerate(assays[:4]):
            if i < len(axes):
                ax = axes[i]
                # Show correlation with LogP as example
                if 'LogP' in available_props:
                    ax.scatter(df['LogP'], df[assay], alpha=0.6, s=20)
                    ax.set_xlabel('Molecular LogP')
                    ax.set_ylabel(f'{ASSAY_INFO[assay]["name"]} (log cm/s)')
                    ax.set_title(f'{ASSAY_INFO[assay]["name"]} vs LogP')

                    # Calculate correlation
                    corr = df['LogP'].corr(df[assay])
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        prop_plot = output_dir / "property_correlations.png"
        plt.savefig(prop_plot, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"   📈 Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch analysis and visualization of cyclic peptide permeability predictions"
    )
    parser.add_argument(
        "--input", "-i",
        default="examples/data/sample_cyclic_peptides.csv",
        help="Input CSV file with SMILES column"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="batch_analysis_output",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda:0"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--calculate-properties",
        action="store_true",
        help="Calculate molecular properties using RDKit"
    )
    parser.add_argument(
        "--max-molecules", "-m",
        type=int,
        default=1000,
        help="Maximum number of molecules to analyze (default: 1000)"
    )

    args = parser.parse_args()

    print("🧬 CPMP Batch Analysis and Visualization")
    print("=" * 50)
    print(f"📁 Input file: {args.input}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🖥️  Device: {args.device}")
    print("=" * 50)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Check input file
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file {args.input} not found!")
            return 1

        # Load input data
        print(f"\n🔄 Loading input data...")
        df = pd.read_csv(args.input)
        print(f"   Found {len(df)} molecules")

        if 'smiles' not in df.columns:
            print("❌ Error: Input CSV must contain 'smiles' column")
            return 1

        # Limit molecules if specified
        if len(df) > args.max_molecules:
            df = df.head(args.max_molecules)
            print(f"   Limited to {len(df)} molecules for analysis")

        # Featurize molecules
        print(f"\n🧪 Featurizing molecules...")
        os.chdir(str(CPMP_DIR))
        X, y = load_data_from_df(
            args.input,
            ff='uff',
            ignoreInterfragInteractions=True,
            one_hot_formal_charge=True,
            use_data_saving=False
        )

        # Create data loader
        data_loader = construct_loader(X, y, batch_size=2, shuffle=False)
        print(f"   ✅ Featurization complete")

        # Run predictions for all assays
        print(f"\n🔬 Running predictions for all assays...")
        predictions = predict_all_assays(data_loader, args.device)

        if not predictions:
            print("❌ Error: No model predictions could be generated")
            return 1

        # Create results dataframe
        results_df = pd.DataFrame({'smiles': df['smiles'].values[:len(list(predictions.values())[0])]})

        for assay, preds in predictions.items():
            results_df[assay] = preds

            # Add binary classification
            threshold = ASSAY_INFO[assay]['threshold']
            results_df[f'{assay}_permeable'] = (preds > threshold).astype(int)

        # Calculate molecular properties if requested
        if args.calculate_properties:
            print(f"\n🧮 Calculating molecular properties...")
            props_df = calculate_molecular_properties(results_df['smiles'].tolist())
            if not props_df.empty:
                results_df = pd.concat([results_df, props_df], axis=1)
                print(f"   ✅ Added {len(props_df.columns)} molecular properties")

        # Save complete results
        results_file = output_dir / "complete_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Complete results saved: {results_file}")

        # Perform statistical analysis
        stats_summary, correlation_matrix = perform_statistical_analysis(results_df, output_dir)

        # Create visualizations
        create_visualizations(results_df, output_dir)

        # Generate summary report
        print(f"\n📋 Generating summary report...")

        report_content = f"""# CPMP Batch Analysis Report

## Dataset Summary
- Total molecules analyzed: {len(results_df)}
- Assays predicted: {', '.join(predictions.keys())}
- Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Permeability Statistics
"""

        for assay in predictions.keys():
            perm_count = results_df[f'{assay}_permeable'].sum()
            perm_pct = perm_count / len(results_df) * 100
            mean_perm = results_df[assay].mean()
            std_perm = results_df[assay].std()

            report_content += f"""
### {ASSAY_INFO[assay]['name']}
- Mean permeability: {mean_perm:.3f} ± {std_perm:.3f} log cm/s
- Permeable molecules: {perm_count}/{len(results_df)} ({perm_pct:.1f}%)
- Range: [{results_df[assay].min():.3f}, {results_df[assay].max():.3f}]
"""

        report_content += f"""
## Assay Correlations
{correlation_matrix.round(3).to_string()}

## Files Generated
- complete_results.csv: All predictions and properties
- statistical_summary.csv: Descriptive statistics
- correlation_matrix.csv: Assay correlation matrix
- permeability_distributions.png: Distribution plots
- correlation_heatmap.png: Correlation visualization
- pairwise_scatterplots.png: Pairwise comparisons
"""

        if args.calculate_properties:
            report_content += "- property_correlations.png: Molecular property correlations\n"

        report_file = output_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        print(f"   📄 Analysis report saved: {report_file}")

        print(f"\n✅ Batch analysis complete!")
        print(f"   📊 Analyzed {len(results_df)} molecules")
        print(f"   🔬 Predicted {len(predictions)} assays")
        print(f"   📁 Results saved in: {output_dir}")

        return 0

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())