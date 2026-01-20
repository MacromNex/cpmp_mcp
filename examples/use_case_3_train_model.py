#!/usr/bin/env python3
"""
Use Case 3: Train CPMP Model for Cyclic Peptide Membrane Permeability Prediction

This script demonstrates how to train a CPMP (Cyclic Peptide Membrane Permeability)
model using the MAT (Molecular Attention Transformer) architecture for a specific assay.

Based on: repo/CPMP/train_pampa.py, train_caco2.py, train_rrck.py, train_mdck.py
Author: CPMP MCP Tool
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn import metrics

# Add CPMP to Python path
SCRIPT_DIR = Path(__file__).parent
CPMP_DIR = SCRIPT_DIR.parent / "repo" / "CPMP"
sys.path.insert(0, str(CPMP_DIR))

# Import CPMP modules
from featurization.data_utils import construct_loader
from model.transformer import make_model


ASSAY_CONFIGS = {
    'pampa': {
        'name': 'PAMPA (Parallel Artificial Membrane Permeability Assay)',
        'batch_size': 32,
        'default_epochs': 600,
        'data_dir': 'pampa_uff_ig_true'
    },
    'caco2': {
        'name': 'Caco-2 (Human colon adenocarcinoma cell line)',
        'batch_size': 64,
        'default_epochs': 600,
        'data_dir': 'caco2_uff_ig_true'
    },
    'rrck': {
        'name': 'RRCK (Ralph Russ Canine Kidney cells)',
        'batch_size': 64,
        'default_epochs': 600,
        'data_dir': 'rrck_uff_ig_true'
    },
    'mdck': {
        'name': 'MDCK (Madin-Darby Canine Kidney cells)',
        'batch_size': 64,
        'default_epochs': 600,
        'data_dir': 'mdck_uff_ig_true'
    }
}


def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: CPMP model to train
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss per sample
    """
    model.train()
    sample_size = 0
    total_loss = 0

    for batch in data_loader:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

        # Move to device
        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)
        batch_mask = batch_mask.to(device)
        y = y.to(device)

        # Forward pass
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        sample_size += len(y)

    return total_loss / sample_size


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: CPMP model to evaluate
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (loss, r2, mse, mae, predictions, targets)
    """
    model.eval()
    outputs = []
    targets = []
    sample_size = 0
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            # Move to device
            adjacency_matrix = adjacency_matrix.to(device)
            node_features = node_features.to(device)
            distance_matrix = distance_matrix.to(device)
            batch_mask = batch_mask.to(device)
            y = y.to(device)

            # Forward pass
            output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            loss = criterion(output, y)

            # Collect predictions and targets
            targets.extend(y.view(-1).cpu().numpy().tolist())
            outputs.extend(output.view(-1).cpu().numpy().tolist())

            total_loss += loss.item()
            sample_size += len(y)

    # Calculate metrics
    r2 = metrics.r2_score(targets, outputs)
    mse = metrics.mean_squared_error(targets, outputs)
    mae = metrics.mean_absolute_error(targets, outputs)

    return total_loss / sample_size, r2, mse, mae, outputs, targets


def load_preprocessed_data(data_dir, assay):
    """
    Load preprocessed training data.

    Args:
        data_dir: Directory containing preprocessed data
        assay: Assay type

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assay_dir = data_dir / ASSAY_CONFIGS[assay]['data_dir']

    if not assay_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory {assay_dir} not found. "
            f"Please run data preprocessing first or download preprocessed data."
        )

    # Load pickled data files
    X_train = pd.read_pickle(assay_dir / 'X_train.pkl').values.tolist()
    X_val = pd.read_pickle(assay_dir / 'X_val.pkl').values.tolist()
    X_test = pd.read_pickle(assay_dir / 'X_test.pkl').values.tolist()
    y_train = pd.read_pickle(assay_dir / 'y_train.pkl').values.tolist()
    y_val = pd.read_pickle(assay_dir / 'y_val.pkl').values.tolist()
    y_test = pd.read_pickle(assay_dir / 'y_test.pkl').values.tolist()

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Train CPMP model for cyclic peptide membrane permeability prediction"
    )
    parser.add_argument(
        "--assay", "-a",
        choices=list(ASSAY_CONFIGS.keys()),
        required=True,
        help="Assay type to train model for"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="../repo/CPMP/data",
        help="Directory containing preprocessed data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="training_output",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs (default: assay-specific)"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-3,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda:0, etc.)"
    )
    parser.add_argument(
        "--merge-train-val",
        action="store_true",
        help="Merge training and validation sets for final training"
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🧬 CPMP Model Training - {ASSAY_CONFIGS[args.assay]['name']}")
    print("=" * 70)
    print(f"🎯 Assay: {args.assay.upper()}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    print("=" * 70)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Load preprocessed data
        print("\n📚 Loading preprocessed data...")
        data_dir = Path(args.data_dir)
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(data_dir, args.assay)

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")

        # Optionally merge training and validation sets
        if args.merge_train_val:
            print("   🔄 Merging training and validation sets...")
            X_train.extend(X_val)
            y_train.extend(y_val)
            print(f"   Combined training samples: {len(X_train)}")

        # Model hyperparameters (from CPMP paper)
        d_atom = X_train[0][0].shape[1]  # Atom feature dimension
        model_params = {
            'd_atom': d_atom,
            'd_model': 64,          # Transformer hidden dimension
            'N': 6,                 # Number of transformer layers
            'h': 64,                # Number of attention heads
            'N_dense': 2,           # Number of dense layers
            'lambda_attention': 0.1,    # Self-attention weight
            'lambda_distance': 0.6,     # Distance matrix weight
            'leaky_relu_slope': 0.16,   # LeakyReLU slope
            'dense_output_nonlinearity': 'relu',
            'distance_matrix_kernel': 'exp',
            'dropout': 0.1,
            'aggregation_type': 'dummy_node'
        }

        print(f"\n🏗️  Building model...")
        print(f"   Atom features: {d_atom}")
        print(f"   Model dimension: {model_params['d_model']}")
        print(f"   Transformer layers: {model_params['N']}")
        print(f"   Attention heads: {model_params['h']}")

        # Create model
        model = make_model(**model_params)
        model = model.to(device)

        # Training parameters
        assay_config = ASSAY_CONFIGS[args.assay]
        batch_size = assay_config['batch_size']
        epochs = args.epochs or assay_config['default_epochs']
        learning_rate = args.learning_rate

        print(f"\n⚙️  Training parameters:")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Epochs: {epochs}")

        # Setup training
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # File paths for saving
        model_save_path = output_dir / f"{args.assay}.best_weight.pth"
        log_file = output_dir / f"{args.assay}_training.csv"

        print(f"\n🚀 Starting training...")
        print(f"   Model will be saved to: {model_save_path}")
        print(f"   Training log: {log_file}")

        # Training loop
        best_loss = float('inf')
        training_history = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Create data loaders
            data_loader_train = construct_loader(X_train, y_train, batch_size)
            data_loader_test = construct_loader(X_test, y_test, batch_size)

            # Training
            train_loss = train_epoch(model, data_loader_train, criterion, optimizer, device)

            # Evaluation
            test_loss, r2, mse, mae, _, _ = evaluate_model(model, data_loader_test, criterion, device)

            # Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"   💾 New best model saved at epoch {epoch} (test_loss: {test_loss:.4f})")

            # Log results
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'best_loss': best_loss
            }
            training_history.append(epoch_results)

            # Print progress
            if epoch % 50 == 0 or epoch <= 10:
                elapsed_time = (time.time() - start_time) / 60
                print(f"   Epoch {epoch:3d}/{epochs}: "
                      f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
                      f"r2={r2:.3f}, best_loss={best_loss:.4f} "
                      f"(time: {elapsed_time:.1f}min)")

            # Save training log
            results_df = pd.DataFrame(training_history)
            results_df.to_csv(log_file, index=False)

        # Final evaluation
        print(f"\n✅ Training completed!")
        total_time = (time.time() - start_time) / 60
        print(f"   Total training time: {total_time:.1f} minutes")
        print(f"   Best test loss: {best_loss:.4f}")

        # Load best model for final evaluation
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        data_loader_test = construct_loader(X_test, y_test, batch_size)
        _, final_r2, final_mse, final_mae, predictions, targets = evaluate_model(
            model, data_loader_test, criterion, device
        )

        print(f"\n📊 Final test performance:")
        print(f"   R²: {final_r2:.3f}")
        print(f"   MSE: {final_mse:.4f}")
        print(f"   MAE: {final_mae:.4f}")

        # Save final predictions
        predictions_df = pd.DataFrame({
            'target': targets,
            'prediction': predictions
        })
        predictions_file = output_dir / f"{args.assay}_test_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"   Test predictions saved to: {predictions_file}")

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 To prepare training data, you need to:")
        print(f"   1. Download preprocessed data from https://zenodo.org/records/14638776")
        print(f"   2. Or run data preprocessing: cd {CPMP_DIR}/data/{ASSAY_CONFIGS[args.assay]['data_dir']} && python process_data.py")
        return 1

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())