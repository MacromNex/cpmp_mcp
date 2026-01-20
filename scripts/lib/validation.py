"""Shared validation functions for cyclic peptide MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem

def validate_smiles_list(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Validate a list of SMILES strings.

    Returns:
        Tuple of (valid_mask, error_messages)
    """
    valid_mask = np.zeros(len(smiles_list), dtype=bool)
    errors = []

    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mask[i] = True
            else:
                errors.append(f"Index {i}: Invalid SMILES '{smiles}'")
        except Exception as e:
            errors.append(f"Index {i}: Error parsing '{smiles}': {e}")

    return valid_mask, errors

def validate_input_dataframe(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> List[str]:
    """Validate input DataFrame structure.

    Returns:
        List of validation error messages
    """
    errors = []

    if df is None or df.empty:
        errors.append("DataFrame is empty or None")
        return errors

    # Check required columns
    required_cols = required_cols or ['smiles']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Required column '{col}' not found")

    # Check for duplicate SMILES
    if 'smiles' in df.columns:
        duplicates = df['smiles'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate SMILES")

        # Check for empty SMILES
        empty_smiles = df['smiles'].isna().sum() + (df['smiles'] == '').sum()
        if empty_smiles > 0:
            errors.append(f"Found {empty_smiles} empty SMILES")

    return errors

def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> List[str]:
    """Validate file path.

    Returns:
        List of validation error messages
    """
    errors = []

    try:
        file_path = Path(file_path)
    except Exception as e:
        errors.append(f"Invalid file path: {e}")
        return errors

    if must_exist and not file_path.exists():
        errors.append(f"File does not exist: {file_path}")

    if must_exist and file_path.is_dir():
        errors.append(f"Path is a directory, not a file: {file_path}")

    return errors

def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> List[str]:
    """Validate configuration dictionary.

    Returns:
        List of validation error messages
    """
    errors = []

    if not isinstance(config, dict):
        errors.append("Config must be a dictionary")
        return errors

    if required_keys:
        for key in required_keys:
            if key not in config:
                errors.append(f"Required config key '{key}' not found")

    # Validate device
    if 'device' in config:
        valid_devices = ['cpu', 'cuda', 'mps']
        if config['device'] not in valid_devices:
            errors.append(f"Invalid device '{config['device']}'. Must be one of: {valid_devices}")

    return errors

def validate_assay_name(assay: str, available_assays: List[str]) -> List[str]:
    """Validate assay name.

    Returns:
        List of validation error messages
    """
    errors = []

    if not isinstance(assay, str):
        errors.append("Assay name must be a string")
        return errors

    if assay.lower() not in [a.lower() for a in available_assays]:
        errors.append(f"Unknown assay '{assay}'. Available: {available_assays}")

    return errors

def validate_predictions(predictions: np.ndarray, expected_range: Optional[Tuple[float, float]] = None) -> List[str]:
    """Validate prediction array.

    Returns:
        List of validation error messages
    """
    errors = []

    if predictions is None:
        errors.append("Predictions array is None")
        return errors

    if not isinstance(predictions, np.ndarray):
        errors.append("Predictions must be numpy array")
        return errors

    if len(predictions) == 0:
        errors.append("Predictions array is empty")
        return errors

    # Check for NaN or infinite values
    nan_count = np.isnan(predictions).sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values in predictions")

    inf_count = np.isinf(predictions).sum()
    if inf_count > 0:
        errors.append(f"Found {inf_count} infinite values in predictions")

    # Check range if specified
    if expected_range:
        min_val, max_val = expected_range
        out_of_range = ((predictions < min_val) | (predictions > max_val)).sum()
        if out_of_range > 0:
            errors.append(f"Found {out_of_range} predictions outside expected range [{min_val}, {max_val}]")

    return errors

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid function with overflow protection."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))