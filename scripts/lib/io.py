"""Shared I/O functions for cyclic peptide MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import pandas as pd
import json
import pickle

def parse_input_file(input_file: Union[str, Path]) -> pd.DataFrame:
    """Parse input CSV file with SMILES data.

    Expected format: CSV with 'smiles' column, optional 'y' column for labels.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    # Standardize column names
    if 'SMILES' in df.columns:
        df = df.rename(columns={'SMILES': 'smiles'})
    if 'Y' in df.columns:
        df = df.rename(columns={'Y': 'y'})

    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must have a 'smiles' or 'SMILES' column")

    # Add dummy y column if not present (required by some data_utils)
    if 'y' not in df.columns:
        df['y'] = -10.0

    return df

def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to CSV file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, **kwargs)

def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        return json.load(f)

def save_config(config: Dict[str, Any], config_file: Union[str, Path]) -> None:
    """Save configuration to JSON file."""
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """Save data to pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_temp_csv(df: pd.DataFrame, temp_dir: Path, name: str = "temp_input.csv") -> Path:
    """Create temporary CSV file for data_utils loading."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / name
    df.to_csv(temp_file, index=False)
    return temp_file

def cleanup_temp_files(temp_dir: Path) -> None:
    """Clean up temporary files and directory."""
    if temp_dir.exists():
        for file in temp_dir.iterdir():
            file.unlink()
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file information (size, exists, etc.)."""
    file_path = Path(file_path)

    if not file_path.exists():
        return {"exists": False, "size": 0}

    stat = file_path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "name": file_path.name,
        "stem": file_path.stem,
        "suffix": file_path.suffix,
        "modified": stat.st_mtime
    }