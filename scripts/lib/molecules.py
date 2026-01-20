"""Shared molecular manipulation functions for cyclic peptide MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES string to RDKit molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    mol = parse_smiles(smiles)
    return mol is not None

def is_cyclic_peptide(mol: Chem.Mol) -> bool:
    """Check if molecule is a cyclic peptide."""
    if mol is None:
        return False

    ring_info = mol.GetRingInfo()
    return ring_info.NumRings() > 0

def calculate_basic_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate basic molecular properties."""
    if mol is None:
        return {prop: np.nan for prop in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Rings']}

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'Rings': Descriptors.RingCount(mol)
    }

def calculate_molecular_properties(smiles_list: List[str]) -> pd.DataFrame:
    """Calculate molecular properties for a list of SMILES."""
    properties = []

    for smiles in smiles_list:
        mol = parse_smiles(smiles)
        props = calculate_basic_properties(mol)
        properties.append(props)

    return pd.DataFrame(properties)

def generate_3d_conformer(mol: Chem.Mol, num_conformers: int = 1) -> Chem.Mol:
    """Generate 3D conformer(s) for a molecule."""
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol

def save_molecule(mol: Chem.Mol, file_path: Union[str, Path], format: str = "pdb") -> None:
    """Save molecule to file in specified format."""
    if mol is None:
        raise ValueError("Cannot save None molecule")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "pdb":
        Chem.MolToPDBFile(mol, str(file_path))
    elif format.lower() == "sdf":
        writer = Chem.SDWriter(str(file_path))
        writer.write(mol)
        writer.close()
    elif format.lower() == "smi":
        with open(file_path, 'w') as f:
            f.write(Chem.MolToSmiles(mol))
    else:
        raise ValueError(f"Unsupported format: {format}")