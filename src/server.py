#!/usr/bin/env python3
"""MCP Server for Cyclic Peptide Tools

Provides both synchronous and asynchronous (submit) APIs for cyclic peptide
computational tools. Built on the clean scripts from Step 5.

Usage:
    fastmcp dev src/server.py
    # or
    mamba run --prefix ./env fastmcp dev src/server.py
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import sys
import tempfile
import json
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("cycpep-tools")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted cyclic peptide computation job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors

    Example:
        get_job_status("abc12345")
        # Returns: {"job_id": "abc12345", "status": "running", "submitted_at": "2025-01-01T10:00:00"}
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed cyclic peptide computation job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed

    Example:
        get_job_result("abc12345")
        # Returns: {"status": "success", "result": {...}, "output_files": [...]}
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count

    Example:
        get_job_log("abc12345", 20)
        # Returns: {"status": "success", "log_lines": [...], "total_lines": 156}
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running cyclic peptide computation job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message

    Example:
        cancel_job("abc12345")
        # Returns: {"status": "success", "message": "Job abc12345 cancelled"}
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted cyclic peptide computation jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status

    Example:
        list_jobs("completed")
        # Returns: {"status": "success", "jobs": [...], "total": 5}
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def preprocess_cyclic_peptide_data(
    input_file: str,
    output_dir: Optional[str] = None,
    assay: Optional[str] = None,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    min_y: float = -10.0,
    skip_featurization: bool = True
) -> dict:
    """
    Preprocess and split cyclic peptide datasets for model training.

    Fast operation - returns results immediately (~15 seconds for small datasets).

    Args:
        input_file: Path to CSV file with 'smiles' and 'y' columns
        output_dir: Directory for output files (optional, creates temp if not provided)
        assay: Predefined assay dataset (caco2, pampa, rrck, mdck) instead of input_file
        train_size: Fraction for training set (default: 0.7)
        val_size: Fraction for validation set (default: 0.1)
        test_size: Fraction for test set (default: 0.2)
        min_y: Minimum y value filter (default: -10.0)
        skip_featurization: Skip molecular featurization (default: True)

    Returns:
        Dictionary with preprocessing results and output files

    Example:
        preprocess_cyclic_peptide_data("data/peptides.csv", "results/", train_size=0.8)
    """
    try:
        from preprocess_data import run_preprocess_data

        # Create temp output dir if not provided
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir

        result = run_preprocess_data(
            assay=assay,
            dataset_path=input_file if not assay else None,
            output_dir=output_dir,
            skip_featurization=skip_featurization,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            min_y=min_y
        )

        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_single_assay_permeability(
    assay: str,
    input_file: str,
    output_file: Optional[str] = None,
    with_labels: bool = True,
    device: str = "cpu",
    batch_size: int = 32
) -> dict:
    """
    Predict cyclic peptide membrane permeability for a specific assay.

    Fast operation for small datasets - returns results immediately (~3 min for 19 molecules).

    Args:
        assay: Assay name (pampa, caco2, rrck, mdck)
        input_file: Path to CSV file with 'smiles' column
        output_file: Output CSV path (optional, creates temp file if not provided)
        with_labels: Include binary classification labels (default: True)
        device: Device for computation (cpu, cuda) (default: cpu)
        batch_size: Batch size for processing (default: 32)

    Returns:
        Dictionary with predictions and metrics

    Example:
        predict_single_assay_permeability("caco2", "peptides.csv", with_labels=True)
    """
    try:
        from predict_single_assay import run_predict_single_assay

        # Create temp output file if not provided
        if output_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            output_file = temp_file.name
            temp_file.close()

        result = run_predict_single_assay(
            assay=assay,
            input_file=input_file,
            output_file=output_file,
            with_labels=with_labels,
            device=device,
            batch_size=batch_size
        )

        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Single assay prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_all_assays_permeability(
    input_file: str,
    output_file: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 32
) -> dict:
    """
    Predict cyclic peptide membrane permeability across all 4 assays.

    Fast operation for small datasets - returns results immediately (~8 min for 19 molecules).

    Args:
        input_file: Path to CSV file with 'smiles' column
        output_file: Output CSV path (optional, creates temp file if not provided)
        device: Device for computation (cpu, cuda) (default: cpu)
        batch_size: Batch size for processing (default: 32)

    Returns:
        Dictionary with all assay predictions

    Example:
        predict_all_assays_permeability("peptides.csv", device="cpu")
    """
    try:
        from predict_all_assays import run_predict_all_assays

        # Create temp output file if not provided
        if output_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            output_file = temp_file.name
            temp_file.close()

        result = run_predict_all_assays(
            input_file=input_file,
            output_file=output_file,
            device=device,
            batch_size=batch_size
        )

        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"All assays prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_cyclic_peptide_batch(
    input_file: str,
    output_dir: Optional[str] = None,
    calculate_properties: bool = True,
    create_visualizations: bool = True,
    device: str = "cpu",
    batch_size: int = 32
) -> dict:
    """
    Comprehensive batch analysis and visualization of cyclic peptide predictions.

    Fast operation for small datasets - returns results immediately (~4 min for 19 molecules).

    Args:
        input_file: Path to CSV file with 'smiles' column
        output_dir: Output directory (optional, creates temp dir if not provided)
        calculate_properties: Calculate molecular properties (default: True)
        create_visualizations: Generate plots (default: True)
        device: Device for computation (cpu, cuda) (default: cpu)
        batch_size: Batch size for processing (default: 32)

    Returns:
        Dictionary with comprehensive analysis results

    Example:
        analyze_cyclic_peptide_batch("peptides.csv", calculate_properties=True)
    """
    try:
        from batch_analysis import run_batch_analysis

        # Create temp output dir if not provided
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir

        result = run_batch_analysis(
            input_file=input_file,
            output_dir=output_dir,
            calculate_properties=calculate_properties,
            create_visualizations=create_visualizations,
            device=device,
            batch_size=batch_size
        )

        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations or large datasets)
# ==============================================================================

@mcp.tool()
def submit_preprocess_data(
    input_file: Optional[str] = None,
    assay: Optional[str] = None,
    output_dir: Optional[str] = None,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    min_y: float = -10.0,
    skip_featurization: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a data preprocessing job for large cyclic peptide datasets.

    Use this for large datasets or when you want to run preprocessing in the background.

    Args:
        input_file: Path to CSV file with 'smiles' and 'y' columns
        assay: Predefined assay dataset (caco2, pampa, rrck, mdck) instead of input_file
        output_dir: Directory for output files
        train_size: Fraction for training set (default: 0.7)
        val_size: Fraction for validation set (default: 0.1)
        test_size: Fraction for test set (default: 0.2)
        min_y: Minimum y value filter (default: -10.0)
        skip_featurization: Skip molecular featurization (default: True)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        submit_preprocess_data("large_dataset.csv", job_name="preprocess_large")
    """
    script_path = str(SCRIPTS_DIR / "preprocess_data.py")

    args = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "min_y": min_y,
        "skip_featurization": skip_featurization
    }

    # Add input source
    if assay:
        args["assay"] = assay
    elif input_file:
        args["input"] = input_file
    else:
        return {"status": "error", "error": "Must provide either input_file or assay"}

    # Add output directory if provided
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "preprocess_data"
    )

@mcp.tool()
def submit_single_assay_prediction(
    assay: str,
    input_file: str,
    output_file: Optional[str] = None,
    with_labels: bool = True,
    device: str = "cpu",
    batch_size: int = 32,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a single assay permeability prediction job for large datasets.

    Use this for large datasets or when you want predictions running in the background.

    Args:
        assay: Assay name (pampa, caco2, rrck, mdck)
        input_file: Path to CSV file with 'smiles' column
        output_file: Output CSV path
        with_labels: Include binary classification labels (default: True)
        device: Device for computation (cpu, cuda) (default: cpu)
        batch_size: Batch size for processing (default: 32)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking

    Example:
        submit_single_assay_prediction("caco2", "large_dataset.csv", job_name="caco2_pred")
    """
    script_path = str(SCRIPTS_DIR / "predict_single_assay.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "assay": assay,
            "input": input_file,
            "output": output_file,
            "with_labels": with_labels,
            "device": device,
            "batch_size": batch_size
        },
        job_name=job_name or f"{assay}_prediction"
    )

@mcp.tool()
def submit_all_assays_prediction(
    input_file: str,
    output_file: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 32,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a multi-assay permeability prediction job for large datasets.

    Use this for large datasets or when processing all 4 assays in the background.

    Args:
        input_file: Path to CSV file with 'smiles' column
        output_file: Output CSV path
        device: Device for computation (cpu, cuda) (default: cpu)
        batch_size: Batch size for processing (default: 32)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking

    Example:
        submit_all_assays_prediction("large_dataset.csv", job_name="all_assays_pred")
    """
    script_path = str(SCRIPTS_DIR / "predict_all_assays.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "output": output_file,
            "device": device,
            "batch_size": batch_size
        },
        job_name=job_name or "all_assays_prediction"
    )

@mcp.tool()
def submit_batch_analysis(
    input_file: str,
    output_dir: Optional[str] = None,
    calculate_properties: bool = True,
    create_visualizations: bool = True,
    device: str = "cpu",
    batch_size: int = 32,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a comprehensive batch analysis job for large cyclic peptide datasets.

    Use this for large datasets or when you want comprehensive analysis running in the background.

    Args:
        input_file: Path to CSV file with 'smiles' column
        output_dir: Output directory for analysis files
        calculate_properties: Calculate molecular properties (default: True)
        create_visualizations: Generate plots (default: True)
        device: Device for computation (cpu, cuda) (default: cpu)
        batch_size: Batch size for processing (default: 32)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking

    Example:
        submit_batch_analysis("large_dataset.csv", job_name="comprehensive_analysis")
    """
    script_path = str(SCRIPTS_DIR / "batch_analysis.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "output_dir": output_dir,
            "calculate_properties": calculate_properties,
            "create_visualizations": create_visualizations,
            "device": device,
            "batch_size": batch_size
        },
        job_name=job_name or "batch_analysis"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_cyclic_peptide_smiles(
    smiles_list: Union[List[str], str]
) -> dict:
    """
    Validate SMILES strings for cyclic peptides.

    Fast validation using RDKit to check if SMILES are valid and represent cyclic structures.

    Args:
        smiles_list: Single SMILES string or list of SMILES strings

    Returns:
        Dictionary with validation results

    Example:
        validate_cyclic_peptide_smiles("CC(=O)NC1CCCC1C(=O)O")
        validate_cyclic_peptide_smiles(["SMILES1", "SMILES2", "SMILES3"])
    """
    try:
        # Import validation function from scripts/lib
        from lib.validation import validate_smiles_list
        from lib.molecules import is_cyclic_peptide, parse_smiles

        # Handle single SMILES or list
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        results = []
        for smiles in smiles_list:
            try:
                # Basic validation
                valid_smiles = validate_smiles_list([smiles])
                is_valid = len(valid_smiles) > 0

                # Check if cyclic peptide
                is_cyclic = False
                error_msg = None

                if is_valid:
                    mol = parse_smiles(smiles)
                    if mol:
                        is_cyclic = is_cyclic_peptide(mol)
                    else:
                        is_valid = False
                        error_msg = "Failed to parse molecule"
                else:
                    error_msg = "Invalid SMILES"

                results.append({
                    "smiles": smiles,
                    "is_valid": is_valid,
                    "is_cyclic": is_cyclic,
                    "error": error_msg
                })

            except Exception as e:
                results.append({
                    "smiles": smiles,
                    "is_valid": False,
                    "is_cyclic": False,
                    "error": str(e)
                })

        return {"status": "success", "results": results, "total": len(results)}

    except Exception as e:
        logger.error(f"SMILES validation failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the MCP server and available tools.

    Returns:
        Dictionary with server information and tool summary
    """
    tools_sync = [
        "preprocess_cyclic_peptide_data",
        "predict_single_assay_permeability",
        "predict_all_assays_permeability",
        "analyze_cyclic_peptide_batch",
        "validate_cyclic_peptide_smiles"
    ]

    tools_submit = [
        "submit_preprocess_data",
        "submit_single_assay_prediction",
        "submit_all_assays_prediction",
        "submit_batch_analysis"
    ]

    tools_job_management = [
        "get_job_status",
        "get_job_result",
        "get_job_log",
        "cancel_job",
        "list_jobs"
    ]

    return {
        "status": "success",
        "server_name": "cycpep-tools",
        "version": "1.0.0",
        "description": "MCP server for cyclic peptide computational tools",
        "tools": {
            "synchronous": {
                "count": len(tools_sync),
                "tools": tools_sync,
                "description": "Fast operations (<10 min), immediate results"
            },
            "asynchronous": {
                "count": len(tools_submit),
                "tools": tools_submit,
                "description": "Long-running or large dataset operations"
            },
            "job_management": {
                "count": len(tools_job_management),
                "tools": tools_job_management,
                "description": "Manage asynchronous jobs"
            }
        },
        "total_tools": len(tools_sync) + len(tools_submit) + len(tools_job_management) + 1,
        "supported_assays": ["pampa", "caco2", "rrck", "mdck"],
        "scripts_directory": str(SCRIPTS_DIR),
        "jobs_directory": str(job_manager.jobs_dir)
    }

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    logger.info("Starting CycPep MCP Server...")
    logger.info(f"Scripts directory: {SCRIPTS_DIR}")
    logger.info(f"Jobs directory: {job_manager.jobs_dir}")
    mcp.run()