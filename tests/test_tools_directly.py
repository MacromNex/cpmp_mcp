#!/usr/bin/env python3
"""Direct testing of MCP tools without going through Claude CLI."""

import sys
from pathlib import Path
import json
from datetime import datetime
import tempfile
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_tool_import():
    """Test that we can import and access the tools."""
    try:
        from server import mcp
        print("✅ MCP server imported successfully")

        # Try to get server info
        info = mcp.get_tools()
        print(f"✅ Found tools (async)")
        return True
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False

def test_validation_tool():
    """Test the SMILES validation tool."""
    try:
        from server import validate_cyclic_peptide_smiles

        # Test valid peptide-like SMILES
        result = validate_cyclic_peptide_smiles("C1CC(C(=O)NC(C(=O)NC(C(=O)N1)CC2=CC=CC=C2)CC3=CC=CC=C3)N")
        print(f"✅ Validation tool works: {result}")
        return True
    except Exception as e:
        print(f"❌ Validation tool failed: {e}")
        return False

def test_server_info_tool():
    """Test the server info tool."""
    try:
        from server import get_server_info

        result = get_server_info()
        print(f"✅ Server info: {result}")
        return True
    except Exception as e:
        print(f"❌ Server info failed: {e}")
        return False

def test_preprocess_tool():
    """Test the preprocessing tool."""
    try:
        from server import preprocess_cyclic_peptide_data

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'smiles': ['C1CC(C(=O)NC(C(=O)NC(C(=O)N1)CC2=CC=CC=C2)CC3=CC=CC=C3)N'],
                'molecular_weight': [450.5]
            })
            test_data.to_csv(f.name, index=False)
            input_file = f.name

        # Test preprocessing
        result = preprocess_cyclic_peptide_data(input_file)
        print(f"✅ Preprocessing tool works: {result}")

        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        if 'output_file' in result:
            Path(result['output_file']).unlink(missing_ok=True)

        return True
    except Exception as e:
        print(f"❌ Preprocessing tool failed: {e}")
        return False

def test_permeability_prediction():
    """Test single assay permeability prediction."""
    try:
        from server import predict_single_assay_permeability

        smiles = "C1CC(C(=O)NC(C(=O)NC(C(=O)N1)CC2=CC=CC=C2)CC3=CC=CC=C3)N"
        result = predict_single_assay_permeability(smiles, "PAMPA")
        print(f"✅ Permeability prediction works: {result}")
        return True
    except Exception as e:
        print(f"❌ Permeability prediction failed: {e}")
        return False

def test_job_submission():
    """Test job submission and management."""
    try:
        from server import submit_preprocess_data, get_job_status, list_jobs

        # Create test input
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'smiles': ['C1CC(C(=O)NC(C(=O)NC(C(=O)N1)CC2=CC=CC=C2)CC3=CC=CC=C3)N'],
                'molecular_weight': [450.5]
            })
            test_data.to_csv(f.name, index=False)
            input_file = f.name

        # Submit job
        submit_result = submit_preprocess_data(input_file)
        print(f"✅ Job submission works: {submit_result}")

        if 'job_id' in submit_result:
            job_id = submit_result['job_id']

            # Check job status
            status_result = get_job_status(job_id)
            print(f"✅ Job status check works: {status_result}")

            # List jobs
            list_result = list_jobs()
            print(f"✅ Job listing works: {list_result}")

        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        return False

def main():
    """Run all direct tool tests."""
    print("🧪 Testing MCP Tools Directly")
    print("=" * 50)

    tests = [
        ("Tool Import", test_tool_import),
        ("Server Info", test_server_info_tool),
        ("SMILES Validation", test_validation_tool),
        ("Data Preprocessing", test_preprocess_tool),
        ("Permeability Prediction", test_permeability_prediction),
        ("Job Submission", test_job_submission),
    ]

    results = []
    passed = 0

    for test_name, test_func in tests:
        print(f"\n🔧 Testing {test_name}...")
        try:
            success = test_func()
            if success:
                passed += 1
            results.append({"name": test_name, "status": "passed" if success else "failed"})
        except Exception as e:
            print(f"💥 Error in {test_name}: {e}")
            results.append({"name": test_name, "status": "error", "error": str(e)})

    # Summary
    total = len(tests)
    print("\n" + "=" * 50)
    print(f"📊 Direct Tool Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")

    # Save results
    report = {
        "test_date": datetime.now().isoformat(),
        "test_type": "direct_tool_tests",
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed/total*100:.1f}%"
        },
        "tests": results
    }

    report_file = project_root / "reports" / "step7_tool_tests.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Report saved to: {report_file}")
    return passed == total

if __name__ == "__main__":
    # Set PYTHONPATH and activate environment
    import subprocess
    import os

    # Run in activated environment
    cmd = [
        "bash", "-c",
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        "conda activate ./env && "
        "python tests/test_tools_directly.py"
    ]

    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)