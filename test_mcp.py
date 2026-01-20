#!/usr/bin/env python3
"""Test script for MCP server functionality."""

import sys
import tempfile
from pathlib import Path
import pandas as pd

# Add src to path to import server
sys.path.insert(0, str(Path(__file__).parent / "src"))

from server import mcp

def test_server_info():
    """Test server info tool."""
    print("🧪 Testing server info...")
    try:
        # Get server info
        info = mcp.call_tool("get_server_info", {})
        print(f"✅ Server info: {info['status']}")
        print(f"   - Server: {info['server_name']}")
        print(f"   - Total tools: {info['total_tools']}")
        print(f"   - Sync tools: {info['tools']['synchronous']['count']}")
        print(f"   - Submit tools: {info['tools']['asynchronous']['count']}")
        return True
    except Exception as e:
        print(f"❌ Server info failed: {e}")
        return False

def test_smiles_validation():
    """Test SMILES validation tool."""
    print("\n🧪 Testing SMILES validation...")
    try:
        # Test with a simple cyclic molecule
        test_smiles = ["CC(=O)NC1CCCC1C(=O)O", "INVALID_SMILES", "CCO"]
        result = mcp.call_tool("validate_cyclic_peptide_smiles", {"smiles_list": test_smiles})

        if result["status"] == "success":
            print(f"✅ Validation completed for {result['total']} SMILES")
            for r in result["results"]:
                status = "✅" if r["is_valid"] else "❌"
                cyclic = "🔄" if r["is_cyclic"] else "⚫"
                print(f"   {status} {cyclic} {r['smiles'][:20]}... valid={r['is_valid']}, cyclic={r['is_cyclic']}")
            return True
        else:
            print(f"❌ Validation failed: {result['error']}")
            return False
    except Exception as e:
        print(f"❌ SMILES validation failed: {e}")
        return False

def test_preprocessing_sync():
    """Test synchronous preprocessing with sample data."""
    print("\n🧪 Testing synchronous preprocessing...")
    try:
        # Check if sample data exists
        sample_file = Path("examples/data/sample_cyclic_peptides.csv")
        if not sample_file.exists():
            print(f"⚠️ Sample data not found at {sample_file}, skipping test")
            return True

        # Test preprocessing
        with tempfile.TemporaryDirectory() as temp_dir:
            result = mcp.call_tool("preprocess_cyclic_peptide_data", {
                "input_file": str(sample_file),
                "output_dir": temp_dir,
                "skip_featurization": True
            })

            if result["status"] == "success":
                print(f"✅ Preprocessing completed")
                print(f"   - Train size: {result.get('statistics', {}).get('train_size', 'unknown')}")
                print(f"   - Output files: {len(result.get('output_files', {}))}")
                return True
            else:
                print(f"❌ Preprocessing failed: {result['error']}")
                return False

    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_job_submission():
    """Test job submission and management."""
    print("\n🧪 Testing job submission...")
    try:
        # Check if sample data exists
        sample_file = Path("examples/data/sample_cyclic_peptides.csv")
        if not sample_file.exists():
            print(f"⚠️ Sample data not found at {sample_file}, creating test data")
            # Create minimal test data
            test_data = pd.DataFrame({
                'smiles': ['CC(=O)NC1CCCC1C(=O)O', 'CNC(C)C(=O)NC1CCCC1'],
                'y': [-5.5, -6.2]
            })
            sample_file.parent.mkdir(parents=True, exist_ok=True)
            test_data.to_csv(sample_file, index=False)

        # Submit a preprocessing job
        result = mcp.call_tool("submit_preprocess_data", {
            "input_file": str(sample_file),
            "job_name": "test_preprocessing"
        })

        if result["status"] == "submitted":
            job_id = result["job_id"]
            print(f"✅ Job submitted with ID: {job_id}")

            # Check job status
            status = mcp.call_tool("get_job_status", {"job_id": job_id})
            if "job_id" in status:
                print(f"✅ Job status retrieved: {status['status']}")

                # List jobs
                jobs = mcp.call_tool("list_jobs", {})
                if jobs["status"] == "success":
                    print(f"✅ Jobs listed: {jobs['total']} total")
                    return True
                else:
                    print(f"❌ List jobs failed: {jobs['error']}")
                    return False
            else:
                print(f"❌ Job status failed: {status}")
                return False
        else:
            print(f"❌ Job submission failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Job submission test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing CycPep MCP Server\n")

    tests = [
        test_server_info,
        test_smiles_validation,
        test_preprocessing_sync,
        test_job_submission
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)