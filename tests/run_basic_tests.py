#!/usr/bin/env python3
"""Basic integration tests for MCP server - simplified version."""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime

def run_test(name: str, command: str, expected_in_output: str = None, expected_return_code: int = 0) -> dict:
    """Run a single test command."""
    print(f"🔧 Testing {name}...", end=" ")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        success = result.returncode == expected_return_code
        if expected_in_output and success:
            success = expected_in_output.lower() in result.stdout.lower() or expected_in_output.lower() in result.stderr.lower()

        test_result = {
            "name": name,
            "status": "passed" if success else "failed",
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }

        print("✅ PASSED" if success else "❌ FAILED")
        if not success and result.stderr:
            print(f"   Error: {result.stderr.strip()[:100]}")

        return test_result

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return {
            "name": name,
            "status": "timeout",
            "command": command,
            "error": "Command timed out after 30 seconds"
        }
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return {
            "name": name,
            "status": "error",
            "command": command,
            "error": str(e)
        }

def main():
    print("🧪 Running Basic MCP Integration Tests")
    print("=" * 60)

    # Get current directory (should be project root)
    project_root = Path.cwd()
    print(f"📁 Project root: {project_root}")

    tests = [
        # Test 1: Environment activation and Python path
        ("Environment Activation",
         "source $(conda info --base)/etc/profile.d/conda.sh && conda activate ./env && python -c 'import sys; print(sys.executable)'",
         "/env/bin/python"),

        # Test 2: Server import
        ("Server Import",
         "source $(conda info --base)/etc/profile.d/conda.sh && conda activate ./env && python -c 'from src.server import mcp; print(\"Import successful\")'",
         "Import successful"),

        # Test 3: RDKit availability
        ("RDKit Import",
         "source $(conda info --base)/etc/profile.d/conda.sh && conda activate ./env && python -c 'from rdkit import Chem; print(\"RDKit OK\")'",
         "RDKit OK"),

        # Test 4: Tool count
        ("Tool Count",
         "grep -c '@mcp.tool' src/server.py",
         "15"),

        # Test 5: Scripts directory
        ("Scripts Directory",
         "ls -la scripts/",
         "preprocess_data"),

        # Test 6: Job directory
        ("Jobs Directory",
         "ls -la src/jobs/",
         "manager.py"),

        # Test 7: Claude MCP registration
        ("Claude MCP List",
         "claude mcp list",
         "cycpep-tools"),

        # Test 8: FastMCP dev (with timeout)
        ("FastMCP Dev Server",
         "source $(conda info --base)/etc/profile.d/conda.sh && conda activate ./env && timeout 3 fastmcp dev src/server.py",
         "Proxy server listening",
         124)  # timeout exit code
    ]

    results = []
    passed = 0

    for test_name, command, expected_output, *expected_code in tests:
        expected_return_code = expected_code[0] if expected_code else 0
        result = run_test(test_name, command, expected_output, expected_return_code)
        results.append(result)
        if result["status"] == "passed":
            passed += 1

    # Summary
    total = len(tests)
    print("=" * 60)
    print(f"📊 Test Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")

    # Save detailed results
    report = {
        "test_date": datetime.now().isoformat(),
        "project_root": str(project_root),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed/total*100:.1f}%"
        },
        "tests": results
    }

    report_file = project_root / "reports" / "step7_basic_tests.json"
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Report saved to: {report_file}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)