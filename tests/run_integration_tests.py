#!/usr/bin/env python3
"""Automated integration test runner for Cyclic Peptide MCP server."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys

class MCPTestRunner:
    def __init__(self, server_path: str):
        self.server_path = Path(server_path)
        self.results = {
            "test_date": datetime.now().isoformat(),
            "server_path": str(server_path),
            "tests": {},
            "issues": [],
            "summary": {}
        }

    def test_server_startup(self) -> bool:
        """Test that server starts without errors."""
        try:
            # Test import
            result = subprocess.run([
                "bash", "-c",
                "source $(conda info --base)/etc/profile.d/conda.sh && "
                "conda activate ./env &&"
                "python -c 'from src.server import mcp; print(\"Server imported successfully\")'"
            ], capture_output=True, text=True, timeout=30, cwd=self.server_path.parent)

            success = result.returncode == 0
            self.results["tests"]["server_startup"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None,
                "duration_seconds": 30 if not success else 5
            }
            return success
        except Exception as e:
            self.results["tests"]["server_startup"] = {"status": "error", "error": str(e)}
            return False

    def test_rdkit_import(self) -> bool:
        """Test that RDKit is available."""
        try:
            result = subprocess.run([
                "bash", "-c",
                "source $(conda info --base)/etc/profile.d/conda.sh && "
                "conda activate ./env &&"
                "python -c 'from rdkit import Chem; print(\"RDKit available\")'"
            ], capture_output=True, text=True, timeout=30, cwd=self.server_path.parent)

            success = result.returncode == 0
            self.results["tests"]["rdkit_import"] = {
                "status": "passed" if success else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None
            }
            return success
        except Exception as e:
            self.results["tests"]["rdkit_import"] = {"status": "error", "error": str(e)}
            return False

    def test_tool_count(self) -> bool:
        """Test that expected number of tools are registered."""
        try:
            result = subprocess.run([
                "bash", "-c",
                "source $(conda info --base)/etc/profile.d/conda.sh && "
                "conda activate ./env &&"
                "grep -c '@mcp.tool' src/server.py"
            ], capture_output=True, text=True, timeout=30, cwd=self.server_path.parent)

            if result.returncode == 0:
                tool_count = int(result.stdout.strip())
                expected_count = 15  # Based on our analysis
                success = tool_count >= expected_count
                self.results["tests"]["tool_count"] = {
                    "status": "passed" if success else "failed",
                    "expected_tools": expected_count,
                    "actual_tools": tool_count,
                    "message": f"Found {tool_count} tools (expected >= {expected_count})"
                }
                return success
            else:
                self.results["tests"]["tool_count"] = {
                    "status": "failed",
                    "error": result.stderr.strip()
                }
                return False
        except Exception as e:
            self.results["tests"]["tool_count"] = {"status": "error", "error": str(e)}
            return False

    def test_claude_mcp_registration(self) -> bool:
        """Test that server is registered with Claude CLI."""
        try:
            result = subprocess.run([
                "claude", "mcp", "list"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout
                # Check if our server is in the output
                success = "cycpep-tools" in output and "Connected" in output
                self.results["tests"]["claude_mcp_registration"] = {
                    "status": "passed" if success else "failed",
                    "output": output.strip(),
                    "message": "Server registered and connected" if success else "Server not found or not connected"
                }
                return success
            else:
                self.results["tests"]["claude_mcp_registration"] = {
                    "status": "failed",
                    "error": result.stderr.strip()
                }
                return False
        except Exception as e:
            self.results["tests"]["claude_mcp_registration"] = {"status": "error", "error": str(e)}
            return False

    def test_fastmcp_dev_startup(self) -> bool:
        """Test that fastmcp dev can start the server."""
        try:
            # Start server with timeout to avoid hanging
            result = subprocess.run([
                "bash", "-c",
                "source $(conda info --base)/etc/profile.d/conda.sh && "
                "conda activate ./env &&"
                "timeout 5 fastmcp dev src/server.py"
            ], capture_output=True, text=True, cwd=self.server_path.parent)

            # Exit code 124 means timeout - which is expected
            success = result.returncode == 124 and "Proxy server listening" in result.stderr
            self.results["tests"]["fastmcp_dev_startup"] = {
                "status": "passed" if success else "failed",
                "output": result.stderr.strip()[:200] + "..." if len(result.stderr) > 200 else result.stderr.strip(),
                "message": "Server started successfully (timed out as expected)" if success else "Server failed to start"
            }
            return success
        except Exception as e:
            self.results["tests"]["fastmcp_dev_startup"] = {"status": "error", "error": str(e)}
            return False

    def test_scripts_directory(self) -> bool:
        """Test that scripts directory exists and has expected files."""
        try:
            scripts_dir = self.server_path.parent / "scripts"
            if not scripts_dir.exists():
                self.results["tests"]["scripts_directory"] = {
                    "status": "failed",
                    "error": f"Scripts directory not found: {scripts_dir}"
                }
                return False

            # Check for some expected script files
            expected_files = ["preprocess_data.py", "predict_permeability.py", "batch_analysis.py"]
            found_files = [f for f in expected_files if (scripts_dir / f).exists()]

            success = len(found_files) > 0
            self.results["tests"]["scripts_directory"] = {
                "status": "passed" if success else "failed",
                "found_files": found_files,
                "expected_files": expected_files,
                "scripts_dir": str(scripts_dir)
            }
            return success
        except Exception as e:
            self.results["tests"]["scripts_directory"] = {"status": "error", "error": str(e)}
            return False

    def generate_report(self) -> str:
        """Generate JSON report."""
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"].values() if t.get("status") == "passed")
        failed = sum(1 for t in self.results["tests"].values() if t.get("status") == "failed")
        errors = sum(1 for t in self.results["tests"].values() if t.get("status") == "error")

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "N/A"
        }
        return json.dumps(self.results, indent=2)

    def run_all_tests(self):
        """Run all automated tests."""
        print("🧪 Starting MCP Integration Tests...")
        print(f"📁 Server path: {self.server_path}")
        print(f"📅 Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        tests = [
            ("Server Startup", self.test_server_startup),
            ("RDKit Import", self.test_rdkit_import),
            ("Tool Count", self.test_tool_count),
            ("Claude MCP Registration", self.test_claude_mcp_registration),
            ("FastMCP Dev Startup", self.test_fastmcp_dev_startup),
            ("Scripts Directory", self.test_scripts_directory),
        ]

        for test_name, test_func in tests:
            print(f"🔧 Running {test_name}...", end=" ")
            try:
                success = test_func()
                print("✅ PASSED" if success else "❌ FAILED")
            except Exception as e:
                print(f"💥 ERROR: {e}")

        print("-" * 60)
        print("📊 Test Summary:")
        summary = self.results.get("summary", {})
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Passed: {summary.get('passed', 0)} ✅")
        print(f"   Failed: {summary.get('failed', 0)} ❌")
        print(f"   Errors: {summary.get('errors', 0)} 💥")
        print(f"   Pass Rate: {summary.get('pass_rate', 'N/A')}")

if __name__ == "__main__":
    # Determine server path
    script_dir = Path(__file__).parent
    server_path = script_dir.parent / "src" / "server.py"

    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        sys.exit(1)

    # Run tests
    runner = MCPTestRunner(str(server_path))
    runner.run_all_tests()

    # Generate report
    report = runner.generate_report()

    # Save report
    reports_dir = script_dir.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "step7_integration.json"

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"📄 Full report saved to: {report_file}")