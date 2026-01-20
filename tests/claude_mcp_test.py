#!/usr/bin/env python3
"""Test MCP integration through Claude CLI with actual prompts."""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

class ClaudeMCPTester:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = []

    def run_claude_command(self, prompt: str, timeout: int = 60) -> dict:
        """Run a prompt through Claude CLI and capture response."""
        try:
            # Use claude CLI to send a message
            cmd = ["claude", "-m", prompt]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": -1
            }

    def test_tool_discovery(self) -> dict:
        """Test that Claude can discover MCP tools."""
        prompt = "What MCP tools are available? Please list them briefly."
        print("🔍 Testing tool discovery...")
        result = self.run_claude_command(prompt)

        success = result["success"] and "cycpep" in result["stdout"].lower()
        test_result = {
            "name": "Tool Discovery",
            "prompt": prompt,
            "success": success,
            "response_length": len(result.get("stdout", "")),
            "contains_tools": "tool" in result.get("stdout", "").lower(),
            "result": result
        }
        self.results.append(test_result)
        return test_result

    def test_server_info(self) -> dict:
        """Test get server info tool."""
        prompt = "Use the cycpep-tools to get server information. What tools are available?"
        print("ℹ️ Testing server info...")
        result = self.run_claude_command(prompt)

        success = result["success"] and len(result.get("stdout", "")) > 50
        test_result = {
            "name": "Server Info",
            "prompt": prompt,
            "success": success,
            "response_length": len(result.get("stdout", "")),
            "result": result
        }
        self.results.append(test_result)
        return test_result

    def test_smiles_validation(self) -> dict:
        """Test SMILES validation tool."""
        prompt = "Use cycpep-tools to validate this SMILES: 'CC(=O)NC1CCCC1C(=O)O'. Is it a valid cyclic peptide?"
        print("✅ Testing SMILES validation...")
        result = self.run_claude_command(prompt)

        success = result["success"] and ("valid" in result.get("stdout", "").lower() or "invalid" in result.get("stdout", "").lower())
        test_result = {
            "name": "SMILES Validation",
            "prompt": prompt,
            "success": success,
            "response_length": len(result.get("stdout", "")),
            "contains_validation": "valid" in result.get("stdout", "").lower(),
            "result": result
        }
        self.results.append(test_result)
        return test_result

    def test_error_handling(self) -> dict:
        """Test error handling with invalid input."""
        prompt = "Use cycpep-tools to validate this invalid SMILES: 'this_is_not_a_smiles'. What happens?"
        print("❌ Testing error handling...")
        result = self.run_claude_command(prompt)

        success = result["success"] and ("error" in result.get("stdout", "").lower() or "invalid" in result.get("stdout", "").lower())
        test_result = {
            "name": "Error Handling",
            "prompt": prompt,
            "success": success,
            "response_length": len(result.get("stdout", "")),
            "handles_error": "error" in result.get("stdout", "").lower() or "invalid" in result.get("stdout", "").lower(),
            "result": result
        }
        self.results.append(test_result)
        return test_result

    def generate_report(self) -> dict:
        """Generate test report."""
        passed = sum(1 for r in self.results if r["success"])
        total = len(self.results)

        report = {
            "test_date": datetime.now().isoformat(),
            "test_type": "claude_mcp_integration",
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "N/A"
            },
            "tests": self.results
        }
        return report

    def run_all_tests(self):
        """Run all Claude MCP integration tests."""
        print("🤖 Testing MCP Integration through Claude CLI")
        print("=" * 60)

        # Run tests
        self.test_tool_discovery()
        time.sleep(2)  # Brief pause between tests

        self.test_server_info()
        time.sleep(2)

        self.test_smiles_validation()
        time.sleep(2)

        self.test_error_handling()

        # Generate and save report
        report = self.generate_report()

        # Summary
        summary = report["summary"]
        print("\n" + "=" * 60)
        print(f"📊 Claude MCP Tests: {summary['passed']}/{summary['total_tests']} passed ({summary['pass_rate']})")

        # Save report
        report_file = self.project_root / "reports" / "step7_claude_mcp_tests.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Report saved to: {report_file}")

        return summary["passed"] == summary["total_tests"]

def main():
    project_root = Path.cwd()
    tester = ClaudeMCPTester(project_root)
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())