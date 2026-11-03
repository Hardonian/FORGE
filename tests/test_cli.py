"""Unit tests for FORGE Typer CLI subcommands."""
import tempfile
from pathlib import Path
import unittest
from typer.testing import CliRunner
from forge.cli.main import app


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.spec_json = Path("examples/duct.json")

    def test_version_command(self):
        res = self.runner.invoke(app, ["--version"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("FORGE", res.output)

    def test_generate_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidate"
            res = self.runner.invoke(app, ["generate", str(self.spec_json), "--out", str(out), "--format", "stl"])
            self.assertEqual(res.exit_code, 0)
            self.assertTrue((out / "duct.stl").exists())
            self.assertTrue((out / "report.json").exists())

    def test_generate_3mf_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidate_3mf"
            res = self.runner.invoke(app, ["generate", str(self.spec_json), "--out", str(out), "--format", "3mf"])
            self.assertEqual(res.exit_code, 0)
            self.assertTrue((out / "duct.3mf").exists())

    def test_simulate_command(self):
        res = self.runner.invoke(app, ["simulate", str(self.spec_json), "--cfm", "120"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Total Pressure Drop", res.output)

    def test_dfm_command(self):
        res = self.runner.invoke(app, ["dfm", str(self.spec_json)])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("DFM Analysis", res.output)

    def test_openfoam_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "openfoam_run"
            res = self.runner.invoke(app, ["openfoam", str(self.spec_json), "--out", str(out)])
            self.assertEqual(res.exit_code, 0)
            self.assertTrue((out / "system" / "controlDict").exists())

    def test_su2_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "su2.cfg"
            res = self.runner.invoke(app, ["su2", str(self.spec_json), "--out", str(out)])
            self.assertEqual(res.exit_code, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()

