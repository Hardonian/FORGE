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

    def test_generate_3mf_and_glb_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_3mf = Path(tmp) / "candidate_3mf"
            res = self.runner.invoke(app, ["generate", str(self.spec_json), "--out", str(out_3mf), "--format", "3mf"])
            self.assertEqual(res.exit_code, 0)
            self.assertTrue((out_3mf / "duct.3mf").exists())

            out_glb = Path(tmp) / "candidate_glb"
            res_glb = self.runner.invoke(app, ["generate", str(self.spec_json), "--out", str(out_glb), "--format", "glb"])
            self.assertEqual(res_glb.exit_code, 0)
            self.assertTrue((out_glb / "duct.glb").exists())

    def test_simulate_command(self):
        res = self.runner.invoke(app, ["simulate", str(self.spec_json), "--cfm", "120"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Total Pressure Drop", res.output)
        self.assertIn("Aeroacoustic Noise", res.output)

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
            self.assertTrue((out / "system" / "snappyHexMeshDict").exists())

    def test_su2_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "su2.cfg"
            res = self.runner.invoke(app, ["su2", str(self.spec_json), "--out", str(out)])
            self.assertEqual(res.exit_code, 0)
            self.assertTrue(out.exists())

    def test_nurbs_step_and_hybrid_aalc_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_stp = Path(tmp) / "nurbs.stp"
            res_nurbs = self.runner.invoke(app, ["nurbs-step", str(self.spec_json), "--out", str(out_stp)])
            self.assertEqual(res_nurbs.exit_code, 0)
            self.assertTrue(out_stp.exists())
            self.assertIn("AP242", out_stp.read_text(encoding="utf-8"))

            out_aalc = Path(tmp) / "aalc_report.json"
            res_aalc = self.runner.invoke(
                app,
                ["hybrid-aalc", str(self.spec_json), "--tdp", "800", "--fluid", "water", "--out", str(out_aalc)],
            )
            self.assertEqual(res_aalc.exit_code, 0)
            self.assertTrue(out_aalc.exists())
            self.assertIn("Capillary Limit", res_aalc.output)

    def test_batch_screen_command(self):
        res = self.runner.invoke(app, ["batch-screen", "--count", "50", "--cfm", "100"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Vectorized Tensor Batch Evaluation Complete", res.output)


if __name__ == "__main__":
    unittest.main()
