"""Unit tests for NSGA-II inverse optimization and reporting."""
import tempfile
from pathlib import Path
import unittest
from forge.core.models import PhysicsConditions
from forge.opt.search_space import DuctSearchSpace, sample_latin_hypercube
from forge.opt.optimizer import run_nsga2_optimization
from forge.opt.report import generate_pareto_svg, generate_html_report


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.space = DuctSearchSpace(
            inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50,
        )
        self.conds = PhysicsConditions(volumetric_flow_cfm=100.0)

    def test_latin_hypercube_sampling(self):
        samples = sample_latin_hypercube(self.space, num_samples=15, seed=123)
        self.assertEqual(len(samples), 15)
        for s in samples:
            self.assertTrue(self.space.length_range.min_val <= s.length <= self.space.length_range.max_val)

    def test_nsga2_pareto_optimizer(self):
        pareto = run_nsga2_optimization(
            self.space, self.conds, population_size=10, generations=2, seed=42
        )
        self.assertGreater(len(pareto), 0)
        for c in pareto:
            self.assertGreater(c.physics.total_pressure_drop_pa, 0)
            self.assertGreater(c.mass_grams, 0)

    def test_report_generation(self):
        pareto = run_nsga2_optimization(
            self.space, self.conds, population_size=6, generations=1, seed=42
        )
        svg = generate_pareto_svg(pareto)
        self.assertIn("<svg", svg)
        self.assertIn("circle", svg)

        with tempfile.TemporaryDirectory() as tmp:
            rep_path = Path(tmp) / "report.html"
            generate_html_report(pareto, rep_path)
            self.assertTrue(rep_path.exists())
            text = rep_path.read_text(encoding="utf-8")
            self.assertIn("Pareto-Optimal Candidates", text)


if __name__ == "__main__":
    unittest.main()
