"""Unit tests for aerodynamic physics and automated CFD case generators."""
import tempfile
from pathlib import Path
import unittest
from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.fluid_domain import extract_fluid_domain
from forge.physics.openfoam import generate_openfoam_case
from forge.physics.su2 import generate_su2_config


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.spec = DuctSpec(
            inlet_width=120, inlet_height=120, outlet_width=140, outlet_height=45,
            length=100, wall=2.0, guide_vanes=0
        )
        self.conds = PhysicsConditions(volumetric_flow_cfm=150.0, heat_dissipation_watts=350.0)

    def test_aerodynamic_solver(self):
        res = solve_duct_physics(self.spec, self.conds)
        self.assertGreater(res.inlet_velocity_m_s, 5.0)
        self.assertGreater(res.outlet_velocity_m_s, 10.0)
        self.assertGreater(res.reynolds_number, 10000)
        self.assertGreater(res.total_pressure_drop_pa, 0.0)
        self.assertTrue(0.0 <= res.flow_uniformity_index <= 1.0)
        self.assertGreater(res.estimated_temp_rise_c, 0.0)

    def test_guide_vanes_effect(self):
        spec_no_vanes = self.spec
        spec_with_vanes = DuctSpec(
            inlet_width=120, inlet_height=120, outlet_width=140, outlet_height=45,
            length=100, wall=2.0, guide_vanes=2
        )
        res_no = solve_duct_physics(spec_no_vanes, self.conds)
        res_with = solve_duct_physics(spec_with_vanes, self.conds)
        # Guide vanes improve flow uniformity
        self.assertGreaterEqual(res_with.flow_uniformity_index, res_no.flow_uniformity_index)

    def test_fluid_domain_extraction(self):
        patches = extract_fluid_domain(self.spec)
        self.assertIn("inlet", patches)
        self.assertIn("outlet", patches)
        self.assertIn("walls", patches)
        self.assertGreater(len(patches["inlet"]), 0)
        self.assertGreater(len(patches["outlet"]), 0)
        self.assertGreater(len(patches["walls"]), 0)

    def test_openfoam_case_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = generate_openfoam_case(self.spec, self.conds, Path(tmp) / "openfoam_case")
            self.assertTrue((case_dir / "system" / "controlDict").exists())
            self.assertTrue((case_dir / "system" / "fvSchemes").exists())
            self.assertTrue((case_dir / "0" / "U").exists())
            self.assertTrue((case_dir / "0" / "p").exists())
            self.assertTrue((case_dir / "constant" / "turbulenceProperties").exists())
            self.assertTrue((case_dir / "constant" / "triSurface" / "duct_fluid.stl").exists())

    def test_su2_config_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = generate_su2_config(self.spec, self.conds, Path(tmp) / "su2.cfg")
            self.assertTrue(cfg_path.exists())
            text = cfg_path.read_text(encoding="utf-8")
            self.assertIn("SOLVER= INC_RANS", text)
            self.assertIn("MARKER_INLET=", text)


if __name__ == "__main__":
    unittest.main()

