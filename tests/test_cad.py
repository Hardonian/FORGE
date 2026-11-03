"""Unit tests for FORGE CAD, mesh generation, exporters, and DFM."""
from collections import Counter
import unittest
from forge.core.models import DuctSpec, DuctMaterial
from forge.cad.mesh import build_duct_mesh
from forge.cad.flanges import generate_flange_mesh, FAN_STANDARDS
from forge.cad.exporters import export_ascii_stl, export_binary_stl, export_3mf, export_obj, export_step
from forge.cad.dfm import analyze_dfm


class CadTests(unittest.TestCase):
    def setUp(self):
        self.spec = DuctSpec(
            inlet_width=120, inlet_height=120, outlet_width=140, outlet_height=45,
            length=110, wall=2.4, offset_x=15.0, offset_y=-10.0, corner_radius=8.0,
            guide_vanes=1, material=DuctMaterial.PC_CF
        )

    def test_mesh_generation_and_manifold_property(self):
        tris = build_duct_mesh(self.spec)
        self.assertGreater(len(tris), 50)
        # Verify manifold closure: each directed edge (u, v) must have matching (v, u)
        counts = Counter()
        for t in tris:
            for a, b in [(t.a, t.b), (t.b, t.c), (t.c, t.a)]:
                k = ((round(a.x, 3), round(a.y, 3), round(a.z, 3)),
                     (round(b.x, 3), round(b.y, 3), round(b.z, 3)))
                counts[k] += 1
        for (u, v), count in counts.items():
            # In a valid 2-manifold without self-intersections or boundary tears, opposite edge exists
            self.assertIn((v, u), counts)

    def test_exporters(self):
        tris = build_duct_mesh(self.spec)

        ascii_stl = export_ascii_stl(tris)
        self.assertTrue(ascii_stl.startswith("solid forge_duct"))
        self.assertTrue(ascii_stl.endswith("endsolid forge_duct\n"))

        bin_stl = export_binary_stl(tris)
        # Header is 80 bytes + 4 bytes count + len(tris) * 50 bytes
        expected_len = 84 + len(tris) * 50
        self.assertEqual(len(bin_stl), expected_len)

        mf_bytes = export_3mf(tris)
        self.assertTrue(len(mf_bytes) > 200)
        # 3MF is a zip archive starting with PK\x03\x04
        self.assertEqual(mf_bytes[:2], b"PK")

        obj_text = export_obj(tris)
        self.assertIn("v ", obj_text)
        self.assertIn("f ", obj_text)

        step_text = export_step(tris)
        self.assertIn("ISO-10303-21;", step_text)
        self.assertIn("END-ISO-10303-21;", step_text)

    def test_dfm_analysis(self):
        tris = build_duct_mesh(self.spec)
        dfm_rep = analyze_dfm(self.spec, tris, target_printer="Bambu_Lab_X1C")
        self.assertEqual(dfm_rep.material, "PC-CF")
        self.assertGreater(dfm_rep.estimated_mass_grams, 10.0)
        self.assertGreater(dfm_rep.estimated_print_time_hours, 0.5)
        self.assertTrue(dfm_rep.bed_fit)

    def test_flange_generation(self):
        flange_tris = generate_flange_mesh(120, 120, 140, 140, 4.0, z_start=0.0)
        self.assertGreater(len(flange_tris), 0)
        self.assertIn("120mm", FAN_STANDARDS)


if __name__ == "__main__":
    unittest.main()
