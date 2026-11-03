"""Unit tests for FORGE CAD, mesh generation, exporters, and DFM."""
from collections import Counter
import json
import struct
import unittest
from forge.core.models import DuctSpec, DuctMaterial, CrossSectionShape, ServerFanStandard
from forge.cad.mesh import (
    build_duct_mesh,
    compute_normal_wall_compensation,
    split_duct_for_printing,
    mesh_to_WebGL_buffers,
)
from forge.cad.flanges import generate_flange_mesh, FAN_STANDARDS, get_fan_hole_positions
from forge.cad.exporters import (
    export_ascii_stl,
    export_binary_stl,
    export_3mf,
    export_obj,
    export_step,
    export_gltf,
    export_glb,
)
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
        counts = Counter()
        for t in tris:
            for a, b in [(t.a, t.b), (t.b, t.c), (t.c, t.a)]:
                k = ((round(a.x, 3), round(a.y, 3), round(a.z, 3)),
                     (round(b.x, 3), round(b.y, 3), round(b.z, 3)))
                counts[k] += 1
        for (u, v), _ in counts.items():
            self.assertIn((v, u), counts)

    def test_circular_to_rectangular_morphed_loft(self):
        morph_spec = DuctSpec(
            inlet_width=120, inlet_height=120, outlet_width=140, outlet_height=50,
            length=100, wall=2.0, inlet_shape=CrossSectionShape.CIRCULAR,
            outlet_shape=CrossSectionShape.RECTANGULAR, preserve_normal_thickness=True,
            fan_standard=ServerFanStandard.FAN_120MM, flange_thickness=4.0,
        )
        tris = build_duct_mesh(morph_spec)
        self.assertGreater(len(tris), 100)
        buffers = mesh_to_WebGL_buffers(tris)
        self.assertEqual(buffers["triangle_count"], len(tris))
        self.assertEqual(len(buffers["positions"]), len(tris) * 9)

    def test_normal_wall_compensation_and_splitting(self):
        s_uncomp = DuctSpec(
            inlet_width=80, inlet_height=80, outlet_width=160, outlet_height=160,
            length=50, wall=2.0, preserve_normal_thickness=False
        )
        s_comp = s_uncomp.model_copy(update={"preserve_normal_thickness": True})
        wx1, _, min_n1 = compute_normal_wall_compensation(s_uncomp)
        wx2, _, min_n2 = compute_normal_wall_compensation(s_comp)
        self.assertLess(min_n1, 2.0)
        self.assertAlmostEqual(min_n2, 2.0, places=4)
        self.assertGreater(wx2, wx1)

        part_a, part_b = split_duct_for_printing(self.spec, split_ratio=0.4)
        self.assertAlmostEqual(part_a.length + part_b.length, self.spec.length, places=3)
        self.assertEqual(part_a.outlet_width, part_b.inlet_width)

    def test_exporters(self):
        tris = build_duct_mesh(self.spec)

        ascii_stl = export_ascii_stl(tris)
        self.assertTrue(ascii_stl.startswith("solid forge_duct"))
        self.assertTrue(ascii_stl.endswith("endsolid forge_duct\n"))

        bin_stl = export_binary_stl(tris)
        expected_len = 84 + len(tris) * 50
        self.assertEqual(len(bin_stl), expected_len)

        mf_bytes = export_3mf(tris)
        self.assertEqual(mf_bytes[:2], b"PK")

        obj_text = export_obj(tris)
        self.assertIn("v ", obj_text)
        self.assertIn("vn ", obj_text)
        self.assertIn("f ", obj_text)

        step_text = export_step(tris)
        self.assertIn("MANIFOLD_SOLID_BREP", step_text)
        self.assertIn("CLOSED_SHELL", step_text)

        gltf_text = export_gltf(tris)
        gltf_json = json.loads(gltf_text)
        self.assertEqual(gltf_json["asset"]["version"], "2.0")

        glb_bytes = export_glb(tris)
        magic, version, total_len = struct.unpack("<III", glb_bytes[:12])
        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(total_len, len(glb_bytes))

    def test_dfm_analysis_and_build_plate_overhang_exemption(self):
        # A straight vertical duct has 0 degree overhangs (bottom rim rests on print bed)
        straight_spec = DuctSpec(
            inlet_width=100, inlet_height=80, outlet_width=100, outlet_height=80,
            length=100, wall=2.0, material=DuctMaterial.PC_CF
        )
        straight_tris = build_duct_mesh(straight_spec)
        straight_dfm = analyze_dfm(straight_spec, straight_tris)
        self.assertEqual(straight_dfm.max_overhang_angle_deg, 0.0)
        self.assertTrue(straight_dfm.overhang_compliance)
        self.assertEqual(len(straight_dfm.warnings), 0)

    def test_flange_generation(self):
        flange_tris = generate_flange_mesh(120, 120, 140, 140, 4.0, z_start=0.0)
        self.assertGreater(len(flange_tris), 0)
        holes = get_fan_hole_positions(120.0, 105.0, z=0.0)
        self.assertEqual(len(holes), 4)
        self.assertIn("120mm", FAN_STANDARDS)


if __name__ == "__main__":
    unittest.main()
