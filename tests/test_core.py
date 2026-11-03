"""Unit tests for FORGE core mathematical and geometry models."""
import math
import unittest
from forge.core.geometry import Vec3, Triangle, calculate_mesh_volume, calculate_mesh_surface_area
from forge.core.hashes import canonical_json, sha256_hash
from forge.core.models import DuctSpec, CrossSectionShape, DuctMaterial


class CoreTests(unittest.TestCase):
    def test_vec3_operations(self):
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, 5.0, 6.0)
        self.assertEqual(v1 + v2, Vec3(5.0, 7.0, 9.0))
        self.assertEqual(v2 - v1, Vec3(3.0, 3.0, 3.0))
        self.assertEqual(v1 * 2.0, Vec3(2.0, 4.0, 6.0))
        self.assertEqual(2.0 * v1, Vec3(2.0, 4.0, 6.0))
        self.assertEqual(v2 / 2.0, Vec3(2.0, 2.5, 3.0))
        self.assertEqual(v1.dot(v2), 32.0)
        self.assertEqual(v1.cross(v2), Vec3(-3.0, 6.0, -3.0))
        self.assertAlmostEqual(v1.length(), (1 + 4 + 9) ** 0.5)
        self.assertEqual(Vec3(0.0, 0.0, 0.0).normalized(), Vec3(0.0, 0.0, 0.0))
        self.assertEqual(v1.to_tuple(), (1.0, 2.0, 3.0))

    def test_triangle_and_mesh_surface_area(self):
        t = Triangle(Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1))
        self.assertEqual(t.normal, Vec3(0, 0, 1))
        self.assertAlmostEqual(t.area, 0.5)
        self.assertAlmostEqual(calculate_mesh_surface_area([t]), 0.5)
        self.assertAlmostEqual(calculate_mesh_volume([t]), 1.0 / 6.0)

    def test_canonical_json_and_sha256_uniqueness(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        self.assertEqual(canonical_json(d1), '{"a":1,"b":2}')
        self.assertEqual(sha256_hash(d1), sha256_hash(d2))

        # Baseline duct has 6 canonical keys
        s_base = DuctSpec(inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50, length=100, wall=2)
        self.assertEqual(len(s_base.to_canonical_dict()), 6)

        # Advanced duct has distinct fingerprint from baseline duct
        s_adv = DuctSpec(
            inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50,
            length=100, wall=2, offset_x=15.0, guide_vanes=2, inlet_shape=CrossSectionShape.CIRCULAR
        )
        self.assertNotEqual(sha256_hash(s_base.to_canonical_dict()), sha256_hash(s_adv.to_canonical_dict()))

    def test_duct_spec_validation(self):
        spec = DuctSpec(
            inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50,
            length=100, wall=2, corner_radius=5.0, material=DuctMaterial.PC_CF
        )
        self.assertEqual(spec.material, DuctMaterial.PC_CF)

        # Wall too thick
        with self.assertRaises(ValueError):
            DuctSpec(inlet_width=10, inlet_height=10, outlet_width=10, outlet_height=10, length=50, wall=6)

        # Corner radius too large
        with self.assertRaises(ValueError):
            DuctSpec(inlet_width=10, inlet_height=10, outlet_width=10, outlet_height=10, length=50, wall=1, corner_radius=6)

        # Non-finite dimension rejected
        with self.assertRaises(ValueError):
            DuctSpec(inlet_width=math.nan, inlet_height=10, outlet_width=10, outlet_height=10, length=50, wall=1)


if __name__ == "__main__":
    unittest.main()
