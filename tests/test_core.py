"""Unit tests for FORGE core mathematical and geometry models."""
import unittest
from forge.core.geometry import Vec3, Triangle
from forge.core.hashes import canonical_json, sha256_hash
from forge.core.models import DuctSpec, DuctMaterial


class CoreTests(unittest.TestCase):
    def test_vec3_operations(self):
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, 5.0, 6.0)
        self.assertEqual(v1 + v2, Vec3(5.0, 7.0, 9.0))
        self.assertEqual(v2 - v1, Vec3(3.0, 3.0, 3.0))
        self.assertEqual(v1 * 2.0, Vec3(2.0, 4.0, 6.0))
        self.assertEqual(v1.dot(v2), 32.0)
        # Cross product: (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
        self.assertEqual(v1.cross(v2), Vec3(-3.0, 6.0, -3.0))
        self.assertAlmostEqual(v1.length(), (1 + 4 + 9) ** 0.5)

    def test_triangle_geometry(self):
        t = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
        self.assertEqual(t.normal, Vec3(0, 0, 1))
        self.assertAlmostEqual(t.area, 0.5)

    def test_canonical_json_and_sha256(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        self.assertEqual(canonical_json(d1), '{"a":1,"b":2}')
        self.assertEqual(sha256_hash(d1), sha256_hash(d2))

    def test_duct_spec_validation(self):
        # Valid spec
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


if __name__ == "__main__":
    unittest.main()
