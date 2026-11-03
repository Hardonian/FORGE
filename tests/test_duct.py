import json
import math
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from forge.duct import Duct, metrics, stl_text, triangles


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.d = Duct(100, 80, 120, 50, 100, 2)

    def test_closed_manifold_and_opposite_edges(self):
        counts = Counter()
        for tri in triangles(self.d):
            for a, b in zip(tri, tri[1:] + tri[:1]):
                counts[(a, b)] += 1
        for (a, b), count in counts.items():
            self.assertEqual(count, 1)
            self.assertEqual(counts[(b, a)], 1)

    def test_normals_face_outward(self):
        volume = 0.0
        for a, b, c in triangles(self.d):
            cross = (b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0])
            volume += sum(a[i]*cross[i] for i in range(3))/6
        self.assertGreater(volume, 0)

    def test_invalid_dimensions(self):
        for d in [Duct(10, 10, 10, 10, 0, 1), Duct(10, 10, 10, 10, 5, 5), Duct(math.nan, 10, 10, 10, 5, 1)]:
            with self.assertRaises(ValueError):
                triangles(d)

    def test_deterministic_geometry_and_metrics(self):
        self.assertEqual(stl_text(self.d), stl_text(self.d))
        self.assertEqual(len(triangles(self.d)), 32)
        self.assertEqual(metrics(self.d)["inlet_open_area_mm2"], 96*76)
        self.assertEqual(self.d.fingerprint(), Duct(100, 80, 120, 50, 100, 2).fingerprint())

    def test_cli_outputs_and_rejects_bad_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidate"
            result = subprocess.run([sys.executable, "-m", "forge", "examples/duct.json", "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((out / "report.json").read_text())["evaluation_type"], "geometry_only")
            self.assertIn("endsolid forge_duct", (out / "duct.stl").read_text())
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"wall": -1}')
            result = subprocess.run([sys.executable, "-m", "forge", str(bad), "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
