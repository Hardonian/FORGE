"""Unit tests for FORGE FastAPI REST endpoints and 3D viewer."""
import unittest
from fastapi.testclient import TestClient
from forge.api.server import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_evaluate_and_mesh_endpoints(self):
        payload = {
            "spec": {
                "inlet_width": 100,
                "inlet_height": 80,
                "outlet_width": 120,
                "outlet_height": 50,
                "length": 100,
                "wall": 2.0
            }
        }
        resp = self.client.post("/api/v1/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("geometry_sha256", data)
        self.assertIn("physics", data)
        self.assertIn("dfm", data)

        mesh_resp = self.client.post("/api/v1/mesh", json=payload)
        self.assertEqual(mesh_resp.status_code, 200)
        m_data = mesh_resp.json()
        self.assertEqual(m_data["mesh"]["triangle_count"], 32)
        self.assertEqual(len(m_data["mesh"]["positions"]), 32 * 9)

    def test_optimize_endpoint(self):
        resp = self.client.post("/api/v1/optimize?population_size=6&generations=1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["pareto_count"], 0)

    def test_export_stl_and_glb_endpoints(self):
        payload = {
            "inlet_width": 100,
            "inlet_height": 80,
            "outlet_width": 120,
            "outlet_height": 50,
            "length": 100,
            "wall": 2.0
        }
        resp = self.client.post("/api/v1/export/stl?binary=true", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.content), 80)

        glb_resp = self.client.post("/api/v1/export/glb", json=payload)
        self.assertEqual(glb_resp.status_code, 200)
        self.assertEqual(glb_resp.content[:4], b"glTF")

    def test_viewer_endpoint(self):
        resp = self.client.get("/viewer")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("FORGE 3D Studio", resp.text)
        self.assertIn("/api/v1/mesh", resp.text)


if __name__ == "__main__":
    unittest.main()
