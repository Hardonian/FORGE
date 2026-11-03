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

    def test_evaluate_endpoint(self):
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

    def test_export_stl_endpoint(self):
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

    def test_viewer_endpoint(self):
        resp = self.client.get("/viewer")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("FORGE 3D Studio", resp.text)


if __name__ == "__main__":
    unittest.main()

