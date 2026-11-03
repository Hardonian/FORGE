"""Unit tests for physical test bench telemetry ingestion and Sim2Real calibration."""
import tempfile
from pathlib import Path
import unittest
from forge.core.models import DuctSpec
from forge.lab.telemetry import TestBenchTelemetry
from forge.lab.ingestion import compare_bench_to_simulation, ingest_telemetry_file
from forge.lab.calibration import calibrate_surface_roughness, calibrate_sim2real_model


class LabTests(unittest.TestCase):
    def setUp(self):
        self.spec = DuctSpec(
            inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50,
            length=100, wall=2.0
        )
        self.telemetry = TestBenchTelemetry(
            session_id="bench_run_001",
            design_sha256="abc123456789",
            timestamp="2026-11-02T20:00:00Z",
            fan_pwm_percent=80.0,
            fan_rpm=4200.0,
            ambient_temp_c=24.5,
            measured_flow_cfm=120.0,
            inlet_static_pressure_pa=15.0,
            outlet_static_pressure_pa=55.0,
            measured_delta_p_pa=7.5,
            thermocouple_temps_c={"inlet": 24.5, "outlet": 31.2},
        )

    def test_compare_bench_to_simulation(self):
        comp = compare_bench_to_simulation(self.spec, self.telemetry)
        self.assertGreater(comp.sim_delta_p_pa, 0.0)
        self.assertEqual(comp.measured_delta_p_pa, 7.5)
        self.assertIn(comp.status, ["MATCH", "DEVIATION_DETECTED", "CALIBRATION_RECOMMENDED"])

    def test_ingest_telemetry_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "telemetry.json"
            fpath.write_text(self.telemetry.model_dump_json(), encoding="utf-8")
            loaded = ingest_telemetry_file(fpath)
            self.assertEqual(loaded.session_id, "bench_run_001")

    def test_surface_roughness_and_joint_sim2real_calibration(self):
        low_dp_run = self.telemetry.model_copy(update={"measured_delta_p_pa": 5.2})
        high_dp_run = self.telemetry.model_copy(update={"measured_delta_p_pa": 9.5})
        r_low = calibrate_surface_roughness(self.spec, [low_dp_run])
        r_high = calibrate_surface_roughness(self.spec, [high_dp_run])
        # Higher measured pressure drop calibrates to a larger effective surface roughness
        self.assertGreater(r_high, r_low)

        cal_res = calibrate_sim2real_model(self.spec, [high_dp_run])
        self.assertLessEqual(cal_res.calibrated_rmse_pa, cal_res.baseline_rmse_pa)


if __name__ == "__main__":
    unittest.main()
