"""Ingestion and discrepancy validation for physical test bench runs."""
from __future__ import annotations
import json
from pathlib import Path
from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.solver_1d import solve_duct_physics
from forge.lab.telemetry import TestBenchTelemetry, ValidationComparison


def compare_bench_to_simulation(spec: DuctSpec, bench_data: TestBenchTelemetry) -> ValidationComparison:
    """Compare physical wind tunnel / flow bench measurements against model predictions."""
    conditions = PhysicsConditions(
        volumetric_flow_cfm=bench_data.measured_flow_cfm,
        air_temp_c=bench_data.ambient_temp_c,
    )
    sim_result = solve_duct_physics(spec, conditions)

    sim_dp = sim_result.total_pressure_drop_pa
    meas_dp = abs(bench_data.measured_delta_p_pa)

    error_pct = (abs(sim_dp - meas_dp) / max(0.1, meas_dp)) * 100.0

    if error_pct <= 10.0:
        status = "MATCH"
    elif error_pct <= 25.0:
        status = "DEVIATION_DETECTED"
    else:
        status = "CALIBRATION_RECOMMENDED"

    return ValidationComparison(
        design_sha256=bench_data.design_sha256,
        sim_delta_p_pa=round(sim_dp, 2),
        measured_delta_p_pa=round(meas_dp, 2),
        delta_p_discrepancy_percent=round(error_pct, 2),
        sim_outlet_temp_c=round(conditions.air_temp_c + sim_result.estimated_temp_rise_c, 2),
        measured_outlet_temp_c=bench_data.thermocouple_temps_c.get("outlet"),
        status=status,
    )


def ingest_telemetry_file(filepath: Path) -> TestBenchTelemetry:
    """Ingest a physical run JSON file."""
    content = Path(filepath).read_text(encoding="utf-8")
    data = json.loads(content)
    return TestBenchTelemetry(**data)
