"""Bayesian and empirical calibration for Sim2Real reality gap closure."""
from __future__ import annotations
import math
from typing import List
from forge.core.models import DuctSpec, PhysicsConditions
from forge.lab.telemetry import TestBenchTelemetry
from forge.physics.solver_1d import solve_duct_physics


def calibrate_surface_roughness(spec: DuctSpec, bench_runs: List[TestBenchTelemetry]) -> float:
    """Find the optimal equivalent surface roughness (in mm) that minimizes prediction error against bench runs."""
    if not bench_runs:
        return 0.05  # Default FDM roughness in mm

    best_roughness = 0.05
    min_rmse = float("inf")

    # Grid search across physical roughness range 0.01mm (smooth SLA/SLS) to 0.3mm (coarse FDM)
    roughness_candidates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]

    for r_cand in roughness_candidates:
        sq_err_sum = 0.0
        for run in bench_runs:
            conds = PhysicsConditions(
                volumetric_flow_cfm=run.measured_flow_cfm,
                air_temp_c=run.ambient_temp_c,
            )
            sim = solve_duct_physics(spec, conds)
            diff = sim.total_pressure_drop_pa - run.measured_delta_p_pa
            sq_err_sum += diff * diff

        rmse = math.sqrt(sq_err_sum / len(bench_runs))
        if rmse < min_rmse:
            min_rmse = rmse
            best_roughness = r_cand

    return best_roughness
