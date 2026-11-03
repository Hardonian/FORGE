"""Bayesian and empirical calibration for Sim2Real reality gap closure."""
from __future__ import annotations
import math
from typing import List
from pydantic import BaseModel
from forge.core.models import DuctSpec, PhysicsConditions
from forge.lab.telemetry import TestBenchTelemetry
from forge.physics.solver_1d import solve_duct_physics


class CalibrationResult(BaseModel):
    """Results of Sim2Real parameter calibration against physical test bench runs."""
    calibrated_roughness_mm: float
    calibrated_loss_multiplier: float
    baseline_rmse_pa: float
    calibrated_rmse_pa: float
    improvement_percent: float


def _evaluate_rmse(
    spec: DuctSpec,
    bench_runs: List[TestBenchTelemetry],
    roughness_mm: float,
    loss_multiplier: float = 1.0,
) -> float:
    """Compute root-mean-square error (Pa) between simulation and physical bench runs."""
    if not bench_runs:
        return 0.0
    sq_err_sum = 0.0
    for run in bench_runs:
        conds = PhysicsConditions(
            volumetric_flow_cfm=run.measured_flow_cfm,
            air_temp_c=run.ambient_temp_c,
            surface_roughness_mm=roughness_mm,
            minor_loss_multiplier=loss_multiplier,
        )
        sim = solve_duct_physics(spec, conds)
        diff = sim.total_pressure_drop_pa - abs(run.measured_delta_p_pa)
        sq_err_sum += diff * diff
    return math.sqrt(sq_err_sum / len(bench_runs))


def calibrate_surface_roughness(spec: DuctSpec, bench_runs: List[TestBenchTelemetry]) -> float:
    """Find the optimal equivalent surface roughness (in mm) that minimizes prediction error against bench runs."""
    if not bench_runs:
        return 0.05  # Default FDM roughness in mm

    # Golden-section search on [0.002, 2.0] mm
    a, b = 0.002, 2.0
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = _evaluate_rmse(spec, bench_runs, x1)
    f2 = _evaluate_rmse(spec, bench_runs, x2)

    for _ in range(32):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = _evaluate_rmse(spec, bench_runs, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = _evaluate_rmse(spec, bench_runs, x2)

    return round(0.5 * (a + b), 4)


def calibrate_sim2real_model(spec: DuctSpec, bench_runs: List[TestBenchTelemetry]) -> CalibrationResult:
    """Jointly calibrate surface roughness (mm) and minor loss multiplier (K_mult) to close the Sim2Real gap."""
    if not bench_runs:
        return CalibrationResult(
            calibrated_roughness_mm=0.05,
            calibrated_loss_multiplier=1.0,
            baseline_rmse_pa=0.0,
            calibrated_rmse_pa=0.0,
            improvement_percent=0.0,
        )

    baseline_rmse = _evaluate_rmse(spec, bench_runs, roughness_mm=0.05, loss_multiplier=1.0)
    best_r = 0.05
    best_k = 1.0
    best_rmse = baseline_rmse

    # Coordinate descent alternating between roughness and minor loss multiplier
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    for _ in range(4):
        # Optimize roughness_mm on [0.002, 2.0] holding best_k fixed
        a, b = 0.002, 2.0
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1 = _evaluate_rmse(spec, bench_runs, x1, best_k)
        f2 = _evaluate_rmse(spec, bench_runs, x2, best_k)
        for _ in range(20):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = a + resphi * (b - a)
                f1 = _evaluate_rmse(spec, bench_runs, x1, best_k)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = b - resphi * (b - a)
                f2 = _evaluate_rmse(spec, bench_runs, x2, best_k)
        best_r = 0.5 * (a + b)

        # Optimize loss_multiplier on [0.1, 5.0] holding best_r fixed
        ka, kb = 0.1, 5.0
        kx1 = ka + resphi * (kb - ka)
        kx2 = kb - resphi * (kb - ka)
        kf1 = _evaluate_rmse(spec, bench_runs, best_r, kx1)
        kf2 = _evaluate_rmse(spec, bench_runs, best_r, kx2)
        for _ in range(20):
            if kf1 < kf2:
                kb, kx2, kf2 = kx2, kx1, kf1
                kx1 = ka + resphi * (kb - ka)
                kf1 = _evaluate_rmse(spec, bench_runs, best_r, kx1)
            else:
                ka, kx1, kf1 = kx1, kx2, kf2
                kx2 = kb - resphi * (kb - ka)
                kf2 = _evaluate_rmse(spec, bench_runs, best_r, kx2)
        best_k = 0.5 * (ka + kb)
        best_rmse = _evaluate_rmse(spec, bench_runs, best_r, best_k)

    improvement = ((baseline_rmse - best_rmse) / max(1e-6, baseline_rmse)) * 100.0 if baseline_rmse > 0 else 0.0
    return CalibrationResult(
        calibrated_roughness_mm=round(best_r, 4),
        calibrated_loss_multiplier=round(best_k, 4),
        baseline_rmse_pa=round(baseline_rmse, 3),
        calibrated_rmse_pa=round(best_rmse, 3),
        improvement_percent=round(max(0.0, improvement), 2),
    )
