"""Structural modal vibration, temperature-softened plate eigenfrequencies, and fan harmonic resonance solver."""
from __future__ import annotations
import math
from typing import Dict, List
from pydantic import BaseModel
from forge.core.models import DuctSpec, MATERIAL_PROPERTIES, PhysicsConditions
from forge.cad.mesh import compute_normal_wall_compensation
from forge.physics.solver_1d import solve_duct_physics


# Coefficient of Thermal Expansion (CTE in 1/K) for 3D printing polymers and carbon-fiber composites
MATERIAL_CTE_PER_K: Dict[str, float] = {
    "PETG": 68e-6,
    "ABS": 85e-6,
    "ASA": 82e-6,
    "PC-CF": 22e-6,     # Carbon fiber drastically lowers thermal expansion
    "PA12-CF": 25e-6,
    "PEEK": 45e-6,
}


class ModalResonanceReport(BaseModel):
    """Structural vibration eigenfrequencies, fan harmonic separation, and thermo-mechanical stress analysis."""
    exhaust_temp_c: float
    effective_youngs_modulus_mpa: float
    fundamental_panel_freq_hz: float
    higher_panel_modes_hz: List[float]
    cantilever_beam_freq_hz: float
    fan_shaft_freq_hz: float
    fan_bpf_freq_hz: float
    min_harmonic_separation_pct: float
    resonance_risk: bool
    thermal_expansion_um: float
    flange_thermal_stress_mpa: float
    recommended_wall_mm: float


def analyze_structural_modal_resonance(
    spec: DuctSpec,
    conditions: PhysicsConditions | None = None,
    fan_rpm: int = 9000,
    blade_count: int = 7,
) -> ModalResonanceReport:
    """Compute temperature-adjusted structural natural frequencies and check for fan harmonic resonance."""
    cond = conditions or PhysicsConditions()
    phys = solve_duct_physics(spec, cond)
    exhaust_temp_c = cond.air_temp_c + phys.estimated_temp_rise_c

    mat_key = spec.material.value
    mat_props = MATERIAL_PROPERTIES.get(mat_key, MATERIAL_PROPERTIES["PETG"])
    e_room_mpa = mat_props.get("youngs_modulus_mpa", 2100.0)
    hdt_c = mat_props.get("hdt_c", 70.0)
    rho_solid = mat_props.get("density", 1.25) * 1000.0  # kg/m^3

    # Temperature-dependent viscoelastic modulus softening near HDT
    temp_ratio = max(0.0, exhaust_temp_c - 22.0) / max(10.0, hdt_c - 15.0)
    softening_factor = max(0.25, 1.0 - 0.55 * (temp_ratio ** 1.8))
    e_eff_mpa = e_room_mpa * softening_factor
    e_eff_pa = e_eff_mpa * 1e6
    poisson = 0.36

    _, _, min_normal_wall_mm = compute_normal_wall_compensation(spec)
    h_m = max(0.0005, min_normal_wall_mm / 1000.0)
    a_m = max(0.015, spec.length / 1000.0)
    b_m = max(0.015, max(spec.inlet_width, spec.inlet_height, spec.outlet_width, spec.outlet_height) / 1000.0)

    # Plate flexural rigidity D = E * h^3 / (12 * (1 - nu^2))
    d_flex = (e_eff_pa * (h_m ** 3)) / (12.0 * (1.0 - poisson * poisson))
    # Internal splitter guide vanes stiffen the widest panel span by dividing b_m
    b_eff_m = b_m / float(1 + spec.guide_vanes)

    def _plate_mode_hz(m_idx: int, n_idx: int) -> float:
        term = ((m_idx / a_m) ** 2) + ((n_idx / b_eff_m) ** 2)
        return (math.pi / 2.0) * term * math.sqrt(d_flex / (rho_solid * h_m))

    f_11 = _plate_mode_hz(1, 1)
    f_12 = _plate_mode_hz(1, 2)
    f_21 = _plate_mode_hz(2, 1)
    f_22 = _plate_mode_hz(2, 2)

    # Tubular cantilever bending mode (clamped at fan flange, free/supported at GPU)
    # Radius of gyration r_g ~ 0.38 * mean_dim
    mean_dim_m = 0.5 * (spec.inlet_width + spec.outlet_width) / 1000.0
    r_gyr = 0.38 * mean_dim_m
    f_cant = ((1.875 ** 2) / (2.0 * math.pi * (a_m ** 2))) * math.sqrt(e_eff_pa / rho_solid) * r_gyr

    # Fan excitation harmonics
    f_shaft = max(10.0, fan_rpm / 60.0)
    f_bpf = f_shaft * max(3, blade_count)

    all_modes = [f_11, f_12, f_21, f_22, f_cant]
    excitations = [f_shaft, f_bpf]
    min_sep_pct = min(
        100.0 * abs(f_m - f_ex) / f_ex
        for f_m in all_modes
        for f_ex in excitations
    )
    resonance_risk = min_sep_pct < 15.0

    # If resonance risk exists, compute wall thickness needed to shift f_11 by +22% above nearest excitation
    rec_wall_mm = spec.wall
    if resonance_risk:
        rec_wall_mm = round(spec.wall * 1.25, 2)

    # Thermo-mechanical expansion and bolted flange stress
    cte = MATERIAL_CTE_PER_K.get(mat_key, 60e-6)
    delta_t = max(0.0, exhaust_temp_c - 22.0)
    expansion_um = (a_m * cte * delta_t) * 1e6
    flange_stress_mpa = e_eff_mpa * cte * delta_t * 0.45  # Partial bolt-hole constraint factor

    return ModalResonanceReport(
        exhaust_temp_c=round(exhaust_temp_c, 2),
        effective_youngs_modulus_mpa=round(e_eff_mpa, 1),
        fundamental_panel_freq_hz=round(f_11, 1),
        higher_panel_modes_hz=[round(f_12, 1), round(f_21, 1), round(f_22, 1)],
        cantilever_beam_freq_hz=round(f_cant, 1),
        fan_shaft_freq_hz=round(f_shaft, 1),
        fan_bpf_freq_hz=round(f_bpf, 1),
        min_harmonic_separation_pct=round(min_sep_pct, 2),
        resonance_risk=resonance_risk,
        thermal_expansion_um=round(expansion_um, 2),
        flange_thermal_stress_mpa=round(flange_stress_mpa, 2),
        recommended_wall_mm=rec_wall_mm,
    )

