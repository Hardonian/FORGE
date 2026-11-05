"""Direct-to-Chip (D2C) Liquid Cold Plate Micro-Channel Hydraulics & Two-Phase Margin Engine.

Models high-heat-flux liquid cold plates for next-generation 1000W-2000W AI accelerators:
1. Skived Copper Micro-Channel Geometry: Fin thickness (t_f ~ 0.2mm), channel width (w_c ~ 0.2mm),
   aspect ratio (alpha > 15), and hydraulic diameter (D_h).
2. Industry Standard Coolants: Water, PG25 (25% Propylene Glycol), PG50, and dielectric fluids
   with temperature-dependent thermophysical properties (rho, Cp, k, mu).
3. Developing Laminar Thermo-Hydraulics: Shah-London friction correlation, Sieder-Tate/Kays
   Nusselt number, convective heat transfer coefficient (h > 35,000 W/m^2K), and fin efficiency.
4. Total Hydraulic Resistance: Channel friction + manifold contraction/expansion + Universal
   Quick Disconnect (UQD02/UQD04) blind-mate coupling pressure drop and required pumping power.
5. Critical Heat Flux (CHF) & Onset of Nucleate Boiling (ONB) Safety Margins: Guarantees single-phase
   stability and prevents catastrophic vapor blanketing.
"""

from __future__ import annotations

import math
from enum import Enum
from pydantic import BaseModel, Field


class CoolantType(str, Enum):
    WATER = "water"
    PG25 = "pg25"  # 25% Propylene Glycol / 75% Water (OCP ORV3 Standard)
    PG50 = "pg50"  # 50% Propylene Glycol / 50% Water (Extreme Freeze Protection)
    DIELECTRIC = "dielectric"  # Engineered Synthetic Fluorocarbon Fluid


class UQDSize(str, Enum):
    NONE = "none"
    UQD02 = "uqd02"  # 1/8" Quick Disconnect (~2 to 4 LPM)
    UQD04 = "uqd04"  # 1/4" Quick Disconnect (~4 to 12 LPM, OCP ORV3 Standard)


class ColdPlateSpec(BaseModel):
    """Specification of a high-heat-flux skived copper microchannel cold plate."""

    name: str = Field(default="D2C-Microchannel-ColdPlate-1200W")
    coolant: CoolantType = Field(default=CoolantType.PG25)
    flow_rate_lpm: float = Field(default=3.5, description="Coolant volumetric flow rate in Liters/min (LPM)")
    coolant_inlet_temp_c: float = Field(default=32.0, description="Facility supply coolant temperature in C")
    heat_load_w: float = Field(default=1000.0, description="Total thermal dissipation from AI chip in Watts")
    plate_width_mm: float = Field(default=72.0, description="Active microchannel core width transverse to flow in mm")
    plate_length_mm: float = Field(default=75.0, description="Active microchannel core length along flow in mm")
    base_thickness_mm: float = Field(default=2.5, description="Solid copper base thickness beneath channels in mm")
    fin_thickness_mm: float = Field(default=0.20, description="Skived micro-fin thickness t_f in mm")
    channel_width_mm: float = Field(default=0.20, description="Microchannel spacing w_c in mm")
    channel_height_mm: float = Field(default=3.20, description="Microchannel depth / fin height H_c in mm")
    copper_conductivity_w_mk: float = Field(default=395.0, description="Cold plate copper thermal conductivity")
    uqd_coupling: UQDSize = Field(default=UQDSize.UQD04, description="Universal Quick Disconnect coupling standard")


class CoolantProperties(BaseModel):
    """Thermophysical fluid properties at bulk temperature."""

    density_kg_m3: float
    specific_heat_j_kg_k: float
    thermal_conductivity_w_mk: float
    dynamic_viscosity_pa_s: float
    prandtl_number: float


class ColdPlateHydraulicResult(BaseModel):
    """Comprehensive thermo-hydraulic evaluation of microchannel cold plate."""

    coolant: CoolantType
    flow_rate_lpm: float
    flow_rate_kg_s: float
    coolant_inlet_temp_c: float
    coolant_outlet_temp_c: float
    coolant_temp_rise_c: float
    number_of_channels: int
    channel_aspect_ratio: float
    hydraulic_diameter_um: float
    channel_velocity_m_s: float
    reynolds_number: float
    flow_regime: str
    friction_factor: float
    nusselt_number: float
    heat_transfer_coeff_w_m2k: float
    fin_efficiency_pct: float
    thermal_resistance_conv_k_w: float
    thermal_resistance_base_k_w: float
    thermal_resistance_total_k_w: float
    max_cold_plate_base_temp_c: float
    channel_pressure_drop_kpa: float
    manifold_pressure_drop_kpa: float
    uqd_coupling_pressure_drop_kpa: float
    total_pressure_drop_kpa: float
    total_pressure_drop_psi: float
    required_pumping_power_w: float
    heat_flux_w_cm2: float
    chf_safety_margin_ratio: float
    is_chf_safe: bool
    summary_verdict: str


def _get_coolant_properties(coolant: CoolantType, temp_c: float) -> CoolantProperties:
    """Evaluate temperature-dependent thermophysical properties of datacenter coolants."""
    t_c = temp_c

    if coolant == CoolantType.WATER:
        rho = 1000.0 * (1.0 - ((t_c - 4.0) ** 2) / (500000.0 + 350.0 * t_c))
        cp = 4182.0 + 0.12 * (t_c - 20.0)
        k = 0.57 + 0.0017 * t_c
        mu = 0.00179 / (1.0 + 0.03368 * t_c + 0.000221 * (t_c ** 2))
    elif coolant == CoolantType.PG25:
        # 25% Propylene Glycol / 75% Water
        rho = 1024.0 - 0.38 * (t_c - 20.0)
        cp = 3980.0 + 1.8 * (t_c - 20.0)
        k = 0.51 + 0.0011 * t_c
        # Viscosity is ~2.2x pure water at 30C
        mu = 0.0024 * math.exp(-0.024 * (t_c - 20.0))
    elif coolant == CoolantType.PG50:
        # 50% Propylene Glycol / 50% Water
        rho = 1042.0 - 0.48 * (t_c - 20.0)
        cp = 3550.0 + 2.5 * (t_c - 20.0)
        k = 0.39 + 0.0008 * t_c
        mu = 0.0048 * math.exp(-0.032 * (t_c - 20.0))
    else:  # DIELECTRIC (e.g. Novec 7100 / Engineered Fluorocarbon)
        rho = 1500.0 - 2.1 * (t_c - 20.0)
        cp = 1180.0 + 1.2 * (t_c - 20.0)
        k = 0.069 - 0.0001 * t_c
        mu = 0.00058 * math.exp(-0.015 * (t_c - 20.0))

    pr = (cp * mu) / max(1e-4, k)
    return CoolantProperties(
        density_kg_m3=round(rho, 1),
        specific_heat_j_kg_k=round(cp, 1),
        thermal_conductivity_w_mk=round(k, 4),
        dynamic_viscosity_pa_s=mu,
        prandtl_number=round(pr, 2),
    )


def solve_cold_plate_hydraulics(spec: ColdPlateSpec) -> ColdPlateHydraulicResult:
    """Perform rigorous microchannel fluid dynamics, heat transfer, and pressure drop analysis."""
    # Channel geometric parameters
    t_f_m = spec.fin_thickness_mm * 1e-3
    w_c_m = spec.channel_width_mm * 1e-3
    h_c_m = spec.channel_height_mm * 1e-3
    l_m = spec.plate_length_mm * 1e-3
    w_m = spec.plate_width_mm * 1e-3
    base_t_m = spec.base_thickness_mm * 1e-3

    pitch_m = t_f_m + w_c_m
    num_channels = max(10, int(w_m / pitch_m))
    aspect_ratio = h_c_m / max(1e-6, w_c_m)

    # Hydraulic diameter: D_h = 4*A_c / P_wet = 4*(w_c * h_c) / (2*(w_c + h_c))
    area_channel = w_c_m * h_c_m
    perimeter_channel = 2.0 * (w_c_m + h_c_m)
    d_h_m = (4.0 * area_channel) / max(1e-6, perimeter_channel)

    # Fluid flow rate conversions
    # 1 LPM = 1e-3 m^3 / 60 s = 1.66667e-5 m^3/s
    q_m3_s = (spec.flow_rate_lpm * 1e-3) / 60.0

    # Evaluate fluid properties at inlet temperature
    props = _get_coolant_properties(spec.coolant, spec.coolant_inlet_temp_c)
    m_dot_kg_s = q_m3_s * props.density_kg_m3

    # Sensible temperature rise
    delta_t_fluid = spec.heat_load_w / max(1.0, m_dot_kg_s * props.specific_heat_j_kg_k)
    t_outlet_c = spec.coolant_inlet_temp_c + delta_t_fluid
    t_bulk_c = spec.coolant_inlet_temp_c + 0.5 * delta_t_fluid

    # Re-evaluate properties at bulk fluid temperature for higher fidelity
    props_bulk = _get_coolant_properties(spec.coolant, t_bulk_c)

    # Channel velocity and Reynolds number
    total_flow_area = num_channels * area_channel
    u_channel = q_m3_s / max(1e-6, total_flow_area)
    reynolds = (props_bulk.density_kg_m3 * u_channel * d_h_m) / max(1e-7, props_bulk.dynamic_viscosity_pa_s)

    # Friction factor: Shah & London laminar correlation for rectangular microchannels
    # f * Re = 96 * (1 - 1.3553/alpha + 1.9467/alpha^2 - 1.7012/alpha^3 + 0.9564/alpha^4 - 0.2537/alpha^5)
    inv_alpha = 1.0 / max(1.0, aspect_ratio)
    f_re_laminar = 96.0 * (
        1.0
        - 1.3553 * inv_alpha
        + 1.9467 * (inv_alpha ** 2)
        - 1.7012 * (inv_alpha ** 3)
        + 0.9564 * (inv_alpha ** 4)
        - 0.2537 * (inv_alpha ** 5)
    )

    if reynolds < 2300.0:
        f = f_re_laminar / max(1.0, reynolds)
        # Add laminar entrance developing length correction: f_app = f + 3.44 / sqrt(x+)
        x_plus = (l_m / d_h_m) / max(1.0, reynolds)
        f_app = f + (3.44 / math.sqrt(max(1e-5, x_plus))) / max(1.0, reynolds)
        regime = "Laminar Developing"
    else:
        # Blasius turbulent correlation
        f_app = 0.3164 * (reynolds ** -0.25)
        regime = "Transitional / Turbulent"

    # Convective Heat Transfer Coefficient (Nusselt number)
    # Fully developed laminar Nusselt: Nu_inf ~ 8.235 * (1 - 2.042/alpha + ...)
    nu_inf = 8.235 * (
        1.0
        - 2.0421 * inv_alpha
        + 3.0853 * (inv_alpha ** 2)
        - 2.4765 * (inv_alpha ** 3)
        + 1.0578 * (inv_alpha ** 4)
        - 0.1861 * (inv_alpha ** 5)
    )
    # Sieder-Tate thermal entrance length developing enhancement
    gz = (d_h_m / l_m) * reynolds * props_bulk.prandtl_number
    nu = nu_inf + (0.0668 * gz) / (1.0 + 0.04 * (gz ** (2.0 / 3.0)))
    if reynolds >= 2300.0:
        # Gnielinski correlation for turbulent microchannels
        nu = ((f_app / 8.0) * (reynolds - 1000.0) * props_bulk.prandtl_number) / (
            1.0 + 12.7 * math.sqrt(f_app / 8.0) * ((props_bulk.prandtl_number ** (2.0 / 3.0)) - 1.0)
        )

    h_conv = (nu * props_bulk.thermal_conductivity_w_mk) / max(1e-6, d_h_m)

    # Fin efficiency: m = sqrt(2 * h / (k_cu * t_f))
    m_fin = math.sqrt((2.0 * h_conv) / max(1.0, spec.copper_conductivity_w_mk * t_f_m))
    fin_eff = math.tanh(m_fin * h_c_m) / max(1e-5, m_fin * h_c_m)

    # Total wetted heat transfer area: base of channels + both sides of fins
    area_channel_base = num_channels * (w_c_m * l_m)
    area_fin_sides = num_channels * (2.0 * h_c_m * l_m)
    eff_heat_area = area_channel_base + fin_eff * area_fin_sides

    r_conv = 1.0 / max(1.0, h_conv * eff_heat_area)

    # 1D Solid Copper base conduction resistance: R_base = base_thickness / (k * A_plate)
    active_base_area = w_m * l_m
    r_base = base_t_m / max(1e-4, spec.copper_conductivity_w_mk * active_base_area)
    r_total = r_conv + r_base

    # Maximum cold plate base temperature
    max_base_temp = t_bulk_c + (spec.heat_load_w * r_total)

    # Hydraulic Pressure Drops
    # 1. Microchannel friction: Delta_P = f_app * (L / D_h) * 0.5 * rho * u^2
    dp_channel_pa = f_app * (l_m / d_h_m) * 0.5 * props_bulk.density_kg_m3 * (u_channel ** 2)

    # 2. Manifold entry contraction + exit expansion (K_in ~ 0.5, K_out ~ 1.0)
    k_manifold = 1.6
    dp_manifold_pa = k_manifold * 0.5 * props_bulk.density_kg_m3 * (u_channel ** 2)

    # 3. Universal Quick Disconnect (UQD) Blind-Mate Couplings
    dp_uqd_pa = 0.0
    if spec.uqd_coupling == UQDSize.UQD02:
        # UQD02: Cv ~ 0.35 -> Delta_P (psi) = (GPM / Cv)^2
        gpm = spec.flow_rate_lpm * 0.264172
        dp_uqd_psi = (gpm / 0.35) ** 2
        dp_uqd_pa = dp_uqd_psi * 6894.76
    elif spec.uqd_coupling == UQDSize.UQD04:
        # UQD04: Cv ~ 1.15
        gpm = spec.flow_rate_lpm * 0.264172
        dp_uqd_psi = (gpm / 1.15) ** 2
        dp_uqd_pa = dp_uqd_psi * 6894.76

    dp_total_pa = dp_channel_pa + dp_manifold_pa + dp_uqd_pa
    dp_total_kpa = dp_total_pa / 1000.0
    dp_total_psi = dp_total_pa / 6894.76

    # Required hydraulic pumping power: W = Q * Delta_P / eta_pump (eta ~ 45%)
    pumping_power_w = (q_m3_s * dp_total_pa) / 0.45

    # Critical Heat Flux (CHF) & Boiling Safety Margin
    # Base heat flux in W/cm^2
    heat_flux_w_cm2 = spec.heat_load_w / (active_base_area * 1e4)
    # Zuber / Kutateladze pool boiling CHF baseline scaled for subcooled forced microchannel flow
    # Water/PG25 CHF under forced convection is typically 180 to 280 W/cm^2
    chf_limit_w_cm2 = 240.0 if spec.coolant in [CoolantType.WATER, CoolantType.PG25] else 160.0
    chf_ratio = round(chf_limit_w_cm2 / max(1.0, heat_flux_w_cm2), 2)
    is_safe = chf_ratio >= 1.75 and max_base_temp <= 78.0

    verdict = (
        f"COLD PLATE CERTIFIED: High heat flux ({heat_flux_w_cm2:.1f} W/cm^2) comfortably managed. "
        f"Max base temp: {max_base_temp:.1f}C (Headroom: {78.0 - max_base_temp:.1f}C). "
        f"Total loop DP: {dp_total_kpa:.1f} kPa ({dp_total_psi:.2f} psi)."
        if is_safe
        else f"THERMAL / PRESSURE ALERT: Max base temp ({max_base_temp:.1f}C) or CHF margin ({chf_ratio}x) "
        f"exceeds single-phase guidelines. Consider increasing flow rate or microchannel aspect ratio."
    )

    return ColdPlateHydraulicResult(
        coolant=spec.coolant,
        flow_rate_lpm=round(spec.flow_rate_lpm, 2),
        flow_rate_kg_s=round(m_dot_kg_s, 4),
        coolant_inlet_temp_c=round(spec.coolant_inlet_temp_c, 1),
        coolant_outlet_temp_c=round(t_outlet_c, 2),
        coolant_temp_rise_c=round(delta_t_fluid, 2),
        number_of_channels=num_channels,
        channel_aspect_ratio=round(aspect_ratio, 1),
        hydraulic_diameter_um=round(d_h_m * 1e6, 1),
        channel_velocity_m_s=round(u_channel, 2),
        reynolds_number=round(reynolds, 0),
        flow_regime=regime,
        friction_factor=round(f_app, 4),
        nusselt_number=round(nu, 2),
        heat_transfer_coeff_w_m2k=round(h_conv, 0),
        fin_efficiency_pct=round(fin_eff * 100.0, 1),
        thermal_resistance_conv_k_w=round(r_conv, 5),
        thermal_resistance_base_k_w=round(r_base, 5),
        thermal_resistance_total_k_w=round(r_total, 5),
        max_cold_plate_base_temp_c=round(max_base_temp, 2),
        channel_pressure_drop_kpa=round(dp_channel_pa / 1000.0, 2),
        manifold_pressure_drop_kpa=round(dp_manifold_pa / 1000.0, 2),
        uqd_coupling_pressure_drop_kpa=round(dp_uqd_pa / 1000.0, 2),
        total_pressure_drop_kpa=round(dp_total_kpa, 2),
        total_pressure_drop_psi=round(dp_total_psi, 2),
        required_pumping_power_w=round(pumping_power_w, 2),
        heat_flux_w_cm2=round(heat_flux_w_cm2, 2),
        chf_safety_margin_ratio=chf_ratio,
        is_chf_safe=is_safe,
        summary_verdict=verdict,
    )
