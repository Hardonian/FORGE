"""Heterogeneous Multi-Die CoWoS Package & Multi-Zone Conjugate Thermal Network.

Models advanced 2.5D/3D AI packaging (NVIDIA Blackwell/Hopper, AMD MI300X, Google TPU v6):
1. Heterogeneous Die Subsystems: Central GPU Logic Die, High-Bandwidth Memory (HBM3e/HBM4)
   stacks, Voltage Regulator Modules (VRM/DrMOS power stages), and high-speed retimers.
2. Micro-Zone Thermal Resistance Network: Die junction-to-case (R_jc), TIM1/TIM2 contact
   resistances, and vapor chamber lateral spreading resistance.
3. Air Preheating Cascade: Upstream component heat dissipation increases the local air
   inlet temperature for downstream zones: T_in,eff = T_ambient + Delta_T_preheat.
4. Per-Component Thermal Throttling & Margin Assessment: Independent junction temperature
   monitoring against strict HBM (85C), GPU (105C), and VRM (115C) reliability ceilings.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class DieType(str, Enum):
    GPU_LOGIC = "gpu_logic"
    HBM3E_STACK = "hbm3e_stack"
    VRM_POWER_STAGE = "vrm_power_stage"
    RETIMER_OPTICAL = "retimer_optical"


class ComponentDieSpec(BaseModel):
    """Specification of an individual silicon die or power stage within the package."""

    name: str = Field(description="Component identifier (e.g. GPU-Compute-Core, HBM3e-Stack-0)")
    die_type: DieType = Field(default=DieType.GPU_LOGIC)
    power_w: float = Field(default=750.0, description="Thermal design power dissipation in Watts")
    width_mm: float = Field(default=32.0, description="Die physical width along airflow X in mm")
    length_mm: float = Field(default=26.0, description="Die physical length transverse Y in mm")
    streamwise_position_mm: float = Field(
        default=50.0,
        description="Streamwise coordinate from package airflow intake edge in mm",
    )
    r_jc_k_w: float = Field(
        default=0.032,
        description="Internal silicon and interposer junction-to-case thermal resistance in K/W",
    )
    max_junction_temp_c: float = Field(
        default=105.0,
        description="Maximum permissible junction temperature before thermal throttling in C",
    )


class TIMLayerSpec(BaseModel):
    """Thermal Interface Material (TIM1 / TIM2) specification."""

    name: str = Field(default="Indium-Solder-TIM1")
    thermal_conductivity_w_mk: float = Field(default=86.0, description="Bulk thermal conductivity in W/mK")
    bond_line_thickness_um: float = Field(default=35.0, description="Bond line thickness in micrometers")
    contact_resistance_k_cm2_w: float = Field(
        default=0.045,
        description="Interfacial micro-asperity contact resistance in K*cm^2/W",
    )


class CoWoSPackageSpec(BaseModel):
    """Heterogeneous CoWoS AI Accelerator multi-die package assembly."""

    package_name: str = Field(default="AI-Accelerator-SXM5-1000W", description="Package identifier")
    substrate_width_mm: float = Field(default=102.0, description="Total substrate package width in mm")
    substrate_length_mm: float = Field(default=140.0, description="Total substrate package length in mm")
    vapor_chamber_thickness_mm: float = Field(default=3.5, description="Copper vapor chamber base thickness in mm")
    vapor_chamber_eff_conductivity_w_mk: float = Field(
        default=1800.0,
        description="Effective planar thermal conductivity of copper vapor chamber in W/mK",
    )
    tim1: TIMLayerSpec = Field(
        default_factory=lambda: TIMLayerSpec(
            name="Indium-Solder-TIM1",
            thermal_conductivity_w_mk=86.0,
            bond_line_thickness_um=35.0,
            contact_resistance_k_cm2_w=0.04,
        )
    )
    tim2: TIMLayerSpec = Field(
        default_factory=lambda: TIMLayerSpec(
            name="Phase-Change-TIM2",
            thermal_conductivity_w_mk=8.5,
            bond_line_thickness_um=50.0,
            contact_resistance_k_cm2_w=0.08,
        )
    )
    components: List[ComponentDieSpec] = Field(
        default_factory=lambda: [
            ComponentDieSpec(
                name="GPU-Compute-Die",
                die_type=DieType.GPU_LOGIC,
                power_w=720.0,
                width_mm=32.0,
                length_mm=26.0,
                streamwise_position_mm=45.0,
                r_jc_k_w=0.028,
                max_junction_temp_c=105.0,
            ),
            ComponentDieSpec(
                name="HBM3e-North-Left",
                die_type=DieType.HBM3E_STACK,
                power_w=42.0,
                width_mm=11.0,
                length_mm=11.0,
                streamwise_position_mm=25.0,
                r_jc_k_w=0.18,
                max_junction_temp_c=85.0,
            ),
            ComponentDieSpec(
                name="HBM3e-North-Right",
                die_type=DieType.HBM3E_STACK,
                power_w=42.0,
                width_mm=11.0,
                length_mm=11.0,
                streamwise_position_mm=25.0,
                r_jc_k_w=0.18,
                max_junction_temp_c=85.0,
            ),
            ComponentDieSpec(
                name="HBM3e-South-Left",
                die_type=DieType.HBM3E_STACK,
                power_w=42.0,
                width_mm=11.0,
                length_mm=11.0,
                streamwise_position_mm=75.0,
                r_jc_k_w=0.18,
                max_junction_temp_c=85.0,
            ),
            ComponentDieSpec(
                name="HBM3e-South-Right",
                die_type=DieType.HBM3E_STACK,
                power_w=42.0,
                width_mm=11.0,
                length_mm=11.0,
                streamwise_position_mm=75.0,
                r_jc_k_w=0.18,
                max_junction_temp_c=85.0,
            ),
            ComponentDieSpec(
                name="VRM-Power-Stage-Bank",
                die_type=DieType.VRM_POWER_STAGE,
                power_w=95.0,
                width_mm=14.0,
                length_mm=75.0,
                streamwise_position_mm=10.0,
                r_jc_k_w=0.09,
                max_junction_temp_c=115.0,
            ),
        ]
    )


class ComponentThermalAssessment(BaseModel):
    """Detailed thermal status of an individual die component."""

    name: str
    die_type: DieType
    power_w: float
    local_air_inlet_temp_c: float
    preheat_delta_t_c: float
    r_jc_k_w: float
    r_tim1_k_w: float
    r_tim2_k_w: float
    r_spreading_k_w: float
    r_convection_k_w: float
    total_r_theta_k_w: float
    junction_temp_c: float
    max_junction_temp_c: float
    thermal_margin_c: float
    is_throttling: bool


class MultiZoneCoWoSReport(BaseModel):
    """Complete multi-die thermal network and bottleneck analysis report."""

    package_name: str
    total_package_power_w: float
    cooling_airflow_cfm: float
    ambient_air_temp_c: float
    bulk_air_exit_temp_c: float
    air_temp_rise_c: float
    components: List[ComponentThermalAssessment]
    critical_bottleneck_component: str
    critical_thermal_margin_c: float
    is_package_throttled: bool
    recommended_min_airflow_cfm: float
    summary_verdict: str


def _calculate_tim_resistance(tim: TIMLayerSpec, area_m2: float) -> float:
    """Compute 1D thermal resistance of a TIM layer in K/W: R = (BLT / k) / Area + R_contact / Area."""
    blt_m = tim.bond_line_thickness_um * 1e-6
    area_cm2 = area_m2 * 1e4
    r_bulk = blt_m / max(1e-4, tim.thermal_conductivity_w_mk * area_m2)
    r_contact = (tim.contact_resistance_k_cm2_w / max(1e-4, area_cm2))
    return r_bulk + r_contact


def _calculate_vapor_chamber_spreading(
    die_w_m: float,
    die_l_m: float,
    base_w_m: float,
    base_l_m: float,
    base_thick_m: float,
    k_eff: float,
) -> float:
    """Compute 3D lateral spreading resistance in vapor chamber base using analytical Biot/Fourier model.

    R_spread ~ (1 - eps) / (k * sqrt(pi) * a) where eps = sqrt(A_die / A_base)
    """
    a_die = die_w_m * die_l_m
    a_base = base_w_m * base_l_m
    eps = math.sqrt(a_die / max(1e-6, a_base))
    char_die_radius = math.sqrt(a_die / math.pi)

    # Kennedy / Lee spreading resistance function for thin vapor chambers
    spreading_factor = (1.0 - 1.410 * eps + 0.344 * (eps ** 3) + 0.043 * (eps ** 5) + 0.034 * (eps ** 7))
    r_spread = spreading_factor / max(1e-4, 2.0 * math.pi * k_eff * char_die_radius)
    # Add 1D through-plane conduction
    r_cond_1d = base_thick_m / max(1e-4, k_eff * a_die)
    return r_spread + r_cond_1d


def solve_cowos_multidie_thermal_network(
    package: CoWoSPackageSpec,
    airflow_cfm: float = 140.0,
    ambient_temp_c: float = 35.0,
    heatsink_base_area_factor: float = 24.0,  # Fin area / Base area ratio
    fin_heat_transfer_coeff_w_m2k: float = 85.0,  # Convective h in air
) -> MultiZoneCoWoSReport:
    """Solve the multi-zone CoWoS conjugate thermal network with upstream-to-downstream air preheating."""
    total_power = sum(c.power_w for c in package.components)

    # Bulk airflow and heat capacity
    q_m3_s = max(0.005, airflow_cfm * 0.000471947443)
    air_rho = 1.184  # kg/m^3 at ~35C
    air_cp = 1007.0  # J/kg*K
    air_heat_capacity_w_k = air_rho * q_m3_s * air_cp

    bulk_air_rise_c = total_power / max(1.0, air_heat_capacity_w_k)
    bulk_air_exit_c = ambient_temp_c + bulk_air_rise_c

    # Sort components by streamwise position from intake (x = 0) to exhaust
    sorted_components = sorted(package.components, key=lambda c: c.streamwise_position_mm)

    assessments: List[ComponentThermalAssessment] = []
    accumulated_upstream_heat = 0.0
    critical_component = ""
    min_margin = 999.0
    any_throttling = False

    base_w_m = package.substrate_width_mm * 1e-3
    base_l_m = package.substrate_length_mm * 1e-3
    base_thick_m = package.vapor_chamber_thickness_mm * 1e-3

    for comp in sorted_components:
        # Preheating cascade: air reaching this die has already absorbed heat from upstream components
        preheat_fraction = min(1.0, comp.streamwise_position_mm / max(1.0, package.substrate_length_mm))
        # Weighted upstream power: components strictly upstream shed heat into the boundary layer
        local_preheat_c = (accumulated_upstream_heat * preheat_fraction) / max(1.0, air_heat_capacity_w_k)
        local_air_inlet_c = ambient_temp_c + local_preheat_c

        die_w_m = comp.width_mm * 1e-3
        die_l_m = comp.length_mm * 1e-3
        die_area_m2 = die_w_m * die_l_m

        # Thermal resistances
        r_jc = comp.r_jc_k_w
        r_tim1 = _calculate_tim_resistance(package.tim1, die_area_m2)
        r_tim2 = _calculate_tim_resistance(package.tim2, die_area_m2)
        r_spread = _calculate_vapor_chamber_spreading(
            die_w_m,
            die_l_m,
            base_w_m,
            base_l_m,
            base_thick_m,
            package.vapor_chamber_eff_conductivity_w_mk,
        )

        # Local heatsink fin convective resistance apportioned by die area and lateral spreading footprint
        effective_fin_footprint_m2 = min(base_w_m * base_l_m, die_area_m2 * 2.8)
        wetted_fin_area_m2 = effective_fin_footprint_m2 * heatsink_base_area_factor
        fin_efficiency = 0.82
        r_conv = 1.0 / max(1e-4, fin_heat_transfer_coeff_w_m2k * wetted_fin_area_m2 * fin_efficiency)

        total_r_theta = r_jc + r_tim1 + r_tim2 + r_spread + r_conv
        junction_temp = local_air_inlet_c + (comp.power_w * total_r_theta)

        margin = comp.max_junction_temp_c - junction_temp
        is_throttled = margin < 0.0
        if is_throttled:
            any_throttling = True

        if margin < min_margin:
            min_margin = margin
            critical_component = comp.name

        assessments.append(
            ComponentThermalAssessment(
                name=comp.name,
                die_type=comp.die_type,
                power_w=round(comp.power_w, 1),
                local_air_inlet_temp_c=round(local_air_inlet_c, 2),
                preheat_delta_t_c=round(local_preheat_c, 2),
                r_jc_k_w=round(r_jc, 4),
                r_tim1_k_w=round(r_tim1, 4),
                r_tim2_k_w=round(r_tim2, 4),
                r_spreading_k_w=round(r_spread, 4),
                r_convection_k_w=round(r_conv, 4),
                total_r_theta_k_w=round(total_r_theta, 4),
                junction_temp_c=round(junction_temp, 2),
                max_junction_temp_c=round(comp.max_junction_temp_c, 1),
                thermal_margin_c=round(margin, 2),
                is_throttling=is_throttled,
            )
        )

        accumulated_upstream_heat += comp.power_w

    # Recommended minimum airflow to clear all margins with at least +5C headroom
    req_flow = airflow_cfm
    if min_margin < 5.0:
        deficit_c = 5.0 - min_margin
        # Airflow scales roughly inversely with temperature rise
        req_flow = round(airflow_cfm * (1.0 + deficit_c / max(5.0, bulk_air_rise_c)), 1)

    verdict = (
        f"PACKAGE SAFE: All {len(package.components)} dies within thermal limits. "
        f"Critical bottleneck: {critical_component} (Margin: {min_margin:.1f}C)."
        if not any_throttling
        else f"THERMAL THROTTLE DETECTED: {critical_component} exceeds limit by {abs(min_margin):.1f}C! "
        f"Minimum airflow must increase to {req_flow} CFM."
    )

    return MultiZoneCoWoSReport(
        package_name=package.package_name,
        total_package_power_w=round(total_power, 1),
        cooling_airflow_cfm=round(airflow_cfm, 1),
        ambient_air_temp_c=round(ambient_temp_c, 1),
        bulk_air_exit_temp_c=round(bulk_air_exit_c, 2),
        air_temp_rise_c=round(bulk_air_rise_c, 2),
        components=assessments,
        critical_bottleneck_component=critical_component,
        critical_thermal_margin_c=round(min_margin, 2),
        is_package_throttled=any_throttling,
        recommended_min_airflow_cfm=req_flow,
        summary_verdict=verdict,
    )
