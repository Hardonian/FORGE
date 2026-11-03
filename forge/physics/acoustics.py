"""Aeroacoustic Blade-Passing-Frequency (BPF) analysis and Helmholtz / Quarter-Wave metamaterial liner design."""
from __future__ import annotations
import math
from typing import List
from pydantic import BaseModel, Field


class AcousticLinerDesign(BaseModel):
    """Tuned sub-wavelength Helmholtz / folded quarter-wave acoustic metamaterial liner specification."""
    fan_rpm: int = Field(..., gt=100)
    blade_count: int = Field(..., ge=3)
    blade_passing_freq_hz: float
    harmonics_hz: List[float]
    quarter_wave_length_mm: float
    helmholtz_neck_diameter_mm: float
    helmholtz_neck_length_mm: float
    helmholtz_cavity_depth_mm: float
    helmholtz_cavity_width_mm: float
    perforation_porosity_pct: float
    peak_transmission_loss_db: float
    effective_stopband_hz: List[float]


def design_acoustic_metamaterial_liner(
    fan_rpm: int = 8500,
    blade_count: int = 7,
    air_temp_c: float = 30.0,
    max_cavity_depth_mm: float = 10.0,
    neck_diameter_mm: float = 1.6,
    neck_length_mm: float = 1.5,
) -> AcousticLinerDesign:
    """Design a 3D-printable Helmholtz resonator / folded quarter-wave metamaterial liner tuned to the fan BPF.

    High-RPM server fans emit intense tonal noise at f_BPF = (RPM * N_blades) / 60.
    By embedding a sub-wavelength Helmholtz cavity array into the duct wall thickness,
    the liner creates a reactive impedance mismatch that cancels the primary blade-pass tone.
    """
    rpm = max(500, fan_rpm)
    blades = max(3, blade_count)
    f_bpf = (rpm * blades) / 60.0
    harmonics = [round(f_bpf * k, 1) for k in (1, 2, 3)]

    # Speed of sound in air: c = 331.3 * sqrt(1 + T_c / 273.15)
    c_sound = 331.3 * math.sqrt(1.0 + max(-20.0, air_temp_c) / 273.15)

    # Quarter-wave length L = c / (4 * f_BPF) in mm
    lambda_quarter_mm = (c_sound / (4.0 * f_bpf)) * 1000.0

    # Helmholtz resonator tuning:
    # f_H = (c / (2 * pi)) * sqrt(S_neck / (V_cavity * L_eff))
    # => V_cavity = S_neck * (c / (2 * pi * f_BPF))^2 / L_eff
    d_neck_m = neck_diameter_mm / 1000.0
    l_neck_m = neck_length_mm / 1000.0
    # End correction delta_L = 0.85 * d_neck
    l_eff_m = l_neck_m + 0.85 * d_neck_m
    # Multi-orifice micro-perforated faceplate (4 necks per resonator unit cell)
    orifices_per_cell = 4
    s_neck_m2 = orifices_per_cell * math.pi * ((0.5 * d_neck_m) ** 2)

    v_cavity_m3 = s_neck_m2 * ((c_sound / (2.0 * math.pi * f_bpf)) ** 2) / max(1e-6, l_eff_m)
    v_cavity_mm3 = v_cavity_m3 * 1e9

    depth_mm = max(2.0, max_cavity_depth_mm)
    # Square unit-cell cavity footprint width W_cav = sqrt(V_cavity / depth)
    cell_width_mm = math.sqrt(max(4.0, v_cavity_mm3 / depth_mm))

    porosity_pct = min(
        35.0,
        max(1.5, 100.0 * (orifices_per_cell * math.pi * ((0.5 * neck_diameter_mm) ** 2)) / (cell_width_mm ** 2)),
    )

    # Ingard / Maa viscous acoustic resistance of resonant orifices normalized by rho*c:
    # r_norm = (sqrt(8 * nu_air * omega) / (sigma_p * c)) * (1 + l_neck / d_neck)
    nu_air = 1.56e-5  # Kinematic viscosity of air (m^2/s)
    omega = 2.0 * math.pi * f_bpf
    sigma_frac = porosity_pct / 100.0
    r_norm = (math.sqrt(8.0 * nu_air * omega) / max(1e-5, sigma_frac * c_sound)) * (1.0 + l_neck_m / d_neck_m)
    # Grazing-incidence transmission loss across a lined duct section (S_liner / S_duct ~ 2.4):
    # TL = 20 * log10(1 + (S_liner / S_duct) / (2 * r_norm))
    liner_area_ratio = 2.4
    tl_db = min(26.0, max(6.0, 20.0 * math.log10(1.0 + liner_area_ratio / max(0.02, 2.0 * r_norm))))

    bandwidth_hz = f_bpf * 0.18
    stopband = [round(f_bpf - 0.5 * bandwidth_hz, 1), round(f_bpf + 0.5 * bandwidth_hz, 1)]

    return AcousticLinerDesign(
        fan_rpm=rpm,
        blade_count=blades,
        blade_passing_freq_hz=round(f_bpf, 1),
        harmonics_hz=harmonics,
        quarter_wave_length_mm=round(lambda_quarter_mm, 2),
        helmholtz_neck_diameter_mm=round(neck_diameter_mm, 2),
        helmholtz_neck_length_mm=round(neck_length_mm, 2),
        helmholtz_cavity_depth_mm=round(depth_mm, 2),
        helmholtz_cavity_width_mm=round(cell_width_mm, 2),
        perforation_porosity_pct=round(porosity_pct, 2),
        peak_transmission_loss_db=round(tl_db, 2),
        effective_stopband_hz=stopband,
    )
