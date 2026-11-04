"""Triply Periodic Minimal Surface (TPMS) Metamaterial Lattice Generator & Solver.

Synthesizes watertight 3D triangle meshes of self-supporting TPMS metamaterial
lattices (Gyroid, Schwarz-P, Diamond) with optional functional porosity grading,
and evaluates Darcy-Forchheimer permeability, specific surface area, and
convective heat transfer enhancement (Nusselt number & Watt dissipation).
"""

from __future__ import annotations

import math
from typing import List, Literal

from pydantic import BaseModel, Field

from forge.core.geometry import Triangle, Vec3

TPMSType = Literal["gyroid", "schwarz_p", "diamond"]


class TPMSSpec(BaseModel):
    """Specification for a 3D TPMS metamaterial lattice core."""

    lattice_type: TPMSType = Field(
        default="gyroid",
        description="TPMS implicit surface family ('gyroid', 'schwarz_p', 'diamond')",
    )
    width_mm: float = Field(default=60.0, gt=5.0, le=500.0, description="Core X width (mm)")
    height_mm: float = Field(default=40.0, gt=5.0, le=500.0, description="Core Y height (mm)")
    length_mm: float = Field(default=30.0, gt=5.0, le=500.0, description="Core Z flow length (mm)")
    unit_cell_mm: float = Field(
        default=8.0, gt=1.5, le=50.0, description="Spatial wavelength of one TPMS unit cell (mm)"
    )
    porosity_inlet: float = Field(
        default=0.72, ge=0.35, le=0.92, description="Void volume fraction at Z=0 inlet"
    )
    porosity_outlet: float = Field(
        default=0.65, ge=0.35, le=0.92, description="Void volume fraction at Z=length outlet"
    )
    wall_thickness_mm: float = Field(
        default=1.0, gt=0.3, le=5.0, description="Sheet thickness of the minimal surface (mm)"
    )
    thermal_conductivity_w_mk: float = Field(
        default=165.0, gt=0.1, description="Solid lattice thermal conductivity (W/m*K)"
    )


class TPMSThermoHydraulicReport(BaseModel):
    """Coupled aerodynamic and conjugate thermal performance of a TPMS core."""

    lattice_type: TPMSType
    mean_porosity: float
    specific_surface_area_m2_m3: float
    total_wetted_area_cm2: float
    hydraulic_diameter_mm: float
    darcy_permeability_m2: float
    forchheimer_coeff: float
    reynolds_number: float
    nusselt_number: float
    convective_h_w_m2k: float
    pressure_drop_pa: float
    thermal_resistance_k_w: float
    heat_dissipation_at_40k_delta_w: float
    self_supporting_printability: bool
    triangle_count: int


def evaluate_tpms_implicit(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    unit_cell_mm: float,
    lattice_type: TPMSType = "gyroid",
) -> float:
    """Evaluate the dimensionless TPMS implicit level-set function f(x, y, z)."""
    k = 2.0 * math.pi / unit_cell_mm
    kx, ky, kz = k * x_mm, k * y_mm, k * z_mm

    if lattice_type == "gyroid":
        return (
            math.sin(kx) * math.cos(ky)
            + math.sin(ky) * math.cos(kz)
            + math.sin(kz) * math.cos(kx)
        )
    if lattice_type == "schwarz_p":
        return math.cos(kx) + math.cos(ky) + math.cos(kz)
    # diamond (Schwarz-D)
    return (
        math.sin(kx) * math.sin(ky) * math.sin(kz)
        + math.sin(kx) * math.cos(ky) * math.cos(kz)
        + math.cos(kx) * math.sin(ky) * math.cos(kz)
        + math.cos(kx) * math.cos(ky) * math.sin(kz)
    )


def build_tpms_lattice_mesh(
    spec: TPMSSpec,
    grid_nx: int = 16,
    grid_ny: int = 12,
    grid_nz: int = 12,
) -> List[Triangle]:
    """Generate a watertight 3D triangle mesh of a functionally graded TPMS lattice.

    Uses voxel sheet classification with exact exposed boundary face triangulation
    so every solid TPMS sheet element forms a closed 2-manifold volume.
    """
    nx = max(6, grid_nx)
    ny = max(6, grid_ny)
    nz = max(6, grid_nz)

    dx = spec.width_mm / nx
    dy = spec.height_mm / ny
    dz = spec.length_mm / nz
    x0 = -spec.width_mm * 0.5
    y0 = -spec.height_mm * 0.5

    # Classify each voxel as solid sheet (True) or fluid void (False)
    solid = [[[False for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    solid_count = 0

    for ix in range(nx):
        cx = x0 + (ix + 0.5) * dx
        for iy in range(ny):
            cy = y0 + (iy + 0.5) * dy
            for iz in range(nz):
                cz = (iz + 0.5) * dz
                t_z = cz / max(1e-6, spec.length_mm)
                local_porosity = spec.porosity_inlet + t_z * (
                    spec.porosity_outlet - spec.porosity_inlet
                )
                # Map solid fraction (1 - porosity) to level-set sheet band |f| <= threshold
                solid_frac = max(0.08, min(0.65, 1.0 - local_porosity))
                band_threshold = solid_frac * 1.45

                val = evaluate_tpms_implicit(
                    cx, cy, cz, spec.unit_cell_mm, spec.lattice_type
                )
                if abs(val) <= band_threshold:
                    solid[ix][iy][iz] = True
                    solid_count += 1

    # Guarantee non-empty mesh even on coarse grids
    if solid_count == 0:
        solid[nx // 2][ny // 2][nz // 2] = True

    triangles: list[Triangle] = []

    def is_solid(i: int, j: int, k: int) -> bool:
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            return solid[i][j][k]
        return False

    for ix in range(nx):
        vx0 = x0 + ix * dx
        vx1 = vx0 + dx
        for iy in range(ny):
            vy0 = y0 + iy * dy
            vy1 = vy0 + dy
            for iz in range(nz):
                if not solid[ix][iy][iz]:
                    continue
                vz0 = iz * dz
                vz1 = vz0 + dz

                p000 = Vec3(vx0, vy0, vz0)
                p100 = Vec3(vx1, vy0, vz0)
                p110 = Vec3(vx1, vy1, vz0)
                p010 = Vec3(vx0, vy1, vz0)
                p001 = Vec3(vx0, vy0, vz1)
                p101 = Vec3(vx1, vy0, vz1)
                p111 = Vec3(vx1, vy1, vz1)
                p011 = Vec3(vx0, vy1, vz1)

                # -X face
                if not is_solid(ix - 1, iy, iz):
                    triangles.append(Triangle(p000, p001, p011))
                    triangles.append(Triangle(p000, p011, p010))
                # +X face
                if not is_solid(ix + 1, iy, iz):
                    triangles.append(Triangle(p100, p110, p111))
                    triangles.append(Triangle(p100, p111, p101))
                # -Y face
                if not is_solid(ix, iy - 1, iz):
                    triangles.append(Triangle(p000, p100, p101))
                    triangles.append(Triangle(p000, p101, p001))
                # +Y face
                if not is_solid(ix, iy + 1, iz):
                    triangles.append(Triangle(p010, p011, p111))
                    triangles.append(Triangle(p010, p111, p110))
                # -Z face
                if not is_solid(ix, iy, iz - 1):
                    triangles.append(Triangle(p000, p010, p110))
                    triangles.append(Triangle(p000, p110, p100))
                # +Z face
                if not is_solid(ix, iy, iz + 1):
                    triangles.append(Triangle(p001, p101, p111))
                    triangles.append(Triangle(p001, p111, p011))

    return triangles


def evaluate_tpms_thermo_hydraulics(
    spec: TPMSSpec,
    flow_cfm: float = 120.0,
) -> TPMSThermoHydraulicReport:
    """Evaluate Darcy-Forchheimer pressure drop and Nusselt convective heat transfer for TPMS."""
    mean_porosity = 0.5 * (spec.porosity_inlet + spec.porosity_outlet)
    cell_m = spec.unit_cell_mm * 1e-3

    # Dimensionless surface area density c_area depends on minimal surface topology:
    # Gyroid ~ 3.09, Schwarz-P ~ 2.35, Diamond ~ 3.84
    topology_factors = {
        "gyroid": (3.09, 0.42, 0.36),      # (area_coeff, nusselt_mult, forch_coeff)
        "schwarz_p": (2.35, 0.35, 0.28),
        "diamond": (3.84, 0.48, 0.44),
    }
    area_coeff, nu_mult, forch_c = topology_factors[spec.lattice_type]

    # Specific surface area S_v (m^2 / m^3)
    specific_area_m2_m3 = (area_coeff / cell_m) * (4.0 * mean_porosity * (1.0 - mean_porosity))
    core_vol_m3 = (spec.width_mm * spec.height_mm * spec.length_mm) * 1e-9
    total_wetted_area_m2 = specific_area_m2_m3 * core_vol_m3

    # Hydraulic diameter D_h = 4 * porosity / S_v
    dh_m = 4.0 * mean_porosity / max(1.0, specific_area_m2_m3)

    # Air properties
    rho = 1.184
    mu = 1.849e-5
    k_air = 0.0262
    pr = 0.71

    flow_m3_s = flow_cfm * 0.00047194745
    frontal_area_m2 = (spec.width_mm * spec.height_mm) * 1e-6
    v_superficial = flow_m3_s / max(1e-6, frontal_area_m2)
    v_interstitial = v_superficial / max(0.2, mean_porosity)

    re_dh = (rho * v_interstitial * dh_m) / mu

    # Kozeny-Carman / TPMS permeability K (m^2)
    darcy_k_m2 = (mean_porosity**3 * cell_m**2) / (
        120.0 * max(0.05, (1.0 - mean_porosity) ** 2)
    )

    # Darcy-Forchheimer pressure gradient: dP/dz = (mu/K)*v_s + (c_F / sqrt(K)) * rho * v_s^2
    length_m = spec.length_mm * 1e-3
    dp_viscous = (mu / darcy_k_m2) * v_superficial * length_m
    dp_inertial = (forch_c / math.sqrt(darcy_k_m2)) * rho * (v_superficial**2) * length_m
    total_dp_pa = dp_viscous + dp_inertial

    # TPMS Nusselt correlation (bicontinuous tortuous mixing enhancement):
    # Nu = c_nu * Re^0.68 * Pr^(1/3)
    nusselt = max(4.36, nu_mult * (max(10.0, re_dh) ** 0.68) * (pr ** (1.0 / 3.0)))
    h_conv = (nusselt * k_air) / max(1e-5, dh_m)

    # Overall finned/lattice effectiveness eta_o
    fin_length_m = 0.5 * min(spec.width_mm, spec.height_mm) * 1e-3
    m_fin = math.sqrt(
        (2.0 * h_conv)
        / max(1e-5, spec.thermal_conductivity_w_mk * (spec.wall_thickness_mm * 1e-3))
    )
    eta_lattice = math.tanh(m_fin * fin_length_m) / max(1e-6, m_fin * fin_length_m)
    effective_ua = eta_lattice * h_conv * total_wetted_area_m2
    r_th_k_w = 1.0 / max(1e-4, effective_ua)

    tris = build_tpms_lattice_mesh(spec, grid_nx=10, grid_ny=8, grid_nz=8)

    return TPMSThermoHydraulicReport(
        lattice_type=spec.lattice_type,
        mean_porosity=round(mean_porosity, 4),
        specific_surface_area_m2_m3=round(specific_area_m2_m3, 1),
        total_wetted_area_cm2=round(total_wetted_area_m2 * 1e4, 2),
        hydraulic_diameter_mm=round(dh_m * 1e3, 3),
        darcy_permeability_m2=darcy_k_m2,
        forchheimer_coeff=round(forch_c, 4),
        reynolds_number=round(re_dh, 1),
        nusselt_number=round(nusselt, 2),
        convective_h_w_m2k=round(h_conv, 2),
        pressure_drop_pa=round(total_dp_pa, 2),
        thermal_resistance_k_w=round(r_th_k_w, 5),
        heat_dissipation_at_40k_delta_w=round(40.0 / r_th_k_w, 1),
        self_supporting_printability=spec.lattice_type in ("gyroid", "diamond"),
        triangle_count=len(tris),
    )
