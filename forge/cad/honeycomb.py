"""Parametric flow-straightener cellular stator (honeycomb/grid) mesh generator and hydraulics."""
from __future__ import annotations
import math
from typing import Dict, List
from pydantic import BaseModel, Field
from forge.core.geometry import Triangle, Vec3


class HoneycombStatorSpec(BaseModel):
    """Specification for an internal cellular flow-straightener stator to eliminate fan swirl."""
    width_mm: float = Field(80.0, gt=10.0, description="Internal opening width in mm")
    height_mm: float = Field(80.0, gt=10.0, description="Internal opening height in mm")
    axial_length_mm: float = Field(12.0, gt=2.0, description="Axial chord length of straightener cells in mm")
    cell_pitch_mm: float = Field(12.0, gt=3.0, description="Cell center-to-center pitch in mm")
    cell_wall_mm: float = Field(0.8, gt=0.3, description="Cell septum wall thickness in mm")
    z_offset_mm: float = Field(0.0, description="Starting Z coordinate in mm")


def _build_capped_strut(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> List[Triangle]:
    """Build a watertight 12-triangle rectangular prism strut."""
    tris: List[Triangle] = []

    def quad(p: Vec3, q: Vec3, r: Vec3, s: Vec3):
        tris.append(Triangle(p, q, r))
        tris.append(Triangle(p, r, s))

    b0 = Vec3(x0, y0, z0)
    b1 = Vec3(x1, y0, z0)
    b2 = Vec3(x1, y1, z0)
    b3 = Vec3(x0, y1, z0)

    t0 = Vec3(x0, y0, z1)
    t1 = Vec3(x1, y0, z1)
    t2 = Vec3(x1, y1, z1)
    t3 = Vec3(x0, y1, z1)

    quad(b0, b1, t1, t0)
    quad(b1, b2, t2, t1)
    quad(b2, b3, t3, t2)
    quad(b3, b0, t0, t3)
    quad(b3, b2, b1, b0)  # Front cap (z = z0)
    quad(t0, t1, t2, t3)  # Back cap (z = z1)
    return tris


def generate_honeycomb_stator_mesh(spec: HoneycombStatorSpec) -> List[Triangle]:
    """Generate a watertight cellular flow-straightener mesh for suppressing fan swirl."""
    tris: List[Triangle] = []
    w_half = spec.width_mm / 2.0
    h_half = spec.height_mm / 2.0
    z0 = spec.z_offset_mm
    z1 = z0 + spec.axial_length_mm
    t_half = spec.cell_wall_mm / 2.0

    # Vertical straightening ribs
    n_x = max(1, int(spec.width_mm // spec.cell_pitch_mm) - 1)
    for i in range(1, n_x + 1):
        vx = -w_half + (i / float(n_x + 1)) * spec.width_mm
        tris.extend(_build_capped_strut(vx - t_half, vx + t_half, -h_half, h_half, z0, z1))

    # Horizontal straightening ribs
    n_y = max(1, int(spec.height_mm // spec.cell_pitch_mm) - 1)
    for j in range(1, n_y + 1):
        vy = -h_half + (j / float(n_y + 1)) * spec.height_mm
        tris.extend(_build_capped_strut(-w_half, w_half, vy - t_half, vy + t_half, z0, z1))

    return tris


def evaluate_honeycomb_hydraulics(
    spec: HoneycombStatorSpec,
    velocity_m_s: float,
    air_density_kg_m3: float = 1.184,
) -> Dict[str, float]:
    """Compute honeycomb screen porosity, Idelchik loss coefficient K, pressure drop (Pa), and swirl reduction (%)."""
    porosity = max(0.25, min(0.96, (1.0 - spec.cell_wall_mm / spec.cell_pitch_mm) ** 2))
    d_cell = max(1.0, spec.cell_pitch_mm - spec.cell_wall_mm)
    aspect_l_d = spec.axial_length_mm / d_cell

    # Idelchik Diagram 8-1 / 8-6 cellular honeycomb loss coefficient K
    k_contraction = 0.50 * (1.0 - porosity) + ((1.0 - porosity) ** 2)
    k_cell_friction = 0.022 * aspect_l_d
    k_total = (k_contraction + k_cell_friction) / (porosity ** 2)

    dp_pa = k_total * 0.5 * air_density_kg_m3 * (velocity_m_s ** 2)
    # Prandtl / Loehrke honeycomb transverse swirl attenuation law
    swirl_reduction_pct = 100.0 * (1.0 - math.exp(-1.45 * aspect_l_d))

    return {
        "porosity": round(porosity, 4),
        "cell_hydraulic_diameter_mm": round(d_cell, 2),
        "length_to_diameter_ratio": round(aspect_l_d, 2),
        "loss_coefficient_k": round(k_total, 4),
        "pressure_drop_pa": round(dp_pa, 2),
        "swirl_attenuation_pct": round(min(99.5, swirl_reduction_pct), 2),
    }

