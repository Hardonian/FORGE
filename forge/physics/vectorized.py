"""Vectorized SIMD & GPU Tensor Batch Aerodynamic Physics Evaluator.

Evaluates 10,000 to 100,000 candidate duct specifications simultaneously in a
single batched tensor pass on CPU SIMD or NVIDIA CUDA GPU using PyTorch,
achieving sub-20ms screening for large-scale genetic algorithms, Latin Hypercube
sampling, and neural active learning loops.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from forge.core.models import DuctSpec, PhysicsConditions


class BatchPhysicsResult(NamedTuple):
    """Batched tensor output for B candidate duct designs."""

    total_pressure_drop_pa: torch.Tensor  # Shape (B,)
    velocity_uniformity_index: torch.Tensor  # Shape (B,)
    reynolds_number: torch.Tensor  # Shape (B,)
    estimated_mass_grams: torch.Tensor  # Shape (B,)


def evaluate_duct_specs_tensor_batch(
    specs: list[DuctSpec] | torch.Tensor,
    conditions: PhysicsConditions | None = None,
    device: str | torch.device = "cpu",
    material_density_g_cm3: float = 1.20,
) -> BatchPhysicsResult:
    """Evaluate an arbitrary batch of duct specifications in parallel via vectorized PyTorch tensors.

    Parameters
    ----------
    specs : list[DuctSpec] or torch.Tensor of shape (B, 6)
        Columns: [inlet_w, inlet_h, outlet_w, outlet_h, length, wall] in mm.
    conditions : PhysicsConditions, optional
    device : str or torch.device, default "cpu"
    material_density_g_cm3 : float, default 1.20 (PC-CF)
    """
    cond = conditions or PhysicsConditions()

    if isinstance(specs, list):
        data = [
            [s.inlet_width, s.inlet_height, s.outlet_width, s.outlet_height, s.length, s.wall]
            for s in specs
        ]
        t_specs = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        t_specs = specs.to(dtype=torch.float32, device=device)

    # Extract columns (B,) in meters
    in_w = torch.clamp((t_specs[:, 0] - 2.0 * t_specs[:, 5]) * 1e-3, min=1e-3)
    in_h = torch.clamp((t_specs[:, 1] - 2.0 * t_specs[:, 5]) * 1e-3, min=1e-3)
    out_w = torch.clamp((t_specs[:, 2] - 2.0 * t_specs[:, 5]) * 1e-3, min=1e-3)
    out_h = torch.clamp((t_specs[:, 3] - 2.0 * t_specs[:, 5]) * 1e-3, min=1e-3)
    length_m = torch.clamp(t_specs[:, 4] * 1e-3, min=1e-3)
    wall_m = torch.clamp(t_specs[:, 5] * 1e-3, min=1e-4)

    # Cross-sectional areas & wetted perimeters
    a_in = in_w * in_h
    p_in = 2.0 * (in_w + in_h)
    a_out = out_w * out_h
    p_out = 2.0 * (out_w + out_h)

    dh_in = 4.0 * a_in / p_in
    dh_out = 4.0 * a_out / p_out
    dh_mean = 0.5 * (dh_in + dh_out)

    q_m3_s = cond.volumetric_flow_cfm * 0.000471947443
    rho = cond.air_density_kg_m3
    mu = cond.air_viscosity_pa_s

    v_in = q_m3_s / a_in
    v_out = q_m3_s / a_out
    v_mean = 0.5 * (v_in + v_out)

    reynolds = (rho * v_mean * dh_mean) / mu

    # Vectorized turbulent friction factor (Haaland equation)
    roughness_m = max(1e-7, cond.surface_roughness_mm * 1e-3)
    term1 = roughness_m / (3.7 * dh_mean)
    term2 = 5.74 / torch.clamp(reynolds, min=100.0) ** 0.9
    f_fric = 0.25 / (torch.log10(term1 + term2) ** 2)
    f_fric = torch.clamp(f_fric, min=0.010, max=0.12)

    dp_friction = f_fric * (length_m / dh_mean) * (0.5 * rho * (v_mean**2))

    # Vectorized expansion loss
    delta_w = torch.abs(out_w - in_w)
    delta_h = torch.abs(out_h - in_h)
    side_angle_rad = torch.atan(torch.maximum(delta_w, delta_h) / (2.0 * length_m))
    side_angle_deg = side_angle_rad * (180.0 / math.pi)

    area_ratio = a_out / a_in
    dynamic_head = 0.5 * rho * (v_in**2)

    # Piecewise phi based on side_angle_deg
    phi = torch.where(
        side_angle_deg <= 20.0,
        2.6 * torch.sin(side_angle_rad),
        3.5 * (torch.sin(side_angle_rad) ** 0.8),
    )

    k_expansion = torch.where(
        area_ratio > 1.0,
        ((1.0 - 1.0 / area_ratio) ** 2) * phi,
        0.45 * (1.0 - area_ratio),
    )
    dp_expansion = k_expansion * cond.minor_loss_multiplier * dynamic_head
    total_dp = dp_friction + dp_expansion

    # Vectorized flow uniformity index gamma
    tan_theta = torch.maximum(delta_w, delta_h) / (2.0 * length_m)
    uniformity = torch.clamp(
        1.0 - 0.42 * torch.sqrt(torch.clamp(tan_theta, min=0.0)) - 0.15 * torch.abs(area_ratio - 1.0) / (area_ratio + 1.0),
        min=0.35,
        max=0.99,
    )

    # Solid shell volume and mass estimation (g)
    # V_shell ~ Outer Volume - Inner Volume
    outer_w = in_w + 2.0 * wall_m
    outer_h = in_h + 2.0 * wall_m
    outer_w_out = out_w + 2.0 * wall_m
    outer_h_out = out_h + 2.0 * wall_m

    a_outer_mean = 0.5 * (outer_w * outer_h + outer_w_out * outer_h_out)
    a_inner_mean = 0.5 * (a_in + a_out)
    vol_shell_m3 = (a_outer_mean - a_inner_mean) * length_m
    # mass in grams = volume (m^3) * 1e6 (cm^3/m^3) * density (g/cm^3)
    mass_grams = torch.clamp(vol_shell_m3 * 1e6 * material_density_g_cm3, min=1.0)

    return BatchPhysicsResult(
        total_pressure_drop_pa=total_dp,
        velocity_uniformity_index=uniformity,
        reynolds_number=reynolds,
        estimated_mass_grams=mass_grams,
    )
