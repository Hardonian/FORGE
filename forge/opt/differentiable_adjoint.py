"""End-to-End Differentiable PyTorch Adjoint Co-Design Optimizer.

Constructs a fully differentiable PyTorch computation graph of multi-station
duct aerodynamics (Darcy-Weisbach friction, Stratford boundary-layer fatigue
expansion profile, and corner fillet secondary-flow relief) coupled with finned
heatsink conjugate heat transfer (developing channel friction, Nusselt & fin
efficiency).

Uses exact automatic differentiation (torch.autograd) to compute analytical
adjoint sensitivities and co-optimize the duct loft and heatsink simultaneously.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.heatsink import (
    HEATSINK_CONDUCTIVITY_W_MK,
    HeatsinkSpec,
    solve_coupled_duct_heatsink_system,
)
from forge.physics.solver_1d import solve_duct_physics


class AdjointCoDesignResult(BaseModel):
    """Result of PyTorch differentiable adjoint co-design optimization."""

    initial_pressure_drop_pa: float
    optimized_pressure_drop_pa: float
    initial_uniformity: float
    optimized_uniformity: float
    initial_hotspot_temp_c: float
    optimized_hotspot_temp_c: float
    pressure_drop_reduction_pct: float
    hotspot_reduction_c: float
    bezier_weights: tuple[float, float]
    optimized_corner_radius_mm: float
    optimized_fin_count: int
    adjoint_sensitivities: dict[str, float] = Field(
        default_factory=dict,
        description="First-order analytical gradients d(Loss)/d(theta) at initial state",
    )
    loss_trajectory: list[float] = Field(default_factory=list)
    optimized_spec: DuctSpec
    optimized_heatsink: HeatsinkSpec


def run_differentiable_adjoint_codesign(
    spec: DuctSpec,
    heatsink: HeatsinkSpec | None = None,
    conditions: PhysicsConditions | None = None,
    steps: int = 40,
    learning_rate: float = 0.12,
) -> AdjointCoDesignResult:
    """Co-optimize duct loft profile, corner fillet, and heatsink fin density via PyTorch autograd."""
    cond = conditions or PhysicsConditions()
    if heatsink is None:
        heatsink = HeatsinkSpec(
            width_mm=spec.outlet_width,
            fin_height_mm=max(10.0, spec.outlet_height - 4.0),
            length_mm=max(80.0, spec.length * 0.85),
            fin_count=max(24, int(spec.outlet_width * 0.32)),
        )

    init_phys = solve_duct_physics(spec, cond)
    init_cht = solve_coupled_duct_heatsink_system(spec, heatsink, cond)

    min_dim = min(spec.inlet_width, spec.inlet_height, spec.outlet_width, spec.outlet_height)
    max_corner = max(2.5, min_dim * 0.25)

    # Start near linear loft (w1 ~ 0.33, w2 ~ 0.67) and moderate fin count
    theta = torch.tensor(
        [
            -0.35,  # logit for w1 in [0.20, 0.78]
            -0.10,  # logit for w2 in [0.45, 0.96]
            -0.80,  # logit for corner_radius in [spec.corner_radius, max_corner]
            0.0,  # logit for fin_count in [init_fins, init_fins + 28]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([theta], lr=learning_rate)

    num_st = 16
    t_grid = torch.linspace(0.0, 1.0, num_st + 1, dtype=torch.float64)
    t0_grid = t_grid[:-1]
    length_m = max(0.001, spec.length * 1e-3)
    dz = length_m / float(num_st)

    q_m3_s = cond.volumetric_flow_cfm * 0.000471947443
    rho = cond.air_density_kg_m3
    mu = cond.air_viscosity_pa_s
    k_air = 0.0263
    cp_air = 1005.0

    w_in = spec.inlet_width * 1e-3
    h_in = spec.inlet_height * 1e-3
    w_out = spec.outlet_width * 1e-3
    h_out = spec.outlet_height * 1e-3
    delta_w = abs(w_out - w_in)
    delta_h = abs(h_out - h_in)
    tan_lin = max(1e-5, max(delta_w, delta_h) / (2.0 * length_m))
    bl_weights = 0.65 + 0.70 * (t0_grid**1.3)
    linear_penalty_sum = torch.sum((tan_lin**1.6) * bl_weights)

    k_solid = HEATSINK_CONDUCTIVITY_W_MK.get(heatsink.material.upper(), 398.0)
    init_fins = float(heatsink.fin_count)

    initial_grads: dict[str, float] = {}
    loss_history: list[float] = []

    for step_idx in range(max(5, steps)):
        optimizer.zero_grad()

        # Differentiable parameter mapping
        w1 = 0.20 + 0.58 * torch.sigmoid(theta[0])
        w2 = 0.45 + 0.51 * torch.sigmoid(theta[1])
        r_corner_mm = spec.corner_radius + (max_corner - spec.corner_radius) * torch.sigmoid(theta[2])
        fin_count_cont = init_fins + 26.0 * torch.sigmoid(theta[3])

        # Exact cubic Bezier profile evaluation matching evaluate_bezier_scalar
        one_minus_t = 1.0 - t_grid
        f_t = 3.0 * (one_minus_t**2) * t_grid * w1 + 3.0 * one_minus_t * (t_grid**2) * w2 + (t_grid**3)
        w_t = w_in + f_t * (w_out - w_in)
        h_t = h_in + f_t * (h_out - h_in)

        dw_seg = torch.sqrt((w_t[1:] - w_t[:-1]) ** 2 + 1e-12)
        dh_seg = torch.sqrt((h_t[1:] - h_t[:-1]) ** 2 + 1e-12)
        tan_seg = torch.maximum(dw_seg, dh_seg) / (2.0 * dz)

        fatigue_penalty_sum = torch.sum((tan_seg**1.6) * bl_weights)
        bezier_ratio = torch.clamp(fatigue_penalty_sum / linear_penalty_sum, min=0.68, max=1.35)

        # Corner fillet area impact vs minor-loss vortex relief
        inner_r_m = torch.clamp((r_corner_mm - spec.wall) * 1e-3, min=0.0)
        in_w_inner = max(0.001, (spec.inlet_width - 2.0 * spec.wall) * 1e-3)
        in_h_inner = max(0.001, (spec.inlet_height - 2.0 * spec.wall) * 1e-3)
        out_w_inner = max(0.001, (spec.outlet_width - 2.0 * spec.wall) * 1e-3)
        out_h_inner = max(0.001, (spec.outlet_height - 2.0 * spec.wall) * 1e-3)

        a_in = in_w_inner * in_h_inner - (4.0 - math.pi) * (inner_r_m**2)
        p_in = 2.0 * (in_w_inner + in_h_inner) - (8.0 - 2.0 * math.pi) * inner_r_m
        a_out = out_w_inner * out_h_inner - (4.0 - math.pi) * (inner_r_m**2)
        p_out = 2.0 * (out_w_inner + out_h_inner) - (8.0 - 2.0 * math.pi) * inner_r_m

        v_in = q_m3_s / a_in
        v_out = q_m3_s / a_out
        v_mean = 0.5 * (v_in + v_out)
        dh_mean = 0.5 * ((4.0 * a_in / p_in) + (4.0 * a_out / p_out))
        reynolds = (rho * v_mean * dh_mean) / mu
        roughness = max(1e-7, cond.surface_roughness_mm * 1e-3)
        f_fric = 0.25 / (torch.log10(roughness / (3.7 * dh_mean) + 5.74 / (reynolds**0.9)) ** 2)
        dp_friction = f_fric * (length_m / dh_mean) * (0.5 * rho * (v_mean**2))

        max_possible_r = 0.5 * min_dim
        fillet_factor = 1.0 - 0.15 * torch.clamp(r_corner_mm / max_possible_r, max=1.0)
        base_dp_exp = max(0.5, init_phys.expansion_loss_pa)
        dp_expansion = base_dp_exp * fillet_factor * bezier_ratio
        total_duct_dp = dp_friction + dp_expansion

        # Differentiable exit uniformity
        exit_slope = tan_seg[-3:].mean() / tan_lin
        uniformity_est = torch.clamp(
            init_phys.flow_uniformity_index + 0.06 * (1.0 - exit_slope) + 0.04 * (r_corner_mm / max_possible_r),
            min=0.55,
            max=0.98,
        )

        # Differentiable coupled heatsink CHT model
        b_width_m = heatsink.width_mm * 1e-3
        b_len_m = heatsink.length_mm * 1e-3
        f_height_m = heatsink.fin_height_mm * 1e-3
        t_fin_m = heatsink.fin_thickness_mm * 1e-3

        gap_m = torch.clamp(
            (b_width_m - fin_count_cont * t_fin_m) / (fin_count_cont - 1.0),
            min=0.0005,
        )
        dh_fin = (2.0 * gap_m * f_height_m) / (gap_m + f_height_m)
        a_open = (fin_count_cont - 1.0) * gap_m * f_height_m
        v_channel = q_m3_s / a_open
        re_hs = (rho * v_channel * dh_fin) / mu

        f_hs = 0.3164 / (torch.clamp(re_hs, min=100.0) ** 0.25)
        dp_heatsink = f_hs * (b_len_m / dh_fin) * (0.5 * rho * (v_channel**2))

        nu_hs = 7.54 + 0.023 * (torch.clamp(re_hs, min=100.0) ** 0.8) * (0.707**0.4)
        h_hs = (nu_hs * k_air) / dh_fin
        m_fin = torch.sqrt((2.0 * h_hs) / (k_solid * t_fin_m))
        lc = f_height_m + 0.5 * t_fin_m
        eta_fin = torch.tanh(m_fin * lc) / (m_fin * lc)
        a_fins = 2.0 * fin_count_cont * f_height_m * b_len_m
        a_base = (fin_count_cont - 1.0) * gap_m * b_len_m
        a_total = a_fins + a_base
        eta_overall = 1.0 - (a_fins / a_total) * (1.0 - eta_fin)
        r_conv = 1.0 / (eta_overall * h_hs * a_total)
        r_hs_total = r_conv + heatsink.tim_resistance_k_w

        maldistribution_factor = 1.0 + 0.65 * (1.0 - uniformity_est)
        dt_air_mean = 0.5 * (cond.heat_dissipation_watts / (rho * q_m3_s * cp_air))
        hotspot_temp = (
            cond.air_temp_c
            + dt_air_mean * maldistribution_factor
            + cond.heat_dissipation_watts * r_hs_total * maldistribution_factor
        )

        # Multi-objective differentiable loss: minimize duct dP + hotspot temp + heatsink dP penalty
        loss = (
            (total_duct_dp / max(1.0, init_phys.total_pressure_drop_pa))
            + 1.5 * (hotspot_temp / max(10.0, init_cht.hotspot_die_temp_c))
            + 0.05 * (dp_heatsink / 50.0)
            - 0.5 * uniformity_est
        )

        loss.backward()

        if step_idx == 0 and theta.grad is not None:
            initial_grads = {
                "dLoss_dBezierW1": round(float(theta.grad[0].item()), 6),
                "dLoss_dBezierW2": round(float(theta.grad[1].item()), 6),
                "dLoss_dCornerRadius": round(float(theta.grad[2].item()), 6),
                "dLoss_dHeatsinkFinCount": round(float(theta.grad[3].item()), 6),
            }

        loss_history.append(round(float(loss.item()), 5))
        optimizer.step()

    with torch.no_grad():
        opt_w1 = round(float(0.20 + 0.58 * torch.sigmoid(theta[0]).item()), 4)
        opt_w2 = round(float(0.45 + 0.51 * torch.sigmoid(theta[1]).item()), 4)
        opt_corner = round(
            float((spec.corner_radius + (max_corner - spec.corner_radius) * torch.sigmoid(theta[2])).item()),
            2,
        )
        opt_fins = int(round(float((init_fins + 26.0 * torch.sigmoid(theta[3])).item())))

    w_ctrl = [
        round(spec.inlet_width + opt_w1 * (spec.outlet_width - spec.inlet_width), 2),
        round(spec.inlet_width + opt_w2 * (spec.outlet_width - spec.inlet_width), 2),
    ]
    h_ctrl = [
        round(spec.inlet_height + opt_w1 * (spec.outlet_height - spec.inlet_height), 2),
        round(spec.inlet_height + opt_w2 * (spec.outlet_height - spec.inlet_height), 2),
    ]

    # Candidate 1: with optimized corner fillet; Candidate 2: keeping original corner radius if area contraction dominates
    cand_spec_a = spec.model_copy(
        update={
            "width_profile_ctrl": w_ctrl,
            "height_profile_ctrl": h_ctrl,
            "corner_radius": opt_corner,
            "guide_vanes": max(1, spec.guide_vanes),
        }
    )
    cand_spec_b = spec.model_copy(
        update={
            "width_profile_ctrl": w_ctrl,
            "height_profile_ctrl": h_ctrl,
            "guide_vanes": max(1, spec.guide_vanes),
        }
    )
    phys_a = solve_duct_physics(cand_spec_a, cond)
    phys_b = solve_duct_physics(cand_spec_b, cond)
    if phys_a.total_pressure_drop_pa <= phys_b.total_pressure_drop_pa:
        opt_spec, opt_phys = cand_spec_a, phys_a
    else:
        opt_spec, opt_phys = cand_spec_b, phys_b

    opt_heatsink = heatsink.model_copy(update={"fin_count": opt_fins})
    opt_cht = solve_coupled_duct_heatsink_system(opt_spec, opt_heatsink, cond)
    if opt_cht.hotspot_die_temp_c >= init_cht.hotspot_die_temp_c:
        # Select optimal fin count balancing convective area vs channel impedance
        best_hs = opt_heatsink
        best_cht = opt_cht
        for f_try in (int(init_fins) + 6, int(init_fins) + 12, max(16, int(init_fins) - 4)):
            hs_try = heatsink.model_copy(update={"fin_count": f_try})
            cht_try = solve_coupled_duct_heatsink_system(opt_spec, hs_try, cond)
            if cht_try.hotspot_die_temp_c < best_cht.hotspot_die_temp_c:
                best_hs = hs_try
                best_cht = cht_try
        opt_heatsink = best_hs
        opt_cht = best_cht

    dp_red_pct = (
        100.0
        * (init_phys.total_pressure_drop_pa - opt_phys.total_pressure_drop_pa)
        / max(1e-4, init_phys.total_pressure_drop_pa)
    )
    hotspot_red_c = init_cht.hotspot_die_temp_c - opt_cht.hotspot_die_temp_c

    return AdjointCoDesignResult(
        initial_pressure_drop_pa=init_phys.total_pressure_drop_pa,
        optimized_pressure_drop_pa=opt_phys.total_pressure_drop_pa,
        initial_uniformity=init_phys.flow_uniformity_index,
        optimized_uniformity=opt_phys.flow_uniformity_index,
        initial_hotspot_temp_c=init_cht.hotspot_die_temp_c,
        optimized_hotspot_temp_c=opt_cht.hotspot_die_temp_c,
        pressure_drop_reduction_pct=round(dp_red_pct, 2),
        hotspot_reduction_c=round(hotspot_red_c, 2),
        bezier_weights=(opt_w1, opt_w2),
        optimized_corner_radius_mm=opt_spec.corner_radius,
        optimized_fin_count=opt_heatsink.fin_count,
        adjoint_sensitivities=initial_grads,
        loss_trajectory=loss_history,
        optimized_spec=opt_spec,
        optimized_heatsink=opt_heatsink,
    )
