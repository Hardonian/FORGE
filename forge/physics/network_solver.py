"""Parallel hydraulic and thermal network solver for 1-to-N splitter manifolds."""
from __future__ import annotations
from typing import Dict, List, Tuple
from pydantic import BaseModel
from forge.core.models import CrossSectionShape, DuctSpec, PhysicsConditions
from forge.cad.manifold import ManifoldSpec, branch_to_duct_spec
from forge.physics.solver_1d import _compute_duct_dp_for_cfm


class BranchFlowResult(BaseModel):
    """Solved hydraulic and thermal state for an individual manifold branch."""
    name: str
    flow_cfm: float
    flow_share_pct: float
    outlet_velocity_m_s: float
    branch_pressure_drop_pa: float
    exhaust_temp_rise_c: float


class ManifoldNetworkReport(BaseModel):
    """Solved hydraulic and thermal equilibrium across a 1-to-N splitter manifold."""
    total_flow_cfm: float
    plenum_pressure_drop_pa: float
    branch_pressure_drop_pa: float
    total_manifold_pressure_drop_pa: float
    branches: List[BranchFlowResult]
    flow_balance_index: float
    thermal_balance_index: float


def _branch_total_dp(
    spec: ManifoldSpec,
    branch_idx: int,
    conditions: PhysicsConditions,
    branch_cfm: float,
) -> Tuple[float, float]:
    """Compute (total_branch_dp_pa, outlet_velocity_m_s) including downstream device K-loss."""
    b_duct = branch_to_duct_spec(spec, branch_idx)
    branch = spec.branches[branch_idx]
    _, _, duct_dp, _, v_out, _, _ = _compute_duct_dp_for_cfm(b_duct, conditions, max(0.05, branch_cfm))
    # Add bifurcation tee-split loss + downstream device resistance (K * 0.5 * rho * v_out^2)
    rho = conditions.air_density_kg_m3
    downstream_dp = (0.35 + branch.downstream_k_loss) * 0.5 * rho * (v_out ** 2)
    return (duct_dp + downstream_dp, v_out)


def solve_manifold_network(
    spec: ManifoldSpec,
    total_flow_cfm: float,
    conditions: PhysicsConditions | None = None,
) -> ManifoldNetworkReport:
    """Solve parallel branch flow distribution where dP_1(Q_1) = dP_2(Q_2) = ... = dP_N(Q_N) and sum(Q_i) = Q_total."""
    cond = conditions or PhysicsConditions(volumetric_flow_cfm=total_flow_cfm)
    n_branches = len(spec.branches)

    # 1. Shared plenum pressure drop at total_flow_cfm
    plenum_duct = DuctSpec(
        inlet_width=spec.inlet_width,
        inlet_height=spec.inlet_height,
        outlet_width=spec.inlet_width,
        outlet_height=spec.inlet_height,
        length=spec.plenum_length,
        wall=spec.wall,
        inlet_shape=spec.inlet_shape,
        outlet_shape=CrossSectionShape.RECTANGULAR,
    )
    _, _, plenum_dp, _, _, _, _ = _compute_duct_dp_for_cfm(plenum_duct, cond, total_flow_cfm)

    # 2. Iterative parallel conductance solver for Q_i across N branches
    q_branches = [total_flow_cfm / float(n_branches)] * n_branches

    for _ in range(35):
        dps: List[float] = []
        for idx in range(n_branches):
            dp_i, _ = _branch_total_dp(spec, idx, cond, q_branches[idx])
            dps.append(max(1e-4, dp_i))

        # Target common pressure drop across parallel branches
        mean_dp = sum(dps) / float(n_branches)
        # Since dP ~ R * Q^2, update Q_i <- Q_i * sqrt(mean_dp / dP_i) with damping
        raw_q = [q_branches[i] * ((mean_dp / dps[i]) ** 0.45) for i in range(n_branches)]
        sum_raw = max(1e-6, sum(raw_q))
        q_branches = [q * (total_flow_cfm / sum_raw) for q in raw_q]

    branch_results: List[BranchFlowResult] = []
    rho = cond.air_density_kg_m3
    cp_air = 1005.0
    temp_rises: List[float] = []

    for idx, branch in enumerate(spec.branches):
        q_i = q_branches[idx]
        dp_i, v_out_i = _branch_total_dp(spec, idx, cond, q_i)
        m_dot_i = rho * (q_i * 0.000471947443)
        dt_i = branch.heat_load_watts / max(1e-5, m_dot_i * cp_air)
        temp_rises.append(dt_i)
        branch_results.append(
            BranchFlowResult(
                name=branch.name,
                flow_cfm=round(q_i, 2),
                flow_share_pct=round(100.0 * q_i / max(1e-6, total_flow_cfm), 2),
                outlet_velocity_m_s=round(v_out_i, 2),
                branch_pressure_drop_pa=round(dp_i, 2),
                exhaust_temp_rise_c=round(dt_i, 2),
            )
        )

    eq_branch_dp = sum(b.branch_pressure_drop_pa for b in branch_results) / float(n_branches)
    mean_q = total_flow_cfm / float(n_branches)
    max_q_dev = max(abs(q - mean_q) for q in q_branches) / max(1e-6, total_flow_cfm)
    flow_balance_idx = max(0.0, min(1.0, 1.0 - max_q_dev))

    mean_dt = max(1e-4, sum(temp_rises) / float(n_branches))
    max_dt_dev = max(abs(dt - mean_dt) for dt in temp_rises) / mean_dt
    thermal_balance_idx = max(0.0, min(1.0, 1.0 - 0.5 * max_dt_dev))

    return ManifoldNetworkReport(
        total_flow_cfm=round(total_flow_cfm, 2),
        plenum_pressure_drop_pa=round(plenum_dp, 2),
        branch_pressure_drop_pa=round(eq_branch_dp, 2),
        total_manifold_pressure_drop_pa=round(plenum_dp + eq_branch_dp, 2),
        branches=branch_results,
        flow_balance_index=round(flow_balance_idx, 4),
        thermal_balance_index=round(thermal_balance_idx, 4),
    )


def auto_balance_manifold(
    spec: ManifoldSpec,
    total_flow_cfm: float,
    conditions: PhysicsConditions | None = None,
    iterations: int = 20,
) -> Tuple[ManifoldSpec, ManifoldNetworkReport, Dict[str, float]]:
    """Automatically adjust branch outlet widths so each branch receives flow proportional to its heat_load_watts."""
    cond = conditions or PhysicsConditions(volumetric_flow_cfm=total_flow_cfm)
    initial_rep = solve_manifold_network(spec, total_flow_cfm, cond)

    total_watts = sum(max(1.0, b.heat_load_watts) for b in spec.branches)
    target_shares = [max(1.0, b.heat_load_watts) / total_watts for b in spec.branches]

    current_branches = [b.model_copy() for b in spec.branches]
    for _ in range(iterations):
        candidate_spec = spec.model_copy(update={"branches": current_branches})
        rep = solve_manifold_network(candidate_spec, total_flow_cfm, cond)
        for idx, b_res in enumerate(rep.branches):
            actual_share = b_res.flow_cfm / max(1e-6, total_flow_cfm)
            ratio = (target_shares[idx] / max(1e-4, actual_share)) ** 0.35
            new_w = min(spec.inlet_width * 1.2, max(spec.wall * 3.0, current_branches[idx].outlet_width * ratio))
            current_branches[idx] = current_branches[idx].model_copy(update={"outlet_width": round(new_w, 2)})

    balanced_spec = spec.model_copy(update={"branches": current_branches})
    final_rep = solve_manifold_network(balanced_spec, total_flow_cfm, cond)

    summary = {
        "initial_thermal_balance_index": initial_rep.thermal_balance_index,
        "balanced_thermal_balance_index": final_rep.thermal_balance_index,
        "initial_pressure_drop_pa": initial_rep.total_manifold_pressure_drop_pa,
        "balanced_pressure_drop_pa": final_rep.total_manifold_pressure_drop_pa,
    }
    return (balanced_spec, final_rep, summary)
