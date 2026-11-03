"""Time-domain transient thermal-fluid workload and closed-loop PID fan PWM controller simulator."""
from __future__ import annotations
from typing import List
from pydantic import BaseModel
from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.heatsink import HeatsinkSpec, evaluate_heatsink_at_cfm, solve_coupled_duct_heatsink_system
from forge.physics.solver_1d import solve_duct_physics


class TransientThermalReport(BaseModel):
    """Time-domain simulation report for an AI training/inference workload power step."""
    duration_s: float
    dt_s: float
    idle_watts: float
    burst_watts: float
    target_temp_c: float
    peak_hotspot_temp_c: float
    final_hotspot_temp_c: float
    fan_spinup_90pct_time_s: float
    thermal_throttling_triggered: bool
    time_s: List[float]
    heat_load_watts: List[float]
    hotspot_die_temp_c: List[float]
    fan_pwm_pct: List[float]
    airflow_cfm: List[float]


def simulate_transient_thermal_response(
    duct: DuctSpec,
    heatsink: HeatsinkSpec | None = None,
    idle_watts: float = 65.0,
    burst_watts: float = 450.0,
    ambient_temp_c: float = 25.0,
    target_temp_c: float = 74.0,
    duration_s: float = 30.0,
    dt_s: float = 0.5,
) -> TransientThermalReport:
    """Simulate transient GPU junction temperature and closed-loop PID fan PWM response under a step TDP burst."""
    hs = heatsink or HeatsinkSpec(
        width_mm=duct.outlet_width,
        fin_height_mm=max(12.0, duct.outlet_height - 4.0),
    )
    cond_burst = PhysicsConditions(
        volumetric_flow_cfm=150.0,
        air_temp_c=ambient_temp_c,
        heat_dissipation_watts=burst_watts,
    )
    nominal_duct_phys = solve_duct_physics(duct, cond_burst)
    gamma = nominal_duct_phys.flow_uniformity_index
    coupled_max = solve_coupled_duct_heatsink_system(duct, hs, cond_burst)
    max_eq_cfm = max(25.0, coupled_max.equilibrium_flow_cfm)

    # Thermal capacitance C_th (J/K) of copper vapor-chamber base + fin stack (~0.65 kg * 385 J/kg*K)
    vol_hs_m3 = (hs.width_mm * hs.fin_height_mm * hs.length_mm) * 1e-9 * 0.28
    rho_hs = 8960.0 if "COPPER" in hs.material.upper() else 2700.0
    cp_hs = 385.0 if "COPPER" in hs.material.upper() else 900.0
    c_th_j_k = max(85.0, vol_hs_m3 * rho_hs * cp_hs)

    # Fan rotor mechanical spin-up time constant (seconds)
    tau_fan_s = 1.6

    # Initial steady-state at idle_watts (25% minimum PWM floor)
    pwm_pct = 25.0
    cur_cfm = max_eq_cfm * (pwm_pct / 100.0)
    idle_cond = PhysicsConditions(volumetric_flow_cfm=cur_cfm, air_temp_c=ambient_temp_c, heat_dissipation_watts=idle_watts)
    idle_hs_rep = evaluate_heatsink_at_cfm(hs, idle_cond, cur_cfm, flow_uniformity_index=gamma)
    cur_temp_c = idle_hs_rep.hotspot_die_temp_c

    n_steps = max(4, int(duration_s / max(0.05, dt_s)))
    t_arr: List[float] = []
    q_arr: List[float] = []
    temp_arr: List[float] = []
    pwm_arr: List[float] = []
    cfm_arr: List[float] = []

    integral_err = 0.0
    prev_err = 0.0
    spinup_90_time = duration_s

    for step in range(n_steps + 1):
        t = round(step * dt_s, 2)
        q_load = idle_watts if t < 2.0 else burst_watts

        t_arr.append(t)
        q_arr.append(round(q_load, 1))
        temp_arr.append(round(cur_temp_c, 2))
        pwm_arr.append(round(pwm_pct, 1))
        cfm_arr.append(round(cur_cfm, 2))

        # BMC/IPMI closed-loop PID + TDP feedforward fan controller
        err = max(0.0, cur_temp_c - (target_temp_c - 18.0))
        integral_err = max(0.0, min(50.0, integral_err + err * dt_s))
        deriv_err = max(0.0, (err - prev_err) / max(1e-3, dt_s))
        prev_err = err

        ff_pwm = 25.0 + 55.0 * (max(0.0, q_load - idle_watts) / max(10.0, burst_watts - idle_watts))
        pid_out = ff_pwm + 2.5 * err + 0.35 * integral_err + 0.5 * deriv_err
        pwm_pct = max(20.0, min(100.0, pid_out))

        cmd_cfm = max_eq_cfm * (pwm_pct / 100.0)
        cur_cfm += (dt_s / tau_fan_s) * (cmd_cfm - cur_cfm)
        if cur_cfm >= 0.90 * max_eq_cfm and spinup_90_time == duration_s:
            spinup_90_time = t

        step_cond = PhysicsConditions(
            volumetric_flow_cfm=max(2.0, cur_cfm),
            air_temp_c=ambient_temp_c,
            heat_dissipation_watts=q_load,
        )
        hs_step = evaluate_heatsink_at_cfm(hs, step_cond, max(2.0, cur_cfm), flow_uniformity_index=gamma)
        t_eq_target = hs_step.hotspot_die_temp_c
        r_eff = max(0.02, hs_step.heatsink_thermal_resistance_k_w)
        tau_thermal = c_th_j_k * r_eff
        cur_temp_c += (dt_s / max(0.25, tau_thermal)) * (t_eq_target - cur_temp_c)

    peak_t = max(temp_arr)
    final_t = temp_arr[-1]
    throttled = peak_t > hs.max_junction_temp_c

    return TransientThermalReport(
        duration_s=duration_s,
        dt_s=dt_s,
        idle_watts=idle_watts,
        burst_watts=burst_watts,
        target_temp_c=target_temp_c,
        peak_hotspot_temp_c=round(peak_t, 2),
        final_hotspot_temp_c=round(final_t, 2),
        fan_spinup_90pct_time_s=round(spinup_90_time, 2),
        thermal_throttling_triggered=throttled,
        time_s=t_arr,
        heat_load_watts=q_arr,
        hotspot_die_temp_c=temp_arr,
        fan_pwm_pct=pwm_arr,
        airflow_cfm=cfm_arr,
    )
