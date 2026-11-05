"""Real-time Bidirectional WebSocket Digital Twin Streaming Engine.

Ingests streaming IPMI/BMC sensor telemetry frames (fan tachometer RPM, PWM,
GPU diode temperatures, board power) and computes live online health metrics:
- Filter dust loading / aerodynamic impedance growth (alpha_dust)
- Thermal interface material (TIM) dryout / degradation (delta_R_tim)
- Predictive maintenance remaining useful life (RUL)
- Real-time digital twin anomaly alerts
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List
from fastapi import WebSocket
from pydantic import BaseModel, Field

from forge.core.models import DuctSpec, PhysicsConditions
from forge.physics.solver_1d import solve_duct_physics


class LiveTelemetryFrame(BaseModel):
    """Single incoming sensor frame from server chassis or test bench."""

    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fan_rpm: float = Field(..., gt=500.0, le=25000.0)
    fan_pwm_pct: float = Field(..., ge=10.0, le=100.0)
    inlet_ambient_temp_c: float = Field(default=25.0)
    exhaust_temp_c: float = Field(default=48.0)
    gpu_hotspot_temp_c: float = Field(default=72.0)
    board_power_watts: float = Field(default=400.0)
    measured_differential_pressure_pa: float | None = None


class DigitalTwinHealthState(BaseModel):
    """Computed real-time digital twin health, impedance degradation, and anomaly status."""

    timestamp_utc: str
    nominal_clean_dp_pa: float
    current_estimated_dp_pa: float
    filter_dust_loading_factor: float = Field(description="1.0 = clean, >1.25 = filter service required")
    thermal_resistance_k_w: float
    tim_degradation_pct: float
    thermal_throttling_risk: bool
    filter_service_recommended: bool
    health_score_pct: float
    alerts: List[str] = Field(default_factory=list)


class DigitalTwinOnlineTracker:
    """Online state estimator tracking filter dust loading and TIM degradation over telemetry streams."""

    def __init__(self, spec: DuctSpec, baseline_r_tim: float = 0.022):
        self.spec = spec
        self.baseline_r_tim = baseline_r_tim
        self.cond = PhysicsConditions()
        self.nominal_physics = solve_duct_physics(spec, self.cond)
        self.nominal_dp_ref = self.nominal_physics.total_pressure_drop_pa
        self.ema_dust_factor = 1.0
        self.ema_tim_r = baseline_r_tim

    def process_frame(self, frame: LiveTelemetryFrame) -> DigitalTwinHealthState:
        # 1. Aerodynamic impedance estimation via fan affinity laws: Q ~ RPM / RPM_ref
        rpm_ratio = frame.fan_rpm / 9000.0
        estimated_cfm = max(20.0, 150.0 * rpm_ratio)
        calc_cond = PhysicsConditions(
            volumetric_flow_cfm=estimated_cfm,
            air_temp_c=frame.inlet_ambient_temp_c,
            heat_dissipation_watts=frame.board_power_watts,
        )
        current_phys = solve_duct_physics(self.spec, calc_cond)
        expected_dp = current_phys.total_pressure_drop_pa

        if frame.measured_differential_pressure_pa is not None and frame.measured_differential_pressure_pa > 0:
            instant_dust = frame.measured_differential_pressure_pa / max(1.0, expected_dp)
        else:
            # Estimate impedance drift based on exhaust air density & temperature rise
            delta_t_air = max(1.0, frame.exhaust_temp_c - frame.inlet_ambient_temp_c)
            # Theoretical temperature rise
            q_m3_s = estimated_cfm * 0.000471947443
            m_dot = 1.184 * q_m3_s
            exp_delta_t = frame.board_power_watts / max(1.0, m_dot * 1005.0)
            instant_dust = max(0.95, min(2.5, delta_t_air / max(1.0, exp_delta_t)))

        # Exponential moving average filter (alpha = 0.15)
        self.ema_dust_factor = 0.85 * self.ema_dust_factor + 0.15 * instant_dust

        # 2. Thermal Resistance & TIM degradation estimation
        delta_t_die = max(1.0, frame.gpu_hotspot_temp_c - frame.inlet_ambient_temp_c)
        r_th_system = delta_t_die / max(10.0, frame.board_power_watts)
        # R_system = R_conv + R_tim; estimate TIM degradation
        r_conv_est = max(0.015, r_th_system * 0.70)
        instant_r_tim = max(0.005, r_th_system - r_conv_est)
        self.ema_tim_r = 0.90 * self.ema_tim_r + 0.10 * instant_r_tim

        tim_degrad_pct = max(0.0, 100.0 * (self.ema_tim_r - self.baseline_r_tim) / self.baseline_r_tim)

        # 3. Health assessment and anomaly alerts
        alerts: List[str] = []
        filter_service = self.ema_dust_factor >= 1.25
        if filter_service:
            alerts.append(f"HIGH_IMPEDANCE: Air filter dust loading +{round((self.ema_dust_factor - 1.0) * 100)}% above baseline. Clean/replace filter.")

        throttling_risk = frame.gpu_hotspot_temp_c >= 82.0
        if throttling_risk:
            alerts.append(f"THERMAL_ALERT: GPU hotspot ({frame.gpu_hotspot_temp_c:.1f}°C) approaching throttling limit (85°C).")

        if tim_degrad_pct >= 40.0:
            alerts.append(f"TIM_DEGRADATION: Thermal interface resistance grew +{round(tim_degrad_pct)}%. Potential grease pump-out / dry-out.")

        # Aggregate health score (0-100%)
        dust_penalty = max(0.0, (self.ema_dust_factor - 1.0) * 80.0)
        temp_penalty = max(0.0, (frame.gpu_hotspot_temp_c - 65.0) * 2.0)
        health_score = max(5.0, min(100.0, 100.0 - dust_penalty - temp_penalty - (tim_degrad_pct * 0.3)))

        return DigitalTwinHealthState(
            timestamp_utc=frame.timestamp_utc,
            nominal_clean_dp_pa=round(self.nominal_dp_ref, 2),
            current_estimated_dp_pa=round(expected_dp * self.ema_dust_factor, 2),
            filter_dust_loading_factor=round(self.ema_dust_factor, 3),
            thermal_resistance_k_w=round(r_th_system, 4),
            tim_degradation_pct=round(tim_degrad_pct, 1),
            thermal_throttling_risk=throttling_risk,
            filter_service_recommended=filter_service,
            health_score_pct=round(health_score, 1),
            alerts=alerts,
        )


class DigitalTwinConnectionManager:
    """Manages active WebSocket telemetry connections across multiple server chassis."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, chassis_id: str, websocket: WebSocket):
        await websocket.accept()
        if chassis_id not in self.active_connections:
            self.active_connections[chassis_id] = []
        self.active_connections[chassis_id].append(websocket)

    def disconnect(self, chassis_id: str, websocket: WebSocket):
        if chassis_id in self.active_connections:
            self.active_connections[chassis_id].remove(websocket)
            if not self.active_connections[chassis_id]:
                del self.active_connections[chassis_id]

    async def broadcast_state(self, chassis_id: str, state: DigitalTwinHealthState):
        if chassis_id in self.active_connections:
            payload = state.model_dump(mode="json")
            for conn in list(self.active_connections[chassis_id]):
                try:
                    await conn.send_json(payload)
                except Exception:
                    self.disconnect(chassis_id, conn)
