"""Core Telemetry and Digital Twin State Models."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field


class LiveTelemetryFrame(BaseModel):
    """Single incoming sensor frame from server chassis or test bench."""

    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fan_rpm: float = Field(..., gt=500.0, le=25000.0)
    fan_pwm_pct: float = Field(..., ge=10.0, le=100.0)
    inlet_ambient_temp_c: float = Field(default=25.0)
    exhaust_temp_c: float = Field(default=30.0)
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
