"""Hardware test bench telemetry schemas and data ingestion."""
from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field


class TestBenchTelemetry(BaseModel):
    """Real-world physical test bench measurement session."""
    __test__ = False
    session_id: str
    design_sha256: str
    timestamp: str
    fan_model: str = "Delta_PFB1212UHE"
    fan_pwm_percent: float = Field(..., ge=0, le=100)
    fan_rpm: float = Field(..., ge=0)
    ambient_temp_c: float
    measured_flow_cfm: float = Field(..., gt=0)
    inlet_static_pressure_pa: float
    outlet_static_pressure_pa: float
    measured_delta_p_pa: float
    thermocouple_temps_c: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None


class ValidationComparison(BaseModel):
    """Comparison between physics simulation and real-world test bench."""
    design_sha256: str
    sim_delta_p_pa: float
    measured_delta_p_pa: float
    delta_p_discrepancy_percent: float
    sim_outlet_temp_c: float
    measured_outlet_temp_c: Optional[float] = None
    status: str  # "MATCH", "DEVIATION_DETECTED", "CALIBRATION_RECOMMENDED"
