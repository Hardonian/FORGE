"""Pydantic v2 domain models for FORGE specifications, physics, and telemetry."""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class CrossSectionShape(str, Enum):
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    ROUNDED_RECTANGLE = "rounded_rectangle"


class ServerFanStandard(str, Enum):
    FAN_40MM = "40mm"
    FAN_60MM = "60mm"
    FAN_80MM = "80mm"
    FAN_92MM = "92mm"
    FAN_120MM = "120mm"
    FAN_140MM = "140mm"
    CUSTOM = "custom"


class DuctMaterial(str, Enum):
    PETG = "PETG"
    ABS = "ABS"
    ASA = "ASA"
    PC_CF = "PC-CF"       # Polycarbonate Carbon Fiber (High Temp, Datacenter Grade)
    PA12_CF = "PA12-CF"   # Nylon Carbon Fiber
    PEEK = "PEEK"         # Ultra-high temperature


# Material densities in g/cm^3 and Heat Deflection Temperatures (HDT at 0.45 MPa) in °C
MATERIAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "PETG": {"density": 1.27, "hdt_c": 70.0, "max_continuous_temp_c": 65.0},
    "ABS": {"density": 1.04, "hdt_c": 95.0, "max_continuous_temp_c": 85.0},
    "ASA": {"density": 1.05, "hdt_c": 98.0, "max_continuous_temp_c": 90.0},
    "PC-CF": {"density": 1.20, "hdt_c": 140.0, "max_continuous_temp_c": 125.0},
    "PA12-CF": {"density": 1.07, "hdt_c": 150.0, "max_continuous_temp_c": 130.0},
    "PEEK": {"density": 1.30, "hdt_c": 250.0, "max_continuous_temp_c": 220.0},
}


class DuctSpec(BaseModel):
    """Full parametric specification for transition cooling ducts."""
    inlet_width: float = Field(..., gt=0, description="Outer inlet width in mm")
    inlet_height: float = Field(..., gt=0, description="Outer inlet height in mm")
    outlet_width: float = Field(..., gt=0, description="Outer outlet width in mm")
    outlet_height: float = Field(..., gt=0, description="Outer outlet height in mm")
    length: float = Field(..., gt=0, description="Total axial length in mm")
    wall: float = Field(..., gt=0, description="Nominal wall thickness in mm")

    # Advanced parametric geometry
    offset_x: float = Field(0.0, description="Lateral offset (dx) between inlet and outlet in mm")
    offset_y: float = Field(0.0, description="Vertical offset (dy) between inlet and outlet in mm")
    corner_radius: float = Field(0.0, ge=0, description="Corner fillet radius in mm for internal aerodynamic smoothing")
    inlet_shape: CrossSectionShape = Field(CrossSectionShape.RECTANGULAR, description="Cross section shape at inlet")
    outlet_shape: CrossSectionShape = Field(CrossSectionShape.RECTANGULAR, description="Cross section shape at outlet")

    # Internal aerodynamic features
    guide_vanes: int = Field(0, ge=0, le=5, description="Number of internal splitter vanes to prevent flow separation")

    # Mounting flange integration
    fan_standard: Optional[ServerFanStandard] = Field(None, description="Optional server fan standard for inlet flange")
    flange_thickness: float = Field(0.0, ge=0, description="Mounting flange thickness in mm (0 for no flange)")

    # Material and manufacturing
    material: DuctMaterial = Field(DuctMaterial.PETG, description="3D printing filament material")

    @model_validator(mode="after")
    def check_clearance(self) -> "DuctSpec":
        min_dim = min(self.inlet_width, self.inlet_height, self.outlet_width, self.outlet_height)
        if self.wall * 2 >= min_dim:
            raise ValueError(f"Wall thickness {self.wall}mm is too large; leaves non-positive opening for min dimension {min_dim}mm")
        if self.corner_radius > 0:
            max_r = min(self.inlet_width, self.inlet_height, self.outlet_width, self.outlet_height) / 2.0
            if self.corner_radius >= max_r:
                raise ValueError(f"Corner radius {self.corner_radius}mm must be less than half of minimum dimension {max_r}mm")
        return self

    def to_canonical_dict(self) -> Dict[str, float]:
        """Return dict of core geometric fields for backward-compatible fingerprinting."""
        return {
            "inlet_height": float(self.inlet_height),
            "inlet_width": float(self.inlet_width),
            "length": float(self.length),
            "outlet_height": float(self.outlet_height),
            "outlet_width": float(self.outlet_width),
            "wall": float(self.wall),
        }


class PhysicsConditions(BaseModel):
    """Aerodynamic operating conditions for duct simulation."""
    volumetric_flow_cfm: float = Field(150.0, gt=0, description="Airflow rate in cubic feet per minute (CFM)")
    air_temp_c: float = Field(25.0, description="Inlet ambient air temperature in °C")
    air_pressure_pa: float = Field(101325.0, gt=0, description="Atmospheric pressure in Pa")
    air_density_kg_m3: float = Field(1.184, gt=0, description="Air density in kg/m^3")
    air_viscosity_pa_s: float = Field(1.85e-5, gt=0, description="Dynamic viscosity in Pa·s")
    heat_dissipation_watts: float = Field(350.0, ge=0, description="Target GPU heat load in Watts")


class PhysicsMetrics(BaseModel):
    """Calculated aerodynamic and thermal performance metrics."""
    flow_rate_m3_s: float
    inlet_velocity_m_s: float
    outlet_velocity_m_s: float
    reynolds_number: float
    hydraulic_diameter_mm: float
    friction_pressure_drop_pa: float
    expansion_loss_pa: float
    total_pressure_drop_pa: float
    flow_uniformity_index: float = Field(..., ge=0.0, le=1.0, description="Velocity uniformity index gamma (1.0 = perfectly uniform)")
    estimated_temp_rise_c: float


class DFMReport(BaseModel):
    """Design for manufacturing analysis."""
    material: str
    estimated_mass_grams: float
    estimated_print_time_hours: float
    overhang_compliance: bool
    max_overhang_angle_deg: float
    bed_fit: bool
    warnings: List[str] = Field(default_factory=list)
