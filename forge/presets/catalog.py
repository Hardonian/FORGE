"""Curated hardware preset library for GPU/TPU/accelerator thermal conversions."""
from __future__ import annotations
from typing import Any, Dict, List
from forge.core.models import (
    CrossSectionShape,
    DuctMaterial,
    DuctSpec,
    PhysicsConditions,
    ServerFanStandard,
)


HARDWARE_PRESETS: Dict[str, Dict[str, Any]] = {
    "H100_PCIe_1U_SanAce40": {
        "description": "High-static 40mm 1U server fan to dual-slot NVIDIA H100/A100 PCIe rear intake shroud",
        "spec": DuctSpec(
            inlet_width=40.0,
            inlet_height=40.0,
            outlet_width=111.0,
            outlet_height=38.0,
            length=65.0,
            wall=2.0,
            corner_radius=3.5,
            inlet_shape=CrossSectionShape.CIRCULAR,
            outlet_shape=CrossSectionShape.ROUNDED_RECTANGLE,
            preserve_normal_thickness=True,
            guide_vanes=2,
            fan_standard=ServerFanStandard.FAN_40MM,
            flange_thickness=4.0,
            material=DuctMaterial.PC_CF,
        ),
        "conditions": PhysicsConditions(volumetric_flow_cfm=26.0, heat_dissipation_watts=350.0, air_temp_c=28.0),
    },
    "L40S_80mm_Server_Duct": {
        "description": "80mm rackmount fan to passive NVIDIA L40S / RTX 6000 Ada dual-slot heatsink adapter",
        "spec": DuctSpec(
            inlet_width=80.0,
            inlet_height=80.0,
            outlet_width=112.0,
            outlet_height=40.0,
            length=85.0,
            wall=2.2,
            offset_y=-12.0,
            corner_radius=5.0,
            inlet_shape=CrossSectionShape.CIRCULAR,
            outlet_shape=CrossSectionShape.ROUNDED_RECTANGLE,
            preserve_normal_thickness=True,
            guide_vanes=1,
            fan_standard=ServerFanStandard.FAN_80MM,
            flange_thickness=4.5,
            material=DuctMaterial.PC_CF,
        ),
        "conditions": PhysicsConditions(volumetric_flow_cfm=95.0, heat_dissipation_watts=350.0, air_temp_c=25.0),
    },
    "RTX_4090_120mm_Blower_Conversion": {
        "description": "120mm high-static industrial fan to 3-slot workstation GPU flow-through conversion shroud",
        "spec": DuctSpec(
            inlet_width=120.0,
            inlet_height=120.0,
            outlet_width=136.0,
            outlet_height=61.0,
            length=105.0,
            wall=2.4,
            offset_x=8.0,
            offset_y=-15.0,
            corner_radius=6.0,
            inlet_shape=CrossSectionShape.CIRCULAR,
            outlet_shape=CrossSectionShape.ROUNDED_RECTANGLE,
            preserve_normal_thickness=True,
            guide_vanes=2,
            fan_standard=ServerFanStandard.FAN_120MM,
            flange_thickness=5.0,
            material=DuctMaterial.PA12_CF,
        ),
        "conditions": PhysicsConditions(volumetric_flow_cfm=165.0, heat_dissipation_watts=450.0, air_temp_c=24.0),
    },
    "MI300X_OAM_120mm_HighFlow": {
        "description": "120mm high-RPM server fan to tall OAM accelerator heatsink with Stratford Bezier expansion",
        "spec": DuctSpec(
            inlet_width=120.0,
            inlet_height=120.0,
            outlet_width=155.0,
            outlet_height=88.0,
            length=110.0,
            wall=2.5,
            corner_radius=8.0,
            inlet_shape=CrossSectionShape.CIRCULAR,
            outlet_shape=CrossSectionShape.ROUNDED_RECTANGLE,
            preserve_normal_thickness=True,
            guide_vanes=2,
            width_profile_ctrl=[142.0, 152.0],
            height_profile_ctrl=[102.0, 91.0],
            fan_standard=ServerFanStandard.FAN_120MM,
            flange_thickness=5.0,
            material=DuctMaterial.PEEK,
        ),
        "conditions": PhysicsConditions(volumetric_flow_cfm=185.0, heat_dissipation_watts=750.0, air_temp_c=25.0),
    },
    "Quiet_Workstation_140mm_GPU": {
        "description": "Low-noise 140mm Noctua industrial fan contracting into a high-velocity GPU cooling tunnel",
        "spec": DuctSpec(
            inlet_width=140.0,
            inlet_height=140.0,
            outlet_width=115.0,
            outlet_height=45.0,
            length=115.0,
            wall=2.0,
            corner_radius=6.0,
            inlet_shape=CrossSectionShape.CIRCULAR,
            outlet_shape=CrossSectionShape.ROUNDED_RECTANGLE,
            preserve_normal_thickness=True,
            guide_vanes=0,
            fan_standard=ServerFanStandard.FAN_140MM,
            flange_thickness=4.0,
            material=DuctMaterial.ASA,
        ),
        "conditions": PhysicsConditions(volumetric_flow_cfm=155.0, heat_dissipation_watts=320.0, air_temp_c=22.0),
    },
}


def list_presets() -> List[Dict[str, Any]]:
    """Return metadata for all available hardware presets."""
    out: List[Dict[str, Any]] = []
    for name, entry in HARDWARE_PRESETS.items():
        spec: DuctSpec = entry["spec"]
        cond: PhysicsConditions = entry["conditions"]
        out.append({
            "name": name,
            "description": entry["description"],
            "spec": spec.model_dump(mode="json"),
            "conditions": cond.model_dump(mode="json"),
        })
    return out


def get_preset_spec(name: str) -> Dict[str, Any]:
    """Retrieve a specific preset by name (case-insensitive fallback)."""
    if name in HARDWARE_PRESETS:
        return HARDWARE_PRESETS[name]
    for k, v in HARDWARE_PRESETS.items():
        if k.lower() == name.lower():
            return v
    raise KeyError(f"Unknown hardware preset '{name}'. Available: {list(HARDWARE_PRESETS.keys())}")
