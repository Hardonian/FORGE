"""Deterministic natural-language-to-parametric-CAD parser and inverse-design synthesizer."""
from __future__ import annotations
import re
from typing import List, Optional
from pydantic import BaseModel
from forge.core.models import (
    CrossSectionShape,
    DFMReport,
    DuctMaterial,
    DuctSpec,
    ObstacleBox,
    PhysicsConditions,
    PhysicsMetrics,
    ServerFanStandard,
)
from forge.cad.dfm import analyze_dfm
from forge.cad.mesh import build_duct_mesh
from forge.cad.obstacles import (
    ObstacleClearanceReport,
    check_obstacle_clearances,
    route_duct_around_obstacles,
)
from forge.opt.bezier_diffuser import optimize_bezier_diffuser
from forge.physics.solver_1d import solve_duct_physics


class CopilotSynthesisResult(BaseModel):
    """End-to-end result synthesized from a natural language engineering prompt."""
    prompt: str
    spec: DuctSpec
    conditions: PhysicsConditions
    physics: PhysicsMetrics
    dfm: DFMReport
    clearance: ObstacleClearanceReport
    target_printer: str
    target_dp_pa: Optional[float] = None
    dp_target_met: bool = True
    reasoning_trace: List[str]


def synthesize_from_prompt(
    prompt: str,
    optimize_profile: bool = True,
) -> CopilotSynthesisResult:
    """Parse an engineering prompt and synthesize an optimized, DFM-verified, obstacle-routed DuctSpec."""
    p_low = prompt.lower()
    trace: List[str] = []

    # 1. Detect fan size / inlet dimensions
    fan_std = ServerFanStandard.FAN_120MM
    in_w, in_h = 120.0, 120.0
    for sz, enum_val in (
        (40, ServerFanStandard.FAN_40MM),
        (60, ServerFanStandard.FAN_60MM),
        (80, ServerFanStandard.FAN_80MM),
        (92, ServerFanStandard.FAN_92MM),
        (120, ServerFanStandard.FAN_120MM),
        (140, ServerFanStandard.FAN_140MM),
    ):
        if re.search(rf"\b{sz}\s*mm\b", p_low):
            fan_std = enum_val
            in_w, in_h = float(sz), float(sz)
            trace.append(f"Detected {sz}mm server fan standard (circular inlet {sz}x{sz} mm).")
            break
    else:
        trace.append("Defaulted to 120mm server fan inlet standard.")

    # 2. Detect target GPU / accelerator outlet geometry
    out_w, out_h = 112.0, 42.0
    length_mm = 95.0
    watts = 350.0
    if "4090" in p_low or "3-slot" in p_low or "three-slot" in p_low:
        out_w, out_h = 136.0, 61.0
        length_mm = 105.0
        watts = 450.0
        trace.append("Matched target accelerator: NVIDIA RTX 4090 3-slot profile (136x61 mm, 450W).")
    elif "h100" in p_low or "a100" in p_low or "1u" in p_low:
        out_w, out_h = 111.0, 38.0
        length_mm = 70.0
        watts = 350.0
        trace.append("Matched target accelerator: 1U/Dual-Slot PCIe H100/A100 shroud (111x38 mm, 350W).")
    elif "mi300" in p_low or "oam" in p_low:
        out_w, out_h = 155.0, 88.0
        length_mm = 110.0
        watts = 750.0
        trace.append("Matched target accelerator: AMD MI300X OAM module (155x88 mm, 750W).")
    elif "l40" in p_low or "6000" in p_low:
        out_w, out_h = 112.0, 40.0
        length_mm = 85.0
        watts = 350.0
        trace.append("Matched target accelerator: Dual-slot L40S / RTX 6000 Ada (112x40 mm, 350W).")

    # Override outlet dimensions if explicit WxH is present (e.g., "130x50mm")
    m_dim = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm", p_low)
    if m_dim:
        out_w = float(m_dim.group(1))
        out_h = float(m_dim.group(2))
        trace.append(f"Applied explicit cross-section dimension override: {out_w}x{out_h} mm.")

    # Override length if explicit "length N mm" is present
    m_len = re.search(r"length\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", p_low)
    if m_len:
        length_mm = float(m_len.group(1))
        trace.append(f"Applied explicit axial length: {length_mm} mm.")

    # 3. Detect material
    material = DuctMaterial.PC_CF
    for key, mat_enum in (
        ("peek", DuctMaterial.PEEK),
        ("pa12", DuctMaterial.PA12_CF),
        ("nylon", DuctMaterial.PA12_CF),
        ("pc-cf", DuctMaterial.PC_CF),
        ("polycarbonate", DuctMaterial.PC_CF),
        ("asa", DuctMaterial.ASA),
        ("abs", DuctMaterial.ABS),
        ("petg", DuctMaterial.PETG),
    ):
        if key in p_low:
            material = mat_enum
            trace.append(f"Selected engineering filament material: {material.value}.")
            break

    # 4. Detect airflow (CFM), heat load (W), and pressure drop budget (Pa)
    cfm = 140.0 if in_w >= 80 else 28.0
    m_cfm = re.search(r"(\d+(?:\.\d+)?)\s*cfm", p_low)
    if m_cfm:
        cfm = float(m_cfm.group(1))
        trace.append(f"Parsed target volumetric flow rate: {cfm} CFM.")

    m_watts = re.search(r"(\d+(?:\.\d+)?)\s*(?:w|watts)\b", p_low)
    if m_watts:
        watts = float(m_watts.group(1))
        trace.append(f"Parsed thermal dissipation load: {watts} W.")

    target_dp: Optional[float] = None
    m_dp = re.search(r"(?:under|below|max|<|less than)\s*(\d+(?:\.\d+)?)\s*pa\b", p_low)
    if m_dp:
        target_dp = float(m_dp.group(1))
        trace.append(f"Parsed maximum static pressure drop constraint: <= {target_dp} Pa.")

    # 5. Detect target 3D printer
    target_printer = "Bambu_Lab_X1C"
    if "prusa" in p_low:
        target_printer = "Prusa_MK4"
    elif "voron" in p_low:
        target_printer = "Voron_2.4_350"

    # 6. Detect obstacles (e.g., "bypass a 20mm RAM obstacle on the left")
    obstacles: List[ObstacleBox] = []
    if any(k in p_low for k in ("obstacle", "bypass", "ram", "cable", "pcie", "vrm")):
        obs_size = 18.0
        m_obs = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:ram|cable|pcie|vrm|obstacle)", p_low)
        if m_obs:
            obs_size = float(m_obs.group(1))
        z_mid = 0.5 * length_mm
        half_span = 0.25 * max(in_w, out_w)
        if "right" in p_low:
            obstacles.append(
                ObstacleBox(
                    name="Right_Chassis_Obstacle",
                    min_x=half_span - obs_size,
                    max_x=half_span + obs_size,
                    min_y=-25.0,
                    max_y=25.0,
                    min_z=max(5.0, z_mid - 12.0),
                    max_z=min(length_mm - 5.0, z_mid + 12.0),
                    clearance_mm=2.5,
                )
            )
        else:
            obstacles.append(
                ObstacleBox(
                    name="Left_RAM_Cable_Obstacle",
                    min_x=-half_span - obs_size,
                    max_x=-half_span + obs_size,
                    min_y=-25.0,
                    max_y=25.0,
                    min_z=max(5.0, z_mid - 12.0),
                    max_z=min(length_mm - 5.0, z_mid + 12.0),
                    clearance_mm=2.5,
                )
            )
        trace.append(f"Registered 3D obstacle '{obstacles[0].name}' ({obs_size}mm intrusion) for automatic path routing.")

    safe_corner_r = min(5.0, min(in_w, in_h, out_w, out_h) / 2.0 - 1.5)
    spec = DuctSpec(
        inlet_width=in_w,
        inlet_height=in_h,
        outlet_width=out_w,
        outlet_height=out_h,
        length=length_mm,
        wall=2.0,
        corner_radius=max(0.0, round(safe_corner_r, 2)),
        inlet_shape=CrossSectionShape.CIRCULAR,
        outlet_shape=CrossSectionShape.ROUNDED_RECTANGLE,
        preserve_normal_thickness=True,
        guide_vanes=1 if max(out_w / in_w, out_h / in_h) > 1.25 else 0,
        fan_standard=fan_std,
        flange_thickness=4.0,
        material=material,
        obstacles=obstacles,
    )
    conditions = PhysicsConditions(volumetric_flow_cfm=cfm, heat_dissipation_watts=watts)

    # 7. Route around 3D obstacles if present
    if spec.obstacles:
        spec, clearance = route_duct_around_obstacles(spec)
        trace.append(
            f"Synthesized Bezier centerline bypass: X_ctrl={spec.centerline_x_ctrl}, "
            f"all_clear={clearance.all_clear} (min margin {clearance.min_margin_mm} mm)."
        )
    else:
        clearance = check_obstacle_clearances(spec)

    # 8. Run Stratford Bezier diffuser optimization to minimize pressure loss
    if optimize_profile:
        spec, physics, opt_sum = optimize_bezier_diffuser(spec, conditions, num_control_points=2, iterations=18)
        trace.append(
            f"Optimized Stratford Bezier expansion profile: dP={physics.total_pressure_drop_pa} Pa "
            f"(uniformity={physics.flow_uniformity_index})."
        )
    else:
        physics = solve_duct_physics(spec, conditions)

    # 9. If target_dp is specified and current dP exceeds it, widen outlet or add guide vanes
    if target_dp is not None and physics.total_pressure_drop_pa > target_dp and spec.guide_vanes < 3:
        spec = spec.model_copy(update={"guide_vanes": min(4, spec.guide_vanes + 2)})
        physics = solve_duct_physics(spec, conditions)
        trace.append(f"Added internal splitter vanes (guide_vanes={spec.guide_vanes}) to reduce expansion loss.")

    tris = build_duct_mesh(spec)
    dfm = analyze_dfm(spec, tris, target_printer=target_printer)
    dp_met = True if target_dp is None else (physics.total_pressure_drop_pa <= target_dp)

    return CopilotSynthesisResult(
        prompt=prompt,
        spec=spec,
        conditions=conditions,
        physics=physics,
        dfm=dfm,
        clearance=clearance,
        target_printer=target_printer,
        target_dp_pa=target_dp,
        dp_target_met=dp_met,
        reasoning_trace=trace,
    )
