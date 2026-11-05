"""FastAPI server providing REST endpoints and interactive Three.js 3D Web Studio."""
from __future__ import annotations
from typing import Dict, List, Optional
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from forge import __version__
from forge.core.models import DuctSpec, PhysicsConditions
from forge.core.hashes import sha256_hash
from forge.cad.mesh import build_duct_mesh, mesh_to_WebGL_buffers
from forge.cad.dfm import analyze_dfm
from forge.cad.exporters import (
    export_ascii_stl,
    export_binary_stl,
    export_3mf,
    export_step,
    export_glb,
)
from forge.cad.nurbs_step import export_exact_nurbs_step
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.heatsink import HeatsinkSpec, solve_coupled_duct_heatsink_system
from forge.physics.hybrid_cooling import HybridAALCSpec, solve_hybrid_aalc_cooling
from forge.physics.vectorized import evaluate_duct_specs_tensor_batch
from forge.api.websocket_twin import (
    DigitalTwinConnectionManager,
    LiveTelemetryFrame,
)
from forge.physics.network_solver import auto_balance_manifold, solve_manifold_network
from forge.physics.acoustics import design_acoustic_metamaterial_liner
from forge.cad.manifold import ManifoldSpec, build_manifold_mesh
from forge.opt.search_space import DuctSearchSpace
from forge.opt.optimizer import run_nsga2_optimization
from forge.opt.bezier_diffuser import optimize_bezier_diffuser
from forge.presets.catalog import get_preset_spec, list_presets
from forge.cad.obstacles import route_duct_around_obstacles
from forge.cad.slicer import export_production_slicer_3mf
from forge.copilot.agent import synthesize_from_prompt
from forge.topopt.darcy_solver import run_fluid_topology_optimization
from forge.surrogate.fno_field import predict_flow_field
from forge.cloud.tenant import CloudEntitlementManager
from forge.cloud.rack_twin import IPMITelemetryFrame, evaluate_datacenter_rack_roi
from forge.cad.cad_plugins import (
    export_freecad_macro,
    export_onshape_featurescript,
    export_openscad_script,
)
from forge.physics.structural_modal import analyze_structural_modal_resonance
from forge.physics.transient_thermal import simulate_transient_thermal_response
from forge.compliance.dossier import (
    export_compliance_dossier_markdown,
    generate_compliance_dossier,
)
from forge.interop.openusd import export_openusd_stage
from forge.interop.modelica import export_modelica_component
from forge.interop.open_schemas import export_all_json_schemas
from forge.cad.tpms import (
    TPMSSpec,
    build_tpms_lattice_mesh,
    evaluate_tpms_thermo_hydraulics,
)
from forge.opt.differentiable_adjoint import run_differentiable_adjoint_codesign
from forge.interop.calculix_fea import export_calculix_inp
from forge.interop.fmi_fmu import export_fmu_package
from forge.compliance.provenance import (
    generate_hardware_provenance_passport,
    verify_hardware_provenance,
)
import tempfile
from pathlib import Path

_ENTITLEMENT_MANAGER = CloudEntitlementManager()
_TWIN_CONNECTION_MANAGER = DigitalTwinConnectionManager()

app = FastAPI(
    title="FORGE Inverse-Design Platform API",
    version=__version__,
    description="Physics-informed generative inverse engineering platform for AI accelerator thermal management.",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": __version__, "platform": "FORGE"}


@app.post("/api/v1/evaluate")
def evaluate_duct(spec: DuctSpec, conditions: Optional[PhysicsConditions] = None):
    """Evaluate full geometry, DFM printability, and aerodynamic physics."""
    if conditions is None:
        conditions = PhysicsConditions()
    tris = build_duct_mesh(spec)
    dfm = analyze_dfm(spec, tris)
    physics = solve_duct_physics(spec, conditions)
    fingerprint = sha256_hash(spec.to_canonical_dict())

    return {
        "geometry_sha256": fingerprint,
        "spec": spec.model_dump(),
        "physics": physics.model_dump(),
        "dfm": dfm.model_dump(),
    }


@app.post("/api/v1/mesh")
def generate_webgl_mesh(spec: DuctSpec, conditions: Optional[PhysicsConditions] = None):
    """Return Three.js BufferGeometry float arrays alongside live physics and DFM telemetry."""
    if conditions is None:
        conditions = PhysicsConditions()
    tris = build_duct_mesh(spec)
    buffers = mesh_to_WebGL_buffers(tris)
    dfm = analyze_dfm(spec, tris)
    physics = solve_duct_physics(spec, conditions)
    fingerprint = sha256_hash(spec.to_canonical_dict())

    return {
        "geometry_sha256": fingerprint,
        "mesh": buffers,
        "physics": physics.model_dump(),
        "dfm": dfm.model_dump(),
    }


@app.post("/api/v1/optimize")
def optimize_endpoint(
    space: Optional[DuctSearchSpace] = None,
    conditions: Optional[PhysicsConditions] = None,
    population_size: int = 16,
    generations: int = 3,
):
    """Execute multi-objective NSGA-II Pareto optimization and return frontier designs."""
    if space is None:
        space = DuctSearchSpace()
    if conditions is None:
        conditions = PhysicsConditions()
    pareto = run_nsga2_optimization(space, conditions, population_size=population_size, generations=generations)
    return {
        "pareto_count": len(pareto),
        "candidates": [
            {
                "spec": c.spec.model_dump(),
                "physics": c.physics.model_dump(),
                "mass_grams": c.mass_grams,
                "crowding_dist": None if c.crowding_dist == float("inf") else round(c.crowding_dist, 4),
            }
            for c in pareto
        ],
    }


@app.get("/api/v1/presets")
def presets_endpoint():
    """Return all curated datacenter and workstation AI accelerator presets."""
    return {"presets": list_presets()}


@app.get("/api/v1/presets/{preset_name}")
def preset_detail_endpoint(preset_name: str):
    """Retrieve a specific hardware preset by name."""
    entry = get_preset_spec(preset_name)
    return {
        "name": preset_name,
        "description": entry["description"],
        "spec": entry["spec"].model_dump(mode="json"),
        "conditions": entry["conditions"].model_dump(mode="json"),
    }


@app.post("/api/v1/heatsink")
def coupled_heatsink_endpoint(
    duct: DuctSpec,
    heatsink: Optional[HeatsinkSpec] = None,
    conditions: Optional[PhysicsConditions] = None,
):
    """Solve coupled Fan + Duct + Finned Heatsink conjugate heat transfer and equilibrium flow."""
    if heatsink is None:
        heatsink = HeatsinkSpec(width_mm=duct.outlet_width, fin_height_mm=max(10.0, duct.outlet_height - 4.0))
    if conditions is None:
        conditions = PhysicsConditions()
    report = solve_coupled_duct_heatsink_system(duct, heatsink, conditions)
    return report.model_dump()


@app.post("/api/v1/manifold")
def manifold_endpoint(
    spec: ManifoldSpec,
    total_flow_cfm: float = 180.0,
    auto_balance: bool = False,
):
    """Solve parallel hydraulic and thermal distribution across a 1-to-N splitter manifold."""
    if auto_balance:
        opt_spec, report, summary = auto_balance_manifold(spec, total_flow_cfm)
        tris = build_manifold_mesh(opt_spec)
        return {
            "manifold_spec": opt_spec.model_dump(mode="json"),
            "network_report": report.model_dump(),
            "balance_summary": summary,
            "triangle_count": len(tris),
        }
    report = solve_manifold_network(spec, total_flow_cfm)
    tris = build_manifold_mesh(spec)
    return {
        "manifold_spec": spec.model_dump(mode="json"),
        "network_report": report.model_dump(),
        "triangle_count": len(tris),
    }


@app.post("/api/v1/bezier-optimize")
def bezier_optimize_endpoint(
    spec: DuctSpec,
    conditions: Optional[PhysicsConditions] = None,
    num_control_points: int = 2,
    iterations: int = 25,
):
    """Synthesize a non-linear Stratford / bell-mouth Bezier diffuser expansion profile."""
    if conditions is None:
        conditions = PhysicsConditions()
    opt_spec, opt_metrics, summary = optimize_bezier_diffuser(
        spec, conditions, num_control_points=num_control_points, iterations=iterations
    )
    return {
        "optimized_spec": opt_spec.model_dump(mode="json"),
        "optimized_physics": opt_metrics.model_dump(),
        "summary": summary,
    }


@app.post("/api/v1/acoustics")
def acoustic_liner_endpoint(
    fan_rpm: int = 8500,
    blade_count: int = 7,
    air_temp_c: float = 30.0,
    max_cavity_depth_mm: float = 10.0,
):
    """Design a tuned sub-wavelength Helmholtz / quarter-wave acoustic metamaterial liner for fan BPF suppression."""
    design = design_acoustic_metamaterial_liner(
        fan_rpm=fan_rpm,
        blade_count=blade_count,
        air_temp_c=air_temp_c,
        max_cavity_depth_mm=max_cavity_depth_mm,
    )
    return design.model_dump()


@app.post("/api/v1/copilot")
def copilot_endpoint(prompt: str, optimize_profile: bool = True):
    """Natural language prompt-to-CAD inverse design synthesizer."""
    res = synthesize_from_prompt(prompt, optimize_profile=optimize_profile)
    return res.model_dump(mode="json")


@app.post("/api/v1/obstacles/route")
def route_obstacles_endpoint(spec: DuctSpec):
    """Evaluate 3D obstacle clearances and automatically synthesize Bezier bypass control points."""
    routed_spec, report = route_duct_around_obstacles(spec)
    return {
        "routed_spec": routed_spec.model_dump(mode="json"),
        "clearance_report": report.model_dump(),
    }


@app.post("/api/v1/topopt")
def topopt_endpoint(
    spec: DuctSpec,
    conditions: Optional[PhysicsConditions] = None,
    nz: int = 20,
    nx: int = 14,
    iterations: int = 15,
):
    """Run Navier-Stokes Brinkman-Darcy fluid topology optimization and extract organic 3D-printable geometry."""
    result, tris = run_fluid_topology_optimization(
        spec, conditions=conditions, nz=nz, nx=nx, iterations=iterations
    )
    return {
        "topopt_result": result.model_dump(mode="json"),
        "triangle_count": len(tris),
    }


@app.post("/api/v1/flow-field")
def flow_field_endpoint(
    spec: DuctSpec,
    conditions: Optional[PhysicsConditions] = None,
    nz: int = 20,
    nx: int = 16,
):
    """Predict full 2D/3D spatial velocity, static pressure, and separation risk fields via FNO surrogate."""
    pred = predict_flow_field(spec, conditions=conditions, nz=nz, nx=nx)
    return pred.model_dump()


@app.post("/api/v1/enterprise/rack-roi")
def enterprise_rack_roi_endpoint(
    spec: DuctSpec,
    ipmi: Optional[IPMITelemetryFrame] = None,
    conditions: Optional[PhysicsConditions] = None,
):
    """Evaluate enterprise datacenter rack fan power savings, hotspot reduction, and financial payback."""
    rep = evaluate_datacenter_rack_roi(spec, ipmi=ipmi, conditions=conditions)
    return rep.model_dump()


@app.get("/api/v1/cloud/billing/{tenant_id}")
def cloud_billing_endpoint(tenant_id: str):
    """Return multi-tenant SaaS subscription quota, compute credits, and billing summary."""
    return _ENTITLEMENT_MANAGER.get_billing_summary(tenant_id)


@app.post("/api/v1/modal")
def modal_resonance_endpoint(
    spec: DuctSpec,
    conditions: Optional[PhysicsConditions] = None,
    fan_rpm: int = 9000,
    blade_count: int = 7,
):
    """Analyze temperature-softened structural natural frequencies and fan harmonic resonance margins."""
    rep = analyze_structural_modal_resonance(spec, conditions=conditions, fan_rpm=fan_rpm, blade_count=blade_count)
    return rep.model_dump()


@app.post("/api/v1/transient")
def transient_thermal_endpoint(
    spec: DuctSpec,
    idle_watts: float = 65.0,
    burst_watts: float = 450.0,
    duration_s: float = 20.0,
):
    """Simulate time-domain transient die temperature and closed-loop PID fan PWM response under burst TDP."""
    rep = simulate_transient_thermal_response(
        spec, idle_watts=idle_watts, burst_watts=burst_watts, duration_s=duration_s
    )
    return rep.model_dump()


@app.post("/api/v1/compliance")
def compliance_dossier_endpoint(
    spec: DuctSpec,
    conditions: Optional[PhysicsConditions] = None,
    fan_rpm: int = 9000,
):
    """Generate an enterprise UL94 / ASHRAE / NEBS hardware qualification dossier and Markdown certificate."""
    dossier = generate_compliance_dossier(spec, conditions=conditions, fan_rpm=fan_rpm)
    return {
        "dossier": dossier.model_dump(mode="json"),
        "markdown_certificate": export_compliance_dossier_markdown(dossier),
    }


@app.post("/api/v1/export/cad-script")
def export_cad_script_endpoint(spec: DuctSpec, target_cad: str = "onshape"):
    """Export native parametric CAD code for Onshape (FeatureScript), FreeCAD (.FCMacro), or OpenSCAD (.scad)."""
    t = target_cad.lower()
    if t == "onshape":
        code = export_onshape_featurescript(spec)
        return {"target_cad": "onshape", "filename": "forge_duct.fs", "script": code}
    elif t == "freecad":
        code = export_freecad_macro(spec)
        return {"target_cad": "freecad", "filename": "forge_duct.FCMacro", "script": code}
    elif t == "openscad":
        code = export_openscad_script(spec)
        return {"target_cad": "openscad", "filename": "forge_duct.scad", "script": code}
    raise ValueError(f"Unsupported target_cad '{target_cad}'. Choose onshape, freecad, or openscad.")


@app.post("/api/v1/export/slicer-3mf")
def download_slicer_3mf(
    spec: DuctSpec,
    printer_profile: str = "Bambu_Lab_X1C",
    auto_split_if_oversized: bool = True,
):
    """Generate and download production-ready Bambu/Prusa 3MF package with embedded filament & modifier settings."""
    content = export_production_slicer_3mf(
        spec,
        printer_profile=printer_profile,
        auto_split_if_oversized=auto_split_if_oversized,
    )
    return Response(
        content=content,
        media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        headers={"Content-Disposition": "attachment; filename=forge_production_slicer.3mf"},
    )


@app.post("/api/v1/export/stl")
def download_stl(spec: DuctSpec, binary: bool = True):
    """Generate and download STL mesh."""
    tris = build_duct_mesh(spec)
    if binary:
        content = export_binary_stl(tris)
        return Response(content=content, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=duct.stl"})
    else:
        content = export_ascii_stl(tris)
        return Response(content=content, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=duct.stl"})


@app.post("/api/v1/export/3mf")
def download_3mf(spec: DuctSpec):
    """Generate and download 3MF manufacturing container."""
    tris = build_duct_mesh(spec)
    content = export_3mf(tris)
    return Response(content=content, media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml", headers={"Content-Disposition": "attachment; filename=duct.3mf"})


@app.post("/api/v1/export/step")
def download_step(spec: DuctSpec):
    """Generate and download ISO-10303-21 STEP CAD file."""
    tris = build_duct_mesh(spec)
    content = export_step(tris)
    return Response(content=content, media_type="application/step", headers={"Content-Disposition": "attachment; filename=duct.stp"})


@app.post("/api/v1/export/glb")
def download_glb(spec: DuctSpec):
    """Generate and download glTF 2.0 binary (.glb) 3D model."""
    tris = build_duct_mesh(spec)
    content = export_glb(tris)
    return Response(content=content, media_type="model/gltf-binary", headers={"Content-Disposition": "attachment; filename=duct.glb"})


@app.post("/api/v1/export/openusd")
def download_openusd(spec: DuctSpec, conditions: Optional[PhysicsConditions] = None):
    """Export NVIDIA Omniverse / Pixar OpenUSD (.usda) scene with embedded physics digital twin attributes."""
    usda = export_openusd_stage(spec, conditions=conditions)
    return Response(content=usda, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=forge_twin.usda"})


@app.post("/api/v1/export/modelica")
def download_modelica(spec: DuctSpec, conditions: Optional[PhysicsConditions] = None):
    """Export Modelica (.mo) 1D system-level thermo-fluid component."""
    mo = export_modelica_component(spec, conditions=conditions)
    return Response(content=mo, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=FORGE_CoolingDuct.mo"})


@app.post("/api/v1/mcp")
def mcp_jsonrpc_endpoint(request: Dict):
    """Model Context Protocol (MCP) JSON-RPC 2.0 server endpoint for AI agent tool discovery and invocation."""
    from forge.interop.mcp_server import handle_mcp_jsonrpc_request

    return handle_mcp_jsonrpc_request(request)


@app.get("/api/v1/tools/schema")
def openai_tools_schema_endpoint():
    """Return OpenAI / Anthropic / Gemini function-calling tool schemas for FORGE."""
    from forge.interop.mcp_server import get_openai_tool_schemas

    return {"tools": get_openai_tool_schemas()}


@app.get("/api/v1/schemas")
def open_json_schemas_endpoint():
    """Return Draft 2020-12 JSON Schemas for all FORGE domain models."""
    return export_all_json_schemas()


@app.post("/api/v1/tpms")
def tpms_lattice_endpoint(spec: TPMSSpec, flow_cfm: float = 120.0):
    """Evaluate TPMS (Gyroid, Schwarz-P, Diamond) metamaterial lattice thermo-hydraulics and printability."""
    rep = evaluate_tpms_thermo_hydraulics(spec, flow_cfm=flow_cfm)
    return rep.model_dump()


@app.post("/api/v1/tpms/mesh")
def tpms_webgl_mesh(spec: TPMSSpec, flow_cfm: float = 120.0):
    """Return Three.js BufferGeometry float arrays and thermo-hydraulics for a TPMS metamaterial core."""
    tris = build_tpms_lattice_mesh(spec, grid_nx=12, grid_ny=10, grid_nz=10)
    buffers = mesh_to_WebGL_buffers(tris)
    rep = evaluate_tpms_thermo_hydraulics(spec, flow_cfm=flow_cfm)
    return {
        "mesh": buffers,
        "report": rep.model_dump(),
    }


@app.post("/api/v1/opt/adjoint")
def differentiable_adjoint_endpoint(
    spec: DuctSpec,
    heatsink: Optional[HeatsinkSpec] = None,
    conditions: Optional[PhysicsConditions] = None,
    steps: int = 40,
):
    """Run PyTorch end-to-end differentiable adjoint co-design of duct loft and finned heatsink."""
    res = run_differentiable_adjoint_codesign(
        spec, heatsink=heatsink, conditions=conditions, steps=steps
    )
    return res.model_dump(mode="json")


@app.post("/api/v1/export/calculix")
def download_calculix_fea(spec: DuctSpec, conditions: Optional[PhysicsConditions] = None):
    """Export CalculiX / Abaqus (.inp) 3D structural, thermal, and modal finite element analysis deck."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = export_calculix_inp(spec, Path(tmp) / "forge_duct.inp", conditions=conditions)
        content = out_path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=forge_duct.inp"},
    )


@app.post("/api/v1/export/fmu")
def download_fmu_package(spec: DuctSpec, conditions: Optional[PhysicsConditions] = None):
    """Export FMI 2.0 Co-Simulation and Model Exchange (.fmu) digital twin package."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = export_fmu_package(spec, Path(tmp) / "forge_duct.fmu", conditions=conditions)
        content = out_path.read_bytes()
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=forge_duct.fmu"},
    )


@app.post("/api/v1/compliance/provenance")
def provenance_passport_endpoint(
    spec: DuctSpec,
    conditions: Optional[PhysicsConditions] = None,
):
    """Generate a cryptographic Hardware Bill of Materials (HBOM) and SLSA provenance passport."""
    passport = generate_hardware_provenance_passport(spec, conditions=conditions)
    return {
        "passport": passport.model_dump(mode="json"),
        "signature_verified": verify_hardware_provenance(passport),
    }


@app.post("/api/v1/export/step-nurbs")
def download_step_nurbs(spec: DuctSpec):
    """Generate and download exact ISO-10303-21 STEP AP242 NURBS B-spline surface CAD file."""
    content = export_exact_nurbs_step(spec)
    return Response(
        content=content,
        media_type="application/step",
        headers={"Content-Disposition": "attachment; filename=duct_nurbs_ap242.stp"},
    )


@app.post("/api/v1/hybrid-aalc")
def hybrid_aalc_endpoint(
    spec: DuctSpec,
    aalc_spec: Optional[HybridAALCSpec] = None,
    conditions: Optional[PhysicsConditions] = None,
):
    """Evaluate coupled two-phase vapor chamber and heat pipe hybrid cooling (AALC) limits."""
    rep = solve_hybrid_aalc_cooling(spec, aalc_spec=aalc_spec, conditions=conditions)
    return rep.model_dump()


@app.post("/api/v1/batch-evaluate")
def batch_evaluate_endpoint(
    specs: List[DuctSpec],
    conditions: Optional[PhysicsConditions] = None,
):
    """Evaluate a batch of candidate duct specifications in parallel via vectorized PyTorch tensor SIMD."""
    batch_res = evaluate_duct_specs_tensor_batch(specs, conditions=conditions)
    return {
        "count": len(specs),
        "total_pressure_drop_pa": [round(float(v), 2) for v in batch_res.total_pressure_drop_pa],
        "velocity_uniformity_index": [round(float(v), 4) for v in batch_res.velocity_uniformity_index],
        "reynolds_number": [round(float(v), 1) for v in batch_res.reynolds_number],
        "estimated_mass_grams": [round(float(v), 1) for v in batch_res.estimated_mass_grams],
    }


@app.websocket("/ws/telemetry/{chassis_id}")
async def digital_twin_websocket(websocket: WebSocket, chassis_id: str):
    """Bidirectional real-time digital twin streaming endpoint for IPMI/BMC sensor telemetry."""
    await _TWIN_CONNECTION_MANAGER.connect(chassis_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            frame = LiveTelemetryFrame(**data)
            health = _TWIN_CONNECTION_MANAGER.process_telemetry(chassis_id, frame)
            await websocket.send_json(health.model_dump())
    except WebSocketDisconnect:
        _TWIN_CONNECTION_MANAGER.disconnect(chassis_id, websocket)


@app.get("/viewer", response_class=HTMLResponse)
def interactive_3d_viewer():
    """Flagship Three.js 3D Web Studio 2.0 with AI Copilot, Presets, TopOpt, Heatmap, and Universal Open Exporters."""
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FORGE 3D Web Studio</title>
  <style>
    body { margin: 0; overflow: hidden; background: #0b1120; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    #copilot-bar { position: absolute; top: 14px; left: 50%; transform: translateX(-50%); display: flex; gap: 8px; background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(12px); padding: 10px 14px; border-radius: 10px; border: 1px solid #38bdf8; width: 560px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); z-index: 10; }
    #copilot-input { flex: 1; background: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 7px 10px; font-size: 12px; }
    #ui { position: absolute; top: 76px; left: 16px; background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(12px); padding: 16px; border-radius: 10px; border: 1px solid #334155; width: 315px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    #telemetry { position: absolute; top: 76px; right: 16px; background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(12px); padding: 16px; border-radius: 10px; border: 1px solid #334155; width: 305px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h2 { margin: 0 0 8px 0; font-size: 15px; color: #38bdf8; display: flex; justify-content: space-between; align-items: center; }
    label { display: flex; justify-content: space-between; font-size: 12px; margin-top: 8px; color: #cbd5e1; }
    input[type=range] { width: 100%; accent-color: #38bdf8; cursor: pointer; }
    select { width: 100%; margin-top: 4px; padding: 5px; background: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 4px; font-size: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; background: #0284c7; color: #fff; font-size: 11px; }
    .metric-row { display: flex; justify-content: space-between; font-size: 12px; padding: 5px 0; border-bottom: 1px solid #1e293b; }
    .metric-val { font-weight: 600; color: #34d399; }
    .btn-row { display: flex; gap: 6px; margin-top: 10px; }
    button { flex: 1; padding: 7px 9px; background: #2563eb; color: #fff; border: none; border-radius: 5px; font-size: 11px; cursor: pointer; font-weight: 600; }
    button:hover { background: #1d4ed8; }
    button.secondary { background: #0f766e; }
    button.secondary:hover { background: #115e59; }
    #trace-box { margin-top: 10px; font-size: 11px; color: #94a3b8; max-height: 75px; overflow-y: auto; background: #0f172a; padding: 6px; border-radius: 4px; border: 1px solid #1e293b; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
  <div id="copilot-bar">
    <input id="copilot-input" type="text" value="80mm Delta fan to RTX 4090, bypass 15mm RAM obstacle on left, PC-CF" placeholder="Describe cooling duct requirements in natural language..." />
    <button id="btn_copilot" style="flex: 0 0 130px;">AI Copilot Synthesize</button>
  </div>

  <div id="ui">
    <h2>FORGE 3D Studio <span class="badge">v0.2.0 Open</span></h2>
    <label>Hardware Accelerator Preset</label>
    <select id="sel_preset">
      <option value="">-- Custom Parametric --</option>
      <option value="H100_PCIe_1U_SanAce40">NVIDIA H100 PCIe 1U (40mm SanAce)</option>
      <option value="RTX_4090_120mm_Blower_Conversion">NVIDIA RTX 4090 (120mm Conversion)</option>
      <option value="L40S_80mm_Server_Duct">NVIDIA L40S / 6000 Ada (80mm Server)</option>
      <option value="MI300X_OAM_120mm_HighFlow">AMD MI300X OAM (120mm Stratford)</option>
      <option value="Quiet_Workstation_140mm_GPU">Quiet Workstation (140mm Noctua)</option>
    </select>
    <label>Inlet Shape</label>
    <select id="sel_in_shape">
      <option value="circular">Circular (Server Fan)</option>
      <option value="rectangular">Rectangular</option>
      <option value="rounded_rectangle">Rounded Rectangle</option>
    </select>
    <label>Length (mm) <span id="lbl_l">105</span></label>
    <input type="range" id="rng_l" min="50" max="220" value="105">
    <label>Offset X (mm) <span id="lbl_ox">10</span></label>
    <input type="range" id="rng_ox" min="-45" max="45" value="10">
    <label>Corner Fillet (mm) <span id="lbl_cr">6</span></label>
    <input type="range" id="rng_cr" min="0" max="18" value="6">
    <label>Splitter Guide Vanes <span id="lbl_gv">1</span></label>
    <input type="range" id="rng_gv" min="0" max="4" value="1">
    <div class="btn-row">
      <button id="btn_bezier" class="secondary">Stratford Bezier</button>
      <button id="btn_topopt" class="secondary">Darcy TopOpt</button>
      <button id="btn_adjoint" class="secondary">PyTorch Adjoint</button>
    </div>
    <div class="btn-row">
      <button id="btn_tpms" class="secondary">Gyroid TPMS Core</button>
      <button id="btn_passport" class="secondary">SLSA Provenance</button>
    </div>
    <div class="btn-row">
      <button id="btn_wire">Wireframe</button>
      <button id="btn_heatmap">Velocity Heatmap</button>
    </div>
    <div id="trace-box">Ready. Use sliders, presets, or the AI Copilot bar above.</div>
  </div>

  <div id="telemetry">
    <h2>Live Multi-Physics & Open Export</h2>
    <div class="metric-row"><span>Total Pressure Drop (dP)</span><span class="metric-val" id="m_dp">-- Pa</span></div>
    <div class="metric-row"><span>Flow Uniformity (gamma)</span><span class="metric-val" id="m_uni">--</span></div>
    <div class="metric-row"><span>Reynolds Number</span><span class="metric-val" id="m_re">--</span></div>
    <div class="metric-row"><span>Aeroacoustic Noise</span><span class="metric-val" id="m_db">-- dBA</span></div>
    <div class="metric-row"><span>Wall Deflection</span><span class="metric-val" id="m_def">-- um</span></div>
    <div class="metric-row"><span>Fan Operating Point</span><span class="metric-val" id="m_cfm">-- CFM</span></div>
    <div class="metric-row"><span>Solid Mass (PC-CF)</span><span class="metric-val" id="m_mass">-- g</span></div>
    <div class="metric-row"><span>Manifold Triangles</span><span class="metric-val" id="m_tris">--</span></div>
    <div class="btn-row">
      <button id="btn_dl_3mf">Slicer .3MF</button>
      <button id="btn_dl_glb">WebGL .GLB</button>
      <button id="btn_dl_usda">Omniverse .USDA</button>
    </div>
    <div class="btn-row">
      <button id="btn_dl_inp" class="secondary">CalculiX .INP FEA</button>
      <button id="btn_dl_fmu" class="secondary">FMI 2.0 .FMU Twin</button>
    </div>
  </div>

  <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const light1 = new THREE.DirectionalLight(0xffffff, 1.15);
    light1.position.set(130, 180, 160);
    scene.add(light1);
    const light2 = new THREE.DirectionalLight(0x38bdf8, 0.55);
    light2.position.set(-120, -100, -80);
    scene.add(light2);
    scene.add(new THREE.AmbientLight(0x404040, 1.5));

    const grid = new THREE.GridHelper(300, 30, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    camera.position.set(175, -165, 150);
    camera.lookAt(0, 0, 55);

    let mesh = null;
    let wireframeMode = false;
    let heatmapMode = false;
    let currentSpec = {
      inlet_width: 120,
      inlet_height: 120,
      outlet_width: 136,
      outlet_height: 61,
      length: 105,
      wall: 2.2,
      offset_x: 10,
      offset_y: 0.0,
      corner_radius: 6,
      inlet_shape: "circular",
      outlet_shape: "rounded_rectangle",
      preserve_normal_thickness: true,
      guide_vanes: 1,
      material: "PC-CF"
    };

    function syncSpecFromSliders() {
      currentSpec.length = parseFloat(document.getElementById('rng_l').value);
      currentSpec.offset_x = parseFloat(document.getElementById('rng_ox').value);
      currentSpec.corner_radius = parseFloat(document.getElementById('rng_cr').value);
      currentSpec.inlet_shape = document.getElementById('sel_in_shape').value;
      currentSpec.guide_vanes = parseInt(document.getElementById('rng_gv').value);
    }

    async function renderSpec(specObj) {
      currentSpec = specObj;
      const res = await fetch('/api/v1/mesh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: currentSpec })
      });
      const data = await res.json();

      if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
      }

      const geom = new THREE.BufferGeometry();
      const positions = new Float32Array(data.mesh.positions);
      const normals = new Float32Array(data.mesh.normals);
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

      if (heatmapMode) {
        const colors = new Float32Array(positions.length);
        const maxZ = Math.max(1.0, currentSpec.length);
        for (let i = 0; i < positions.length; i += 3) {
          const frac = Math.min(1.0, Math.max(0.0, positions[i + 2] / maxZ));
          colors[i] = 0.15 + 0.80 * (1.0 - frac);
          colors[i + 1] = 0.75 * Math.sin(Math.PI * frac);
          colors[i + 2] = 0.25 + 0.75 * frac;
        }
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      }

      const mat = new THREE.MeshStandardMaterial({
        color: heatmapMode ? 0xffffff : 0x38bdf8,
        vertexColors: heatmapMode,
        roughness: 0.32,
        metalness: 0.18,
        side: THREE.DoubleSide,
        wireframe: wireframeMode
      });
      mesh = new THREE.Mesh(geom, mat);
      scene.add(mesh);

      document.getElementById('m_dp').innerText = data.physics.total_pressure_drop_pa + ' Pa';
      document.getElementById('m_uni').innerText = data.physics.flow_uniformity_index;
      document.getElementById('m_re').innerText = data.physics.reynolds_number.toLocaleString();
      document.getElementById('m_db').innerText = data.physics.acoustic_noise_dba + ' dBA';
      document.getElementById('m_def').innerText = data.physics.max_wall_deflection_um + ' um';
      document.getElementById('m_cfm').innerText = data.physics.fan_operating_cfm + ' CFM';
      document.getElementById('m_mass').innerText = data.dfm.estimated_mass_grams + ' g';
      document.getElementById('m_tris').innerText = data.mesh.triangle_count;
    }

    ['rng_l', 'rng_ox', 'rng_cr', 'rng_gv', 'sel_in_shape'].forEach(id => {
      document.getElementById(id).oninput = (e) => {
        const map = { rng_l: 'lbl_l', rng_ox: 'lbl_ox', rng_cr: 'lbl_cr', rng_gv: 'lbl_gv' };
        if (map[id]) document.getElementById(map[id]).innerText = e.target.value;
        syncSpecFromSliders();
        renderSpec(currentSpec);
      };
    });

    document.getElementById('sel_preset').onchange = async (e) => {
      if (!e.target.value) return;
      const r = await fetch('/api/v1/presets/' + e.target.value);
      const d = await r.json();
      document.getElementById('trace-box').innerText = 'Loaded Preset: ' + d.description;
      await renderSpec(d.spec);
    };

    document.getElementById('btn_copilot').onclick = async () => {
      const prompt = document.getElementById('copilot-input').value;
      document.getElementById('trace-box').innerText = 'Synthesizing via AI Copilot...';
      const r = await fetch('/api/v1/copilot?prompt=' + encodeURIComponent(prompt), { method: 'POST' });
      const d = await r.json();
      document.getElementById('trace-box').innerText = d.reasoning_trace.join(' | ');
      await renderSpec(d.spec);
    };

    document.getElementById('btn_bezier').onclick = async () => {
      document.getElementById('trace-box').innerText = 'Running Stratford Bezier Diffuser Optimization...';
      const r = await fetch('/api/v1/bezier-optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: currentSpec })
      });
      const d = await r.json();
      document.getElementById('trace-box').innerText = 'Bezier Opt: dP reduced by ' + d.summary.dp_reduction_pct + '%';
      await renderSpec(d.optimized_spec);
    };

    document.getElementById('btn_topopt').onclick = async () => {
      document.getElementById('trace-box').innerText = 'Running Brinkman-Darcy Fluid Topology Optimization...';
      const r = await fetch('/api/v1/topopt?iterations=12', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: currentSpec })
      });
      const d = await r.json();
      document.getElementById('trace-box').innerText = 'Darcy TopOpt: dissipation -' + d.topopt_result.dissipation_reduction_pct + '%';
      await renderSpec(d.topopt_result.optimized_spec);
    };

    document.getElementById('btn_adjoint').onclick = async () => {
      document.getElementById('trace-box').innerText = 'Running PyTorch Differentiable Adjoint Co-Design...';
      const r = await fetch('/api/v1/opt/adjoint?steps=35', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: currentSpec })
      });
      const d = await r.json();
      document.getElementById('trace-box').innerText = 'PyTorch Adjoint: dP -' + d.pressure_drop_reduction_pct + '%, Hotspot -' + d.hotspot_reduction_c + ' C';
      await renderSpec(d.optimized_spec);
    };

    document.getElementById('btn_tpms').onclick = async () => {
      document.getElementById('trace-box').innerText = 'Generating 3D Gyroid TPMS Metamaterial Core...';
      const r = await fetch('/api/v1/tpms/mesh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spec: {
            lattice_type: 'gyroid',
            width_mm: currentSpec.outlet_width || 120,
            height_mm: currentSpec.outlet_height || 60,
            length_mm: 32.0,
            unit_cell_mm: 8.0,
            porosity_inlet: 0.74,
            porosity_outlet: 0.64
          },
          flow_cfm: 120.0
        })
      });
      const d = await r.json();
      if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(d.mesh.positions), 3));
      geom.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(d.mesh.normals), 3));
      mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.28, metalness: 0.25, side: THREE.DoubleSide }));
      scene.add(mesh);
      document.getElementById('m_dp').innerText = d.report.pressure_drop_pa + ' Pa';
      document.getElementById('m_uni').innerText = 'TPMS Core';
      document.getElementById('m_re').innerText = d.report.reynolds_number.toLocaleString();
      document.getElementById('m_mass').innerText = d.report.total_wetted_area_cm2 + ' cm²';
      document.getElementById('m_tris').innerText = d.report.triangle_count;
      document.getElementById('trace-box').innerText = 'TPMS Gyroid: Area ' + d.report.specific_surface_area_m2_m3 + ' m²/m³, Nu=' + d.report.nusselt_number + ', Heat Cap @40K=' + d.report.heat_dissipation_at_40k_delta_w + 'W';
    };

    document.getElementById('btn_passport').onclick = async () => {
      document.getElementById('trace-box').innerText = 'Cryptographically Signing SLSA Provenance Passport...';
      const r = await fetch('/api/v1/compliance/provenance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentSpec)
      });
      const d = await r.json();
      const p = d.passport;
      document.getElementById('trace-box').innerText = p.passport_id + ' | HMAC: ' + (d.signature_verified ? 'VERIFIED ✓' : 'INVALID ✗') + ' | Cost: $' + p.total_estimated_unit_cost_usd + ' | Drying: ' + p.drying_and_process.drying_temp_c + 'C/' + p.drying_and_process.drying_duration_hours + 'h';
    };

    document.getElementById('btn_wire').onclick = () => {
      wireframeMode = !wireframeMode;
      if (mesh) mesh.material.wireframe = wireframeMode;
    };

    document.getElementById('btn_heatmap').onclick = () => {
      heatmapMode = !heatmapMode;
      renderSpec(currentSpec);
    };

    async function downloadFormat(endpoint, filename) {
      const r = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentSpec)
      });
      const blob = await r.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
    }

    document.getElementById('btn_dl_3mf').onclick = () => downloadFormat('/api/v1/export/slicer-3mf', 'forge_slicer.3mf');
    document.getElementById('btn_dl_glb').onclick = () => downloadFormat('/api/v1/export/glb', 'forge_duct.glb');
    document.getElementById('btn_dl_usda').onclick = () => downloadFormat('/api/v1/export/openusd', 'forge_twin.usda');
    document.getElementById('btn_dl_inp').onclick = () => downloadFormat('/api/v1/export/calculix', 'forge_duct.inp');
    document.getElementById('btn_dl_fmu').onclick = () => downloadFormat('/api/v1/export/fmu', 'forge_duct.fmu');

    renderSpec(currentSpec);

    function animate() {
      requestAnimationFrame(animate);
      if (mesh) mesh.rotation.z += 0.0035;
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
