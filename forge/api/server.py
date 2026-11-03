"""FastAPI server providing REST endpoints and interactive Three.js 3D Web Studio."""
from __future__ import annotations
from typing import Dict, Optional
from fastapi import FastAPI, Response
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
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.heatsink import HeatsinkSpec, solve_coupled_duct_heatsink_system
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

_ENTITLEMENT_MANAGER = CloudEntitlementManager()

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


@app.get("/viewer", response_class=HTMLResponse)
def interactive_3d_viewer():
    """Interactive Three.js 3D Web Studio rendering real manifold BufferGeometry and live physics."""
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FORGE 3D Web Studio</title>
  <style>
    body { margin: 0; overflow: hidden; background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    #ui { position: absolute; top: 16px; left: 16px; background: rgba(15, 23, 42, 0.90); backdrop-filter: blur(10px); padding: 18px; border-radius: 10px; border: 1px solid #334155; width: 310px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    #telemetry { position: absolute; top: 16px; right: 16px; background: rgba(15, 23, 42, 0.90); backdrop-filter: blur(10px); padding: 18px; border-radius: 10px; border: 1px solid #334155; width: 290px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h2 { margin: 0 0 10px 0; font-size: 16px; color: #38bdf8; display: flex; justify-content: space-between; align-items: center; }
    label { display: flex; justify-content: space-between; font-size: 12px; margin-top: 10px; color: #cbd5e1; }
    input[type=range] { width: 100%; accent-color: #38bdf8; cursor: pointer; }
    select { width: 100%; margin-top: 4px; padding: 5px; background: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 4px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; background: #0284c7; color: #fff; font-size: 11px; }
    .metric-row { display: flex; justify-content: space-between; font-size: 12px; padding: 6px 0; border-bottom: 1px solid #1e293b; }
    .metric-val { font-weight: 600; color: #34d399; }
    .btn-row { display: flex; gap: 8px; margin-top: 14px; }
    button { flex: 1; padding: 7px; background: #2563eb; color: #fff; border: none; border-radius: 5px; font-size: 12px; cursor: pointer; font-weight: 600; }
    button:hover { background: #1d4ed8; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
  <div id="ui">
    <h2>FORGE 3D Studio <span class="badge">v0.2.0</span></h2>
    <label>Inlet Shape</label>
    <select id="sel_in_shape">
      <option value="rectangular">Rectangular</option>
      <option value="circular">Circular (Server Fan)</option>
      <option value="rounded_rectangle">Rounded Rectangle</option>
    </select>
    <label>Length (mm) <span id="lbl_l">110</span></label>
    <input type="range" id="rng_l" min="50" max="200" value="110">
    <label>Offset X (mm) <span id="lbl_ox">15</span></label>
    <input type="range" id="rng_ox" min="-45" max="45" value="15">
    <label>Corner Fillet (mm) <span id="lbl_cr">6</span></label>
    <input type="range" id="rng_cr" min="0" max="20" value="6">
    <label>Splitter Guide Vanes <span id="lbl_gv">1</span></label>
    <input type="range" id="rng_gv" min="0" max="4" value="1">
    <div class="btn-row">
      <button id="btn_wire">Toggle Wireframe</button>
    </div>
  </div>

  <div id="telemetry">
    <h2>Live Physics & DFM</h2>
    <div class="metric-row"><span>Total Pressure Drop (dP)</span><span class="metric-val" id="m_dp">-- Pa</span></div>
    <div class="metric-row"><span>Flow Uniformity (gamma)</span><span class="metric-val" id="m_uni">--</span></div>
    <div class="metric-row"><span>Reynolds Number</span><span class="metric-val" id="m_re">--</span></div>
    <div class="metric-row"><span>Aeroacoustic Noise</span><span class="metric-val" id="m_db">-- dBA</span></div>
    <div class="metric-row"><span>Wall Deflection</span><span class="metric-val" id="m_def">-- um</span></div>
    <div class="metric-row"><span>Fan Operating Point</span><span class="metric-val" id="m_cfm">-- CFM</span></div>
    <div class="metric-row"><span>Solid Mass (PC-CF)</span><span class="metric-val" id="m_mass">-- g</span></div>
    <div class="metric-row"><span>Manifold Triangles</span><span class="metric-val" id="m_tris">--</span></div>
  </div>

  <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const light1 = new THREE.DirectionalLight(0xffffff, 1.1);
    light1.position.set(120, 180, 160);
    scene.add(light1);
    const light2 = new THREE.DirectionalLight(0x38bdf8, 0.5);
    light2.position.set(-120, -100, -80);
    scene.add(light2);
    scene.add(new THREE.AmbientLight(0x404040, 1.5));

    const grid = new THREE.GridHelper(300, 30, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    camera.position.set(170, -160, 150);
    camera.lookAt(0, 0, 55);

    let mesh = null;
    let wireframeMode = false;

    async function updateDuctMesh() {
      const payload = {
        spec: {
          inlet_width: 120,
          inlet_height: 120,
          outlet_width: 140,
          outlet_height: 48,
          length: parseFloat(document.getElementById('rng_l').value),
          wall: 2.2,
          offset_x: parseFloat(document.getElementById('rng_ox').value),
          offset_y: 0.0,
          corner_radius: parseFloat(document.getElementById('rng_cr').value),
          inlet_shape: document.getElementById('sel_in_shape').value,
          outlet_shape: "rectangular",
          guide_vanes: parseInt(document.getElementById('rng_gv').value),
          material: "PC-CF"
        }
      };

      const res = await fetch('/api/v1/mesh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
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

      const mat = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        roughness: 0.35,
        metalness: 0.15,
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
        updateDuctMesh();
      };
    });

    document.getElementById('btn_wire').onclick = () => {
      wireframeMode = !wireframeMode;
      if (mesh) mesh.material.wireframe = wireframeMode;
    };

    updateDuctMesh();

    function animate() {
      requestAnimationFrame(animate);
      if (mesh) mesh.rotation.z += 0.004;
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
