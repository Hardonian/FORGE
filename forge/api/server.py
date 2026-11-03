"""FastAPI server providing REST endpoints and interactive 3D Web Studio."""
from __future__ import annotations
from typing import Dict, Optional
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from forge import __version__
from forge.core.models import DuctSpec, PhysicsConditions
from forge.core.hashes import sha256_hash
from forge.cad.mesh import build_duct_mesh
from forge.cad.dfm import analyze_dfm
from forge.cad.exporters import export_ascii_stl, export_binary_stl, export_3mf, export_step
from forge.physics.solver_1d import solve_duct_physics

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


@app.get("/viewer", response_class=HTMLResponse)
def interactive_3d_viewer():
    """Interactive 3D WebGL viewer using Three.js."""
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FORGE 3D Web Studio</title>
  <style>
    body { margin: 0; overflow: hidden; background: #0f172a; color: #f8fafc; font-family: sans-serif; }
    #ui { position: absolute; top: 16px; left: 16px; background: rgba(30, 41, 59, 0.85); backdrop-filter: blur(8px); padding: 16px; border-radius: 8px; border: 1px solid #334155; width: 280px; }
    h2 { margin-top: 0; font-size: 16px; color: #38bdf8; }
    label { display: block; font-size: 12px; margin-top: 8px; color: #94a3b8; }
    input[type=range] { width: 100%; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; background: #0284c7; font-size: 11px; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
  <div id="ui">
    <h2>FORGE 3D Studio <span class="badge">v0.2.0</span></h2>
    <label>Length (mm): <span id="lbl_l">100</span></label>
    <input type="range" id="rng_l" min="50" max="200" value="100">
    <label>Offset X (mm): <span id="lbl_ox">0</span></label>
    <input type="range" id="rng_ox" min="-40" max="40" value="0">
    <label>Guide Vanes: <span id="lbl_gv">0</span></label>
    <input type="range" id="rng_gv" min="0" max="3" value="0">
  </div>
  <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const light1 = new THREE.DirectionalLight(0xffffff, 1);
    light1.position.set(10, 20, 15);
    scene.add(light1);
    const light2 = new THREE.AmbientLight(0x404040, 2);
    scene.add(light2);

    const grid = new THREE.GridHelper(200, 20, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    camera.position.set(150, 150, 200);
    camera.lookAt(0, 0, 50);

    let mesh;
    function createDuctGeometry(length, ox, vanes) {
      if (mesh) scene.remove(mesh);
      const geom = new THREE.BoxGeometry(100, 80, length);
      const mat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3, metalness: 0.1, wireframe: false });
      mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(ox / 2, 0, length / 2);
      scene.add(mesh);
    }
    createDuctGeometry(100, 0, 0);

    document.getElementById('rng_l').oninput = (e) => {
      document.getElementById('lbl_l').innerText = e.target.value;
      createDuctGeometry(parseFloat(e.target.value), parseFloat(document.getElementById('rng_ox').value), 0);
    };
    document.getElementById('rng_ox').oninput = (e) => {
      document.getElementById('lbl_ox').innerText = e.target.value;
      createDuctGeometry(parseFloat(document.getElementById('rng_l').value), parseFloat(e.target.value), 0);
    };

    function animate() {
      requestAnimationFrame(animate);
      if (mesh) mesh.rotation.y += 0.005;
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
