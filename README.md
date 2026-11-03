# FORGE: AI-Accelerated Physical Inverse Design Platform

[![CI](https://github.com/example/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/example/forge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version: 0.2.0](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/example/forge)

**FORGE** is an open-source, physics-informed generative inverse engineering platform for specialized thermal and fluid hardware. Its initial domain is custom high-efficiency cooling ducts and transition baffles for heterogeneous AI accelerators (GPUs, TPUs, workstation blower conversions, and high-density server racks).

FORGE bridges the loop between **computational geometry**, **automated computational fluid dynamics (CFD)**, **multi-objective generative optimization**, and **real-world hardware telemetry calibration (Sim2Real)**.

---

## Key Capabilities

* **Parametric CAD & B-Rep Mesh Kernel (`forge.cad`):**
  * Phase-aligned, zero-twist transitions (`interpolate_morphed_ring`) between rectangular, rounded-rectangle, and circular profiles (e.g. 120mm circular server fan to wide rectangular GPU heatsink).
  * True 3D normal wall thickness compensation ($\sec\theta$ adjustment) preventing thin sloped walls during steep transitions.
  * Smooth $S$-curve Hermite centerline interpolation ($dx, dy$ offsets to bypass PCIe slots, memory sticks, and power cables).
  * Watertight internal aerodynamic splitter guide vanes that divide diffusers and suppress boundary layer separation stall.
  * Standard server fan mounting flanges (40mm, 60mm, 80mm, 92mm, 120mm, 140mm) with fastener holes and M3/M4 heat-set brass insert bosses, plus automated multi-part split-and-flange segmentation (`split_duct_for_printing`) for oversized ducts.
* **Multi-Format Industrial & WebGL Export:**
  * **Binary STL & ASCII STL:** Fast, compact IEEE-754 little-endian binary and ASCII meshes.
  * **3MF (3D Manufacturing Format):** Production archive with XML mesh data, millimeter units, and slicer metadata.
  * **STEP (ISO-10303-21):** Faceted `MANIFOLD_SOLID_BREP` with explicit `VERTEX_POINT` topology for CAD assemblies (SolidWorks, FreeCAD, Onshape).
  * **glTF 2.0 & Binary GLB:** Khronos glTF 2.0 JSON and self-contained binary `.glb` containers for AR/VR and WebGL pipelines.
  * **Wavefront OBJ:** Universal 3D visualization format with per-vertex area-weighted surface normals.
* **Design For Manufacturing (DFM) & Material Safety:**
  * Downward overhang angle analysis against print bed orientation ($>50^\circ$ warnings, excluding build-plate contact faces at $z = z_{\min}$).
  * Normal 3D wall thickness verification (`min_normal_wall_mm`) against nozzle extrusion minimums.
  * Solid material mass and print time estimates based on engineering filament profiles (PC-CF, PA12-CF, PEEK, PETG, ABS).
  * Datacenter thermal creep warnings for materials with Heat Deflection Temperatures below exhaust thresholds, plus build volume collision checking (Bambu Lab, Prusa, Voron).
* **Multi-Physics & Automated CFD Generation (`forge.physics`):**
  * High-fidelity 1D/2D analytical and empirical solver: exact Ramanujan ellipse / rounded-rect hydraulic diameters, Swamee-Jain friction, Idelchik diffuser loss with corner fillet secondary-flow suppression, and S-curve Dean vortex loss.
  * **Fan $P-Q$ Curve Operating Point Solver (`solve_fan_operating_point`):** Bisection solver intersecting non-linear server fan curves (`SanAce_40mm_1U`, `Delta_80mm_Server`, `Delta_120mm_HighSpeed`, `Noctua_140mm_Industrial`) with system impedance.
  * **Aeroacoustics & Metamaterial Liners:** Curle dipole aeroacoustic vortex-shedding noise (`acoustic_noise_dba`) and tuned sub-wavelength Helmholtz / quarter-wave metamaterial liner synthesis (`design_acoustic_metamaterial_liner`) to cancel fan Blade Passing Frequency ($f_{\text{BPF}}$) tones.
  * **Coupled Conjugate Heat Transfer (CHT) Heatsink Solver (`forge.physics.heatsink`):** End-to-end Fan + Duct + Finned Heatsink solver (`solve_coupled_duct_heatsink_system`) computing developing channel friction (Shah & London), Kays & London entrance/exit losses, Gnielinski/Muzychka Nusselt numbers, fin efficiency ($\eta_f$), and GPU die hotspot temperatures under outlet velocity maldistribution.
  * **Multi-Branch $1 \to N$ Splitter Manifolds (`forge.cad.manifold`, `forge.physics.network_solver`):** Parametric $1 \to N$ splitter manifold CAD mesh generation (`build_manifold_mesh`), parallel conductance hydraulic/thermal equilibrium solver (`solve_manifold_network`), and automated branch sizing (`auto_balance_manifold`) for isothermal multi-GPU cooling.
  * **Swirl-Canceling Honeycomb Stators (`forge.cad.honeycomb`):** Parametric cellular flow-straightener mesh and Idelchik screen hydraulics (`evaluate_honeycomb_hydraulics`).
  * Automated **OpenFOAM** `simpleFoam` case generator with $k-\omega$ SST turbulence, 3-layer prism boundary layer inflation (`addLayersControls`), and negative fluid domain extraction.
  * Automated **SU2** incompressible RANS configuration generator (`duct_cfd.cfg`).
* **Multi-Objective & Gradient Inverse Design (`forge.opt`):**
  * **Stratford / Bell-Mouth Multi-Station Bézier Diffuser Optimizer (`optimize_bezier_diffuser`):** Gradient-based non-linear expansion profile synthesis that expands rapidly while boundary-layer kinetic energy is high and gently decelerates near the exit to suppress separation.
  * Latin Hypercube Sampling (LHS) across bounded parameter search spaces (`inlet`, `outlet`, `length`, `vanes`, `corner_radius`).
  * Full NSGA-II genetic algorithm (`fast_non_dominated_sort`, `compute_crowding_distances`, crowded binary tournament selection) synthesizing Pareto-optimal designs balancing **minimum pressure drop ($\Delta P$)**, **maximum velocity uniformity ($\gamma$)**, and **minimum material-specific mass**.
  * Automated HTML engineering reports with embedded interactive SVG Pareto trade-off charts.
* **Persistent SQLite Digital Twin Registry & Hardware Presets (`forge.registry`, `forge.presets`):**
  * Zero-dependency SQLite lineage store (`ForgeRegistry`) tracking SHA-256 canonical hashes, physics/DFM evaluations, Pareto queries, surrogate training datasets, and Sim2Real telemetry runs.
  * Built-in hardware accelerator presets (`H100_PCIe_1U_SanAce40`, `L40S_80mm_Server_Duct`, `RTX_4090_120mm_Blower_Conversion`, `MI300X_OAM_120mm_HighFlow`, `Quiet_Workstation_140mm_GPU`).
* **Sim2Real Hardware Test Bench Calibration (`forge.lab`):**
  * Telemetry schema for physical wind tunnel / flow bench instrumentation (differential pressure transducers, anemometers, thermocouples).
  * Golden-section search (`calibrate_surface_roughness`) and joint 2D alternating coordinate descent (`calibrate_sim2real_model`) calibrating both empirical print roughness ($\varepsilon$) and minor loss multiplier ($K_{\text{mult}}$).
* **Physics-Informed Neural Surrogates (`forge.surrogate`):**
  * PyTorch Physics-Informed Residual Neural Operator (`DuctFlowSurrogate`) combining a differentiable analytical Darcy-Weisbach/Borda-Carnot prior with a learned MLP residual network (`train_on_specs`).
  * Monte Carlo Dropout uncertainty quantification (`predict_with_uncertainty`) and Lower Confidence Bound (`LCB`) active learning acquisition (`rank_candidates_by_acquisition`) to screen $10,000+$ designs in milliseconds.
* **Interactive 3D Web Studio & REST API (`forge.api`):**
  * FastAPI server with endpoints for design evaluation (`/api/v1/evaluate`), live WebGL buffer streaming (`/api/v1/mesh`), asynchronous Pareto optimization (`/api/v1/optimize`), CAD export (`/api/v1/export/{stl,3mf,step,glb}`), and an interactive Three.js 3D Web Studio (`/viewer`) rendering exact parametric duct and vane meshes in real time.

---

## Architecture

```text
forge/
├── core/         # Geometric primitives (Vec3, Triangle), Pydantic v2 schemas, canonical hashing
├── cad/          # Parametric lofting, server fan flanges, 3MF/STL/STEP exporters, DFM analysis
├── physics/      # Aerodynamic solver (ΔP, uniformity, thermal rise), OpenFOAM & SU2 generators
├── opt/          # Latin Hypercube sampling, NSGA-II Pareto optimizer, HTML reporting
├── lab/          # Test bench telemetry ingestion, Sim2Real discrepancy, roughness calibration
├── surrogate/    # PyTorch neural operator surrogate model and active learning screener
├── cli/          # Typer command-line interface
├── api/          # FastAPI web server and Three.js 3D Web Studio
└── duct.py       # Backward-compatible transition duct interface
```

---

## Installation

FORGE requires **Python 3.10+**.

```bash
git clone https://github.com/example/forge.git
cd FORGE

# Install package with all dependencies
pip install .

# Or install in editable development mode with test and lint tooling
pip install -e .[dev]
```

---

## Command-Line Usage

FORGE provides a unified CLI with subcommands:

### 1. Generate 3D Meshes (STL, 3MF, STEP, OBJ)

```bash
# Export standard ASCII STL
python -m forge generate examples/duct.json --out runs/example --format stl

# Export compact Binary STL
python -m forge generate examples/duct.json --out runs/example --format bin_stl

# Export 3MF manufacturing container
python -m forge generate examples/duct.json --out runs/example --format 3mf

# Export STEP CAD file
python -m forge generate examples/duct.json --out runs/example --format step
```

### 2. Aerodynamic & Thermal Simulation

```bash
python -m forge simulate examples/duct.json --cfm 150 --watts 400
```

*Outputs flow velocities, Reynolds number, friction loss, minor expansion loss, total pressure drop $\Delta P$, velocity uniformity $\gamma$, and estimated air temperature rise.*

### 3. DFM Printability & Thermal Assessment

```bash
python -m forge dfm examples/duct.json --printer Bambu_Lab_X1C
```

*Outputs solid mass (g), print time (hrs), overhang angle checks, build bed volume fit, and material thermal creep warnings.*

### 4. Automated CFD Case Generation

```bash
# Generate OpenFOAM simpleFoam case
python -m forge openfoam examples/duct.json --out runs/openfoam_case --cfm 150

# Generate SU2 CFD configuration
python -m forge su2 examples/duct.json --out runs/su2.cfg --cfm 150
```

### 5. Multi-Objective Generative Optimization (NSGA-II)

```bash
python -m forge optimize --pop 20 --gen 5 --out runs/optimization
```

*Synthesizes Pareto-optimal trade-offs and produces an interactive standalone report at `runs/optimization/optimization_report.html`.*

### 6. Hardware Sim2Real Telemetry Ingestion

```bash
python -m forge lab telemetry.json examples/duct.json
```

*Calculates physical measurement discrepancy against simulation predictions.*

### 7. Interactive 3D Web Studio & REST API

```bash
python -m forge serve --port 8000
```

*Navigate to `http://127.0.0.1:8000/viewer` to view and interact with ducts in real-time 3D WebGL.*

---

## Legacy Backward Compatibility

The original script invocation continues to work seamlessly:

```bash
python -m forge examples/duct.json --out runs/example
```

And in Python code:

```python
from forge.duct import Duct, write_candidate

duct = Duct(100, 80, 120, 50, 100, 2)
report = write_candidate(duct, "runs/example")
```

---

## Testing & Quality Assurance

Run the test suite:

```bash
# Run all unit and integration tests
pytest -v

# Run linting and code style checks
ruff check .
```

---

## License

MIT; see [LICENSE](file:///C:/Users/scott/FORGE/LICENSE).
