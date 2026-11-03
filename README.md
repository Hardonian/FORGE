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
  * Seamless transitions between rectangular, rounded-rectangle, and circular profiles (e.g. 120mm circular server fan to wide rectangular GPU heatsink).
  * Smooth $S$-curve centerline interpolation ($dx, dy$ offsets to bypass PCIe slots, memory sticks, and power cables).
  * Internal aerodynamic splitter guide vanes that divide diffusers and suppress boundary layer separation stall.
  * Standard server fan mounting flanges (40mm, 60mm, 80mm, 92mm, 120mm, 140mm) with fastener holes and M3/M4 heat-set brass insert bosses.
* **Multi-Format Industrial Export:**
  * **Binary STL:** Fast, compact, IEEE-754 little-endian binary mesh.
  * **3MF (3D Manufacturing Format):** Production archive with XML mesh data, millimeter units, and slicer metadata.
  * **STEP (ISO-10303-21):** Faceted boundary representation for CAD assemblies (SolidWorks, FreeCAD, Onshape).
  * **Wavefront OBJ & ASCII STL:** Universal 3D visualization formats.
* **Design For Manufacturing (DFM) & Material Safety:**
  * Downward overhang angle analysis against print bed orientation ($>50^\circ$ warnings).
  * Solid material mass and print time estimates based on engineering filament profiles (PC-CF, PA12-CF, PEEK, PETG, ABS).
  * Datacenter thermal creep warnings for materials with Heat Deflection Temperatures below exhaust thresholds.
  * Build volume collision checking against common 3D printers (Bambu Lab, Prusa, Voron).
* **Physics & Automated CFD Generation (`forge.physics`):**
  * Built-in 1D/2D analytical and empirical aerodynamic solver (Swamee-Jain / Colebrook-White friction + Idelchik diffuser loss coefficients).
  * Automated **OpenFOAM** `simpleFoam` case generator with $k-\omega$ SST turbulence, prism boundary layers, and fan boundary conditions.
  * Automated **SU2** incompressible RANS configuration generator (`duct_cfd.cfg`).
* **Multi-Objective Inverse Design (`forge.opt`):**
  * Latin Hypercube Sampling (LHS) across bounded parameter search spaces.
  * Genetic algorithm (NSGA-II) synthesizing Pareto-optimal designs balancing **minimum pressure drop ($\Delta P$)**, **maximum velocity uniformity ($\gamma$)**, and **minimum mass**.
  * Automated HTML engineering reports with embedded interactive SVG Pareto trade-off charts.
* **Sim2Real Hardware Test Bench Calibration (`forge.lab`):**
  * Telemetry schema for physical wind tunnel / flow bench instrumentation (differential pressure transducers, anemometers, thermocouples).
  * Automated discrepancy calculation ($\epsilon_{\Delta P}$) and Bayesian empirical roughness calibration.
* **Deep Learning Neural Surrogates (`forge.surrogate`):**
  * PyTorch neural operator predicting flow metrics in $<1$ ms.
  * Active learning screening pipeline to evaluate $10,000+$ candidate designs prior to invoking CFD.
* **Interactive 3D Web Studio & REST API (`forge.api`):**
  * Built-in FastAPI server with endpoints for design generation, export, evaluation, and an interactive Three.js 3D WebGL studio.

---

## Architecture

```
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
