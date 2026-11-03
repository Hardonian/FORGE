# FORGE

FORGE is an experimental inverse-design project. Its long-term goal is to generate physical designs, evaluate them with physics models, and learn from real measurements. The initial domain is GPU cooling ducts for heterogeneous AI hardware.

## Current milestone: geometry foundation

The first implementation creates a hollow, rectangular transition duct from a JSON specification, exports a closed ASCII STL mesh, and records geometric metrics and a reproducibility hash. It requires Python 3.10+ and has no third-party dependencies.

```bash
python -m forge examples/duct.json --out runs/example
python -m unittest discover -s tests -v
```

`runs/example/duct.stl` is an experimental geometry; `report.json` contains input dimensions, opening areas, the maximum side angle, and the SHA-256 of the canonical geometry inputs. Reusing an output directory overwrites these two files.

**This is not a cooling simulation or an optimized design.** It does not establish airflow, pressure, temperature, noise, structural integrity, attachment, printability, material suitability, chassis clearance, or hardware safety. The sample dimensions are illustrative, not measurements of any GPU. Verify physical fit and material selection and independently monitor temperatures when testing actual equipment.

All dimensions are outer dimensions in millimetres; wall thickness is subtracted from both sides to determine the openings. The current loft is centered, straight and rectangular. It does not create flanges, mounting holes, fan curves or seals.

## Planned research path

1. **Geometry:** parameter validation, printable watertight meshes, constraints and reproducible candidate generation. Initial subset implemented.
2. **Candidate experiments:** explicit search spaces and geometric feasibility constraints; transparent, reproducible ranking with no unsupported claims about airflow.
3. **Physics:** integrate an established CFD solver with documented boundary conditions, mesh convergence and benchmark cases; preserve simulation logs and solver versions.
4. **Hardware validation:** document a safe, repeatable baseline and compare against instrumented real-world results. Record uncertainty and failed tests.
5. **Surrogate modeling:** only after a validated dataset exists, evaluate whether a predictive model speeds up search and generalizes to held-out designs.

Simulation is an approximation; success depends on boundary conditions, mesh quality, physical measurement and independent verification. No thermal performance or production-readiness claims are made by this milestone.

## Development

Run `python -m unittest discover -s tests -v`. Tests check geometric manifold closure, face orientation, validation and end-to-end export. Generated artifacts are excluded from version control.

## License

MIT; see LICENSE.

