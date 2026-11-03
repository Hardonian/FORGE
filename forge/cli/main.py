"""Command-line interface for the FORGE generative physical design platform."""
from __future__ import annotations
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from forge import __version__
from forge.core.models import DuctSpec, PhysicsConditions
from forge.core.hashes import sha256_hash
from forge.cad.mesh import build_duct_mesh
from forge.cad.dfm import analyze_dfm
from forge.cad.exporters import export_ascii_stl, export_binary_stl, export_3mf, export_obj, export_step
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.openfoam import generate_openfoam_case
from forge.physics.su2 import generate_su2_config
from forge.opt.search_space import DuctSearchSpace
from forge.opt.optimizer import run_nsga2_optimization
from forge.opt.report import generate_html_report
from forge.lab.ingestion import compare_bench_to_simulation, ingest_telemetry_file

app = typer.Typer(
    name="forge",
    help="FORGE: AI-accelerated generative inverse design platform for AI accelerator thermal management.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def version_callback(
    version: bool = typer.Option(False, "--version", "-v", help="Display FORGE version"),
):
    if version:
        console.print(f"[bold cyan]FORGE[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.command()
def generate(
    spec_path: Path = typer.Argument(..., help="Path to JSON specification file"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    format: str = typer.Option("stl", "--format", "-f", help="Export format: stl, bin_stl, 3mf, obj, step"),
):
    """Generate 3D watertight cooling duct mesh and geometry report."""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = DuctSpec(**data)
    tris = build_duct_mesh(spec)
    out.mkdir(parents=True, exist_ok=True)

    fingerprint = sha256_hash(spec.to_canonical_dict())

    if format == "stl":
        (out / "duct.stl").write_text(export_ascii_stl(tris), encoding="ascii")
    elif format == "bin_stl":
        (out / "duct.stl").write_bytes(export_binary_stl(tris))
    elif format == "3mf":
        (out / "duct.3mf").write_bytes(export_3mf(tris))
    elif format == "obj":
        (out / "duct.obj").write_text(export_obj(tris), encoding="utf-8")
    elif format == "step":
        (out / "duct.stp").write_text(export_step(tris), encoding="utf-8")
    else:
        raise typer.BadParameter(f"Unsupported format: {format}")

    # Geometry metrics
    inlet_area = (spec.inlet_width - 2 * spec.wall) * (spec.inlet_height - 2 * spec.wall)
    outlet_area = (spec.outlet_width - 2 * spec.wall) * (spec.outlet_height - 2 * spec.wall)
    report = {
        "schema_version": 2,
        "geometry_sha256": fingerprint,
        "dimensions_mm": spec.model_dump(),
        "geometry_metrics": {
            "inlet_open_area_mm2": inlet_area,
            "outlet_open_area_mm2": outlet_area,
            "area_ratio_outlet_to_inlet": outlet_area / max(1.0, inlet_area),
        },
        "evaluation_type": "geometry_only",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(f"[bold green][OK][/bold green] Generated {format.upper()} duct in [cyan]{out}[/cyan] (SHA-256: [dim]{fingerprint[:12]}...[/dim])")


@app.command()
def simulate(
    spec_path: Path = typer.Argument(..., help="Path to JSON specification file"),
    cfm: float = typer.Option(150.0, "--cfm", help="Volumetric airflow rate in CFM"),
    heat_watts: float = typer.Option(350.0, "--watts", help="Heat load to dissipate in Watts"),
):
    """Run real-time aerodynamic and thermal physics simulation."""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = DuctSpec(**data)
    conds = PhysicsConditions(volumetric_flow_cfm=cfm, heat_dissipation_watts=heat_watts)
    res = solve_duct_physics(spec, conds)

    table = Table(title=f"Aerodynamic & Thermal Performance ({cfm} CFM)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")

    table.add_row("Flow Velocity (Inlet)", f"{res.inlet_velocity_m_s:.2f} m/s")
    table.add_row("Flow Velocity (Outlet)", f"{res.outlet_velocity_m_s:.2f} m/s")
    table.add_row("Reynolds Number", f"{res.reynolds_number:,.0f}")
    table.add_row("Friction Loss", f"{res.friction_pressure_drop_pa:.2f} Pa")
    table.add_row("Minor Expansion Loss", f"{res.expansion_loss_pa:.2f} Pa")
    table.add_row("Total Pressure Drop (dP)", f"{res.total_pressure_drop_pa:.2f} Pa")
    table.add_row("Velocity Uniformity (gamma)", f"{res.flow_uniformity_index:.3f} / 1.000")
    table.add_row("Estimated Air Temp Rise", f"{res.estimated_temp_rise_c:.2f} C")

    console.print(table)


@app.command()
def dfm(
    spec_path: Path = typer.Argument(..., help="Path to JSON specification file"),
    printer: str = typer.Option("Bambu_Lab_X1C", "--printer", help="Target 3D printer"),
):
    """Analyze 3D printability, overhang angles, material mass, and heat limits."""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = DuctSpec(**data)
    tris = build_duct_mesh(spec)
    report = analyze_dfm(spec, tris, target_printer=printer)

    console.print(f"[bold cyan]DFM Analysis[/bold cyan] (Material: [bold green]{report.material}[/bold green])")
    console.print(f"- Estimated Solid Mass: [yellow]{report.estimated_mass_grams} g[/yellow]")
    console.print(f"- Estimated Print Time: [yellow]{report.estimated_print_time_hours} hrs[/yellow]")
    console.print(f"- Max Overhang Angle: [yellow]{report.max_overhang_angle_deg} deg[/yellow] (Compliant: {report.overhang_compliance})")
    console.print(f"- Fits Build Bed: {'[green]YES[/green]' if report.bed_fit else '[red]NO[/red]'}")
    if report.warnings:
        for w in report.warnings:
            console.print(f"  [bold red][!] Warning:[/bold red] {w}")


@app.command()
def openfoam(
    spec_path: Path = typer.Argument(..., help="Path to JSON specification file"),
    out: Path = typer.Option(..., "--out", "-o", help="Target OpenFOAM case directory"),
    cfm: float = typer.Option(150.0, "--cfm", help="Inlet airflow CFM"),
):
    """Generate complete containerized OpenFOAM CFD case directory."""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = DuctSpec(**data)
    conds = PhysicsConditions(volumetric_flow_cfm=cfm)
    case_path = generate_openfoam_case(spec, conds, out)
    console.print(f"[bold green][OK][/bold green] Generated OpenFOAM simpleFoam case at [cyan]{case_path}[/cyan]")


@app.command()
def su2(
    spec_path: Path = typer.Argument(..., help="Path to JSON specification file"),
    out: Path = typer.Option(Path("duct_cfd.cfg"), "--out", "-o", help="Target SU2 config file path"),
    cfm: float = typer.Option(150.0, "--cfm", help="Inlet airflow CFM"),
):
    """Generate SU2 CFD configuration file."""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = DuctSpec(**data)
    conds = PhysicsConditions(volumetric_flow_cfm=cfm)
    generate_su2_config(spec, conds, out)
    console.print(f"[bold green][OK][/bold green] Generated SU2 config at [cyan]{out}[/cyan]")


@app.command()
def optimize(
    out: Path = typer.Option(Path("runs/optimization"), "--out", "-o", help="Output directory"),
    population: int = typer.Option(20, "--pop", help="Population size"),
    generations: int = typer.Option(4, "--gen", help="Number of generations"),
):
    """Run generative inverse design optimization with NSGA-II."""
    space = DuctSearchSpace()
    conds = PhysicsConditions()
    console.print(f"[bold cyan]Running NSGA-II Pareto Optimization[/bold cyan] ({population} candidates x {generations} gens)...")
    pareto = run_nsga2_optimization(space, conds, population_size=population, generations=generations)

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "optimization_report.html"
    generate_html_report(pareto, report_path)
    console.print(f"[bold green][OK][/bold green] Pareto frontier synthesized with [bold]{len(pareto)}[/bold] designs.")
    console.print(f"- Report written to: [cyan]{report_path}[/cyan]")


@app.command()
def lab(
    telemetry_path: Path = typer.Argument(..., help="Path to test bench telemetry JSON"),
    spec_path: Path = typer.Argument(..., help="Path to duct spec JSON"),
):
    """Ingest test bench telemetry and compare against simulation."""
    bench = ingest_telemetry_file(telemetry_path)
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    res = compare_bench_to_simulation(spec, bench)
    console.print("[bold cyan]Sim2Real Telemetry Comparison[/bold cyan]")
    console.print(f"- Simulated dP: [yellow]{res.sim_delta_p_pa} Pa[/yellow]")
    console.print(f"- Measured dP:  [green]{res.measured_delta_p_pa} Pa[/green]")
    console.print(f"- Discrepancy:  [bold]{res.delta_p_discrepancy_percent}%[/bold] ({res.status})")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
):
    """Start the interactive 3D Web Studio and REST API server."""
    import uvicorn
    console.print(f"[bold cyan]Starting FORGE Web Studio[/bold cyan] at [green]http://{host}:{port}/viewer[/green]")
    uvicorn.run("forge.api.server:app", host=host, port=port, reload=False)


def main():
    app()


if __name__ == "__main__":
    main()
