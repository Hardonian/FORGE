"""Command-line interface for the FORGE generative physical design platform."""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.table import Table

from forge import __version__
from forge.core.models import DuctSpec, PhysicsConditions
from forge.core.hashes import sha256_hash
from forge.cad.mesh import build_duct_mesh
from forge.cad.dfm import analyze_dfm
from forge.cad.exporters import (
    export_ascii_stl,
    export_binary_stl,
    export_3mf,
    export_obj,
    export_step,
    export_gltf,
    export_glb,
)
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.openfoam import generate_openfoam_case
from forge.physics.su2 import generate_su2_config
from forge.opt.search_space import DuctSearchSpace
from forge.opt.optimizer import run_nsga2_optimization
from forge.opt.report import generate_html_report
from forge.lab.ingestion import compare_bench_to_simulation, ingest_telemetry_file
from forge.lab.calibration import calibrate_sim2real_model
from forge.physics.heatsink import HeatsinkSpec, solve_coupled_duct_heatsink_system
from forge.physics.acoustics import design_acoustic_metamaterial_liner
from forge.presets.catalog import get_preset_spec, list_presets
from forge.cad.slicer import export_production_slicer_3mf
from forge.copilot.agent import synthesize_from_prompt
from forge.topopt.darcy_solver import run_fluid_topology_optimization
from forge.cloud.rack_twin import evaluate_datacenter_rack_roi
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
from forge.interop.mcp_server import get_openai_tool_schemas
from forge.interop.open_schemas import export_all_json_schemas
from forge.cad.nurbs_step import export_exact_nurbs_step
from forge.cad.tpms import TPMSSpec, build_tpms_lattice_mesh, evaluate_tpms_thermo_hydraulics
from forge.opt.differentiable_adjoint import run_differentiable_adjoint_codesign
from forge.physics.hybrid_cooling import HybridAALCSpec, solve_hybrid_aalc_cooling
from forge.physics.vectorized import evaluate_duct_specs_tensor_batch
from forge.interop.calculix_fea import export_calculix_inp
from forge.interop.fmi_fmu import export_fmu_package
from forge.compliance.provenance import (
    generate_hardware_provenance_passport,
    verify_hardware_provenance,
)

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
    format: str = typer.Option("stl", "--format", "-f", help="Export format: stl, bin_stl, 3mf, obj, step, gltf, glb"),
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
    elif format == "gltf":
        (out / "duct.gltf").write_text(export_gltf(tris), encoding="utf-8")
    elif format == "glb":
        (out / "duct.glb").write_bytes(export_glb(tris))
    elif format == "usda":
        (out / "duct.usda").write_text(export_openusd_stage(spec, triangles=tris), encoding="utf-8")
    elif format == "modelica":
        (out / "duct.mo").write_text(export_modelica_component(spec), encoding="utf-8")
    else:
        raise typer.BadParameter(f"Unsupported format: {format}")

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
    """Run real-time aerodynamic, thermal, acoustic, and structural physics simulation."""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = DuctSpec(**data)
    conds = PhysicsConditions(volumetric_flow_cfm=cfm, heat_dissipation_watts=heat_watts)
    res = solve_duct_physics(spec, conds)

    table = Table(title=f"Aerodynamic, Thermal & Structural Performance ({cfm} CFM)")
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
    table.add_row("Aeroacoustic Noise", f"{res.acoustic_noise_dba:.1f} dBA")
    table.add_row("Max Wall Deflection", f"{res.max_wall_deflection_um:.3f} um")
    table.add_row("Fan Operating Equilibrium", f"{res.fan_operating_cfm:.1f} CFM")

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
    console.print(f"- Min Normal Wall:      [yellow]{report.min_normal_wall_mm} mm[/yellow]")
    console.print(f"- Max Overhang Angle:   [yellow]{report.max_overhang_angle_deg} deg[/yellow] (Compliant: {report.overhang_compliance})")
    console.print(f"- Fits Build Bed:       {'[green]YES[/green]' if report.bed_fit else '[red]NO[/red]'}")
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
    """Ingest test bench telemetry, compare against simulation, and calibrate Sim2Real model."""
    bench = ingest_telemetry_file(telemetry_path)
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    res = compare_bench_to_simulation(spec, bench)
    cal = calibrate_sim2real_model(spec, [bench])
    console.print("[bold cyan]Sim2Real Telemetry Comparison & Calibration[/bold cyan]")
    console.print(f"- Simulated dP: [yellow]{res.sim_delta_p_pa} Pa[/yellow]")
    console.print(f"- Measured dP:  [green]{res.measured_delta_p_pa} Pa[/green]")
    console.print(f"- Discrepancy:  [bold]{res.delta_p_discrepancy_percent}%[/bold] ({res.status})")
    console.print(f"- Calibrated Roughness: [cyan]{cal.calibrated_roughness_mm} mm[/cyan] (RMSE reduced {cal.baseline_rmse_pa} -> {cal.calibrated_rmse_pa} Pa)")


@app.command()
def presets(
    export_name: str = typer.Option("", "--export", "-e", help="Optional preset name to export to JSON"),
    out: Path = typer.Option(Path("preset_duct.json"), "--out", "-o", help="Output JSON file path when exporting"),
):
    """List built-in AI accelerator cooling duct presets or export a preset to JSON."""
    if export_name:
        entry = get_preset_spec(export_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(entry["spec"].model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        console.print(f"[bold green][OK][/bold green] Exported preset [cyan]{export_name}[/cyan] to [green]{out}[/green]")
        return

    table = Table(title="FORGE Hardware Accelerator Presets")
    table.add_column("Preset Name", style="cyan bold")
    table.add_column("Fan Standard", style="green")
    table.add_column("Material", style="yellow")
    table.add_column("Description", style="white")
    for p in list_presets():
        table.add_row(
            p["name"],
            str(p["spec"].get("fan_standard") or "custom"),
            p["spec"]["material"],
            p["description"],
        )
    console.print(table)


@app.command()
def heatsink(
    spec_path: Path = typer.Argument(..., help="Path to duct JSON specification"),
    watts: float = typer.Option(400.0, "--watts", help="Accelerator TDP heat load in Watts"),
    fins: int = typer.Option(54, "--fins", help="Number of parallel cooling fins"),
    material: str = typer.Option("COPPER", "--material", help="Heatsink material (COPPER or ALUMINUM_6063)"),
):
    """Solve coupled Fan + Duct + Finned Heatsink conjugate heat transfer and junction temperatures."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    hs = HeatsinkSpec(
        width_mm=spec.outlet_width,
        fin_height_mm=max(10.0, spec.outlet_height - 4.0),
        fin_count=fins,
        material=material,
    )
    conds = PhysicsConditions(heat_dissipation_watts=watts)
    rep = solve_coupled_duct_heatsink_system(spec, hs, conds)

    table = Table(title=f"Coupled Conjugate Heat Transfer ({watts} W, {fins} {material} Fins)")
    table.add_column("System Metric", style="cyan")
    table.add_column("Value", style="green bold")
    table.add_row("Equilibrium System Airflow", f"{rep.equilibrium_flow_cfm:.1f} CFM")
    table.add_row("Duct Pressure Drop", f"{rep.duct_pressure_drop_pa:.2f} Pa")
    table.add_row("Heatsink Fin-Array dP", f"{rep.heatsink_pressure_drop_pa:.2f} Pa")
    table.add_row("Total System dP", f"{rep.total_system_pressure_drop_pa:.2f} Pa")
    table.add_row("Convective Heat Transfer (h)", f"{rep.convective_h_w_m2k:.1f} W/(m^2*K)")
    table.add_row("Fin Efficiency (eta_f)", f"{rep.fin_efficiency * 100.0:.1f} %")
    table.add_row("Total Thermal Resistance", f"{rep.heatsink_thermal_resistance_k_w:.4f} K/W")
    table.add_row("Mean Die Temperature", f"{rep.mean_die_temp_c:.1f} C")
    table.add_row("Hotspot Die Temperature", f"{rep.hotspot_die_temp_c:.1f} C")
    table.add_row("Thermal Margin to Throttle", f"{rep.thermal_margin_c:.1f} C ({'SAFE' if not rep.throttling_risk else 'THROTTLE RISK'})")
    console.print(table)


@app.command()
def acoustics(
    rpm: int = typer.Option(8500, "--rpm", help="Server fan speed in RPM"),
    blades: int = typer.Option(7, "--blades", help="Number of fan rotor blades"),
    depth_mm: float = typer.Option(10.0, "--depth", help="Available liner cavity depth in mm"),
):
    """Design a tuned Helmholtz / quarter-wave acoustic metamaterial liner for fan BPF suppression."""
    des = design_acoustic_metamaterial_liner(fan_rpm=rpm, blade_count=blades, max_cavity_depth_mm=depth_mm)
    console.print(f"[bold cyan]Tuned Acoustic Metamaterial Liner[/bold cyan] ({rpm} RPM, {blades} blades)")
    console.print(f"- Blade Passing Frequency (BPF): [green]{des.blade_passing_freq_hz} Hz[/green] (Harmonics: {des.harmonics_hz} Hz)")
    console.print(f"- Quarter-Wave Folded Length:    [yellow]{des.quarter_wave_length_mm} mm[/yellow]")
    console.print(f"- Helmholtz Cell Footprint:      [yellow]{des.helmholtz_cavity_width_mm} x {des.helmholtz_cavity_width_mm} x {des.helmholtz_cavity_depth_mm} mm[/yellow]")
    console.print(f"- Perforation Porosity:          [yellow]{des.perforation_porosity_pct} %[/yellow]")
    console.print(f"- Peak BPF Transmission Loss:    [bold green]-{des.peak_transmission_loss_db} dB[/bold green] (Stopband: {des.effective_stopband_hz[0]} - {des.effective_stopband_hz[1]} Hz)")


@app.command()
def copilot(
    prompt: str = typer.Argument(..., help="Natural language engineering prompt"),
    out: Path = typer.Option(Path("runs/copilot"), "--out", "-o", help="Output directory for spec and 3MF"),
):
    """Synthesize an obstacle-routed, Bezier-optimized cooling duct directly from a natural language prompt."""
    res = synthesize_from_prompt(prompt)
    out.mkdir(parents=True, exist_ok=True)
    spec_file = out / "synthesized_duct.json"
    slicer_file = out / "production_duct.3mf"
    spec_file.write_text(json.dumps(res.spec.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    slicer_file.write_bytes(export_production_slicer_3mf(res.spec, printer_profile=res.target_printer))

    console.print("[bold cyan]FORGE Design Copilot Synthesis[/bold cyan]")
    for step in res.reasoning_trace:
        console.print(f"  - {step}")
    console.print(
        f"[bold green][OK][/bold green] Saved spec to [cyan]{spec_file}[/cyan] and slicer 3MF to [cyan]{slicer_file}[/cyan] "
        f"(dP={res.physics.total_pressure_drop_pa} Pa, gamma={res.physics.flow_uniformity_index})"
    )


@app.command()
def slicer(
    spec_path: Path = typer.Argument(..., help="Path to duct JSON specification"),
    out: Path = typer.Option(Path("production_slicer.3mf"), "--out", "-o", help="Output .3mf file path"),
    printer: str = typer.Option("Bambu_Lab_X1C", "--printer", help="Target 3D printer profile"),
):
    """Export a production-ready 3MF container with filament profiles and interlocking split flanges."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = export_production_slicer_3mf(spec, printer_profile=printer)
    out.write_bytes(payload)
    console.print(f"[bold green][OK][/bold green] Exported production slicer 3MF ({len(payload)} bytes) to [cyan]{out}[/cyan]")


@app.command()
def topopt(
    spec_path: Path = typer.Argument(..., help="Path to baseline duct JSON specification"),
    out: Path = typer.Option(Path("runs/topopt"), "--out", "-o", help="Output directory"),
    iterations: int = typer.Option(15, "--iter", help="Brinkman-Darcy optimization iterations"),
):
    """Run Navier-Stokes Brinkman-Darcy fluid topology optimization and extract organic 3D mesh."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    res, tris = run_fluid_topology_optimization(spec, iterations=iterations)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topopt_spec.json").write_text(json.dumps(res.optimized_spec.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (out / "topopt_duct.stl").write_bytes(export_binary_stl(tris))
    console.print(
        f"[bold green][OK][/bold green] Brinkman-Darcy TopOpt complete: dissipation reduced by [bold]{res.dissipation_reduction_pct}%[/bold], "
        f"dP: {res.baseline_dp_pa} -> {res.optimized_dp_pa} Pa, extracted vanes: {res.extracted_guide_vanes}"
    )


@app.command()
def rack_roi(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
):
    """Evaluate enterprise datacenter 42U rack power savings, CO2 reduction, and financial payback."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    rep = evaluate_datacenter_rack_roi(spec)
    table = Table(title=f"Datacenter Rack Digital Twin & ROI ({rep.rack_id}, {rep.total_gpus_in_rack} GPUs)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")
    table.add_row("OEM vs FORGE Pressure Drop", f"{rep.oem_duct_pressure_drop_pa} Pa -> {rep.forge_duct_pressure_drop_pa} Pa (-{rep.pressure_drop_reduction_pct}%)")
    table.add_row("Fan Power Saved per Rack", f"{rep.fan_power_saved_watts_per_rack} W")
    table.add_row("Annual Savings per Rack", f"${rep.annual_savings_usd_per_rack:,.2f} / yr ({rep.annual_energy_saved_kwh_per_rack:,.1f} kWh)")
    table.add_row("100-Rack Cluster Savings", f"${rep.cluster_100_rack_annual_savings_usd:,.2f} / yr")
    table.add_row("CO2 Reduction (100 Racks)", f"{rep.co2_metric_tons_saved_per_year_100_racks} metric tons / yr")
    table.add_row("Hotspot Temp Reduction", f"-{rep.hotspot_temp_reduction_c} C")
    table.add_row("Additive Retrofit Payback", f"{rep.payback_period_days} days")
    console.print(table)


@app.command()
def modal(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    rpm: int = typer.Option(9000, "--rpm", help="Server fan speed in RPM"),
    blades: int = typer.Option(7, "--blades", help="Number of fan blades"),
):
    """Analyze structural natural frequencies, temperature softening, and fan harmonic resonance."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    rep = analyze_structural_modal_resonance(spec, fan_rpm=rpm, blade_count=blades)
    console.print(f"[bold cyan]Structural Modal & Harmonic Resonance Report[/bold cyan] ({rpm} RPM)")
    console.print(f"- Fundamental Panel Freq (f_11): [green]{rep.fundamental_panel_freq_hz} Hz[/green] (Cantilever: {rep.cantilever_beam_freq_hz} Hz)")
    console.print(f"- Fan Shaft / BPF Excitation:    [yellow]{rep.fan_shaft_freq_hz} Hz / {rep.fan_bpf_freq_hz} Hz[/yellow]")
    console.print(f"- Min Harmonic Separation:       [bold]{rep.min_harmonic_separation_pct}%[/bold] (Resonance Risk: {rep.resonance_risk})")
    console.print(f"- Flange Thermal Stress:         [yellow]{rep.flange_thermal_stress_mpa} MPa[/yellow] (Expansion: {rep.thermal_expansion_um} um)")


@app.command()
def transient(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    burst_watts: float = typer.Option(450.0, "--watts", help="Burst GPU TDP in Watts"),
    duration_s: float = typer.Option(20.0, "--duration", help="Simulation duration in seconds"),
):
    """Simulate time-domain transient die temperature and closed-loop PID fan PWM response."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    rep = simulate_transient_thermal_response(spec, burst_watts=burst_watts, duration_s=duration_s)
    console.print(f"[bold cyan]Transient Thermal & PID Fan Response[/bold cyan] ({rep.idle_watts}W -> {rep.burst_watts}W)")
    console.print(f"- Peak Hotspot Die Temp:    [yellow]{rep.peak_hotspot_temp_c} C[/yellow] (Final: {rep.final_hotspot_temp_c} C)")
    console.print(f"- 90% Fan Spin-Up Time:     [green]{rep.fan_spinup_90pct_time_s} s[/green]")
    console.print(f"- Thermal Throttling Risk:  {'[red]YES[/red]' if rep.thermal_throttling_triggered else '[green]NO[/green]'}")


@app.command()
def compliance(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    out: Path = typer.Option(Path("compliance_dossier.md"), "--out", "-o", help="Output Markdown certificate path"),
):
    """Generate an enterprise UL94 / ASHRAE / NEBS hardware qualification dossier."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    dossier = generate_compliance_dossier(spec)
    md = export_compliance_dossier_markdown(dossier)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    console.print(
        f"[bold green][OK][/bold green] Generated Qualification Dossier [cyan]{dossier.certificate_id}[/cyan] "
        f"({dossier.overall_qualification_status}) at [green]{out}[/green]"
    )


@app.command()
def cad_script(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    target: str = typer.Option("onshape", "--target", "-t", help="Target CAD ecosystem: onshape, freecad, openscad"),
    out: Path = typer.Option(Path("forge_duct.fs"), "--out", "-o", help="Output CAD script file path"),
):
    """Export native parametric CAD code for Onshape (FeatureScript), FreeCAD (.FCMacro), or OpenSCAD (.scad)."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    t = target.lower()
    if t == "onshape":
        code = export_onshape_featurescript(spec)
    elif t == "freecad":
        code = export_freecad_macro(spec)
    elif t == "openscad":
        code = export_openscad_script(spec)
    else:
        raise typer.BadParameter(f"Unsupported CAD target '{target}'. Choose onshape, freecad, or openscad.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8")
    console.print(f"[bold green][OK][/bold green] Exported {target.upper()} parametric script to [cyan]{out}[/cyan]")


@app.command()
def interop(
    target: str = typer.Option(
        "schemas",
        "--target",
        "-t",
        help="Interop target: schemas, tools, openusd, modelica, calculix, fmu",
    ),
    spec_path: Path = typer.Option(Path("examples/duct.json"), "--spec", "-s", help="Path to spec JSON"),
    out: Path = typer.Option(Path("forge_interop.json"), "--out", "-o", help="Output file path"),
):
    """Export open standards: Draft 2020-12 JSON Schemas, MCP Tools, OpenUSD (.usda), Modelica (.mo), CalculiX FEA (.inp), or FMI 2.0 (.fmu)."""
    t = target.lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if t == "schemas":
        out.write_text(json.dumps(export_all_json_schemas(), indent=2) + "\n", encoding="utf-8")
    elif t == "tools":
        out.write_text(json.dumps({"tools": get_openai_tool_schemas()}, indent=2) + "\n", encoding="utf-8")
    elif t == "openusd":
        spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        out.write_text(export_openusd_stage(spec), encoding="utf-8")
    elif t == "modelica":
        spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        out.write_text(export_modelica_component(spec), encoding="utf-8")
    elif t == "calculix":
        spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        export_calculix_inp(spec, out)
    elif t == "fmu":
        spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        export_fmu_package(spec, out)
    elif t == "nurbs":
        spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        out.write_text(export_exact_nurbs_step(spec), encoding="utf-8")
    else:
        raise typer.BadParameter(
            f"Unknown interop target '{target}'. Choose schemas, tools, openusd, modelica, calculix, fmu, or nurbs."
        )
    console.print(f"[bold green][OK][/bold green] Exported open interop target [cyan]{target}[/cyan] to [green]{out}[/green]")


@app.command()
def tpms(
    lattice_type: str = typer.Option("gyroid", "--type", help="TPMS family: gyroid, schwarz_p, diamond"),
    unit_cell_mm: float = typer.Option(8.0, "--cell", help="Unit cell wavelength (mm)"),
    porosity_inlet: float = typer.Option(0.72, "--porosity-in", help="Inlet porosity (0.35-0.92)"),
    porosity_outlet: float = typer.Option(0.65, "--porosity-out", help="Outlet porosity (0.35-0.92)"),
    flow_cfm: float = typer.Option(120.0, "--cfm", help="Volumetric airflow in CFM"),
    out_stl: Path = typer.Option(Path("tpms_core.stl"), "--out", "-o", help="Output binary STL path"),
):
    """Synthesize a 3D TPMS (Gyroid/Schwarz-P/Diamond) metamaterial heat-exchanger core and evaluate thermo-hydraulics."""
    spec = TPMSSpec(
        lattice_type=lattice_type,  # type: ignore[arg-type]
        unit_cell_mm=unit_cell_mm,
        porosity_inlet=porosity_inlet,
        porosity_outlet=porosity_outlet,
    )
    rep = evaluate_tpms_thermo_hydraulics(spec, flow_cfm=flow_cfm)
    tris = build_tpms_lattice_mesh(spec, grid_nx=12, grid_ny=10, grid_nz=10)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_stl.write_bytes(export_binary_stl(tris))
    console.print(f"[bold cyan]TPMS Metamaterial Core ({rep.lattice_type})[/bold cyan]")
    console.print(f"- Specific Surface Area: [green]{rep.specific_surface_area_m2_m3} m^2/m^3[/green] (Wetted: {rep.total_wetted_area_cm2} cm^2)")
    console.print(f"- Nusselt / Convective h:[green]{rep.nusselt_number}[/green] / [green]{rep.convective_h_w_m2k} W/m^2K[/green]")
    console.print(f"- Pressure Drop:         [yellow]{rep.pressure_drop_pa} Pa[/yellow] | Heat Capacity @ 40K: [bold green]{rep.heat_dissipation_at_40k_delta_w} W[/bold green]")
    console.print(f"[bold green][OK][/bold green] Exported {len(tris)} TPMS triangles to [cyan]{out_stl}[/cyan]")


@app.command()
def adjoint(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    steps: int = typer.Option(40, "--steps", help="PyTorch autograd optimization steps"),
    out: Path = typer.Option(Path("adjoint_codesign.json"), "--out", "-o", help="Output co-optimized JSON path"),
):
    """Run PyTorch end-to-end differentiable adjoint co-design of duct loft and finned heatsink."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    res = run_differentiable_adjoint_codesign(spec, steps=steps)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    console.print("[bold cyan]PyTorch Differentiable Adjoint Co-Design Complete[/bold cyan]")
    console.print(
        f"- Pressure Drop: [yellow]{res.initial_pressure_drop_pa} Pa[/yellow] -> "
        f"[green]{res.optimized_pressure_drop_pa} Pa[/green] (-{res.pressure_drop_reduction_pct}%)"
    )
    console.print(
        f"- Die Hotspot:   [yellow]{res.initial_hotspot_temp_c} C[/yellow] -> "
        f"[green]{res.optimized_hotspot_temp_c} C[/green] (-{res.hotspot_reduction_c} C)"
    )
    console.print(f"- Adjoint Grads: {res.adjoint_sensitivities}")


@app.command()
def provenance(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    out: Path = typer.Option(Path("forge_passport.json"), "--out", "-o", help="Output cryptographic passport JSON"),
):
    """Generate a cryptographic Hardware Bill of Materials (HBOM) and SLSA provenance attestation passport."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    passport = generate_hardware_provenance_passport(spec)
    verified = verify_hardware_provenance(passport)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(passport.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[bold green][OK][/bold green] Generated Cryptographic Passport [cyan]{passport.passport_id}[/cyan] "
        f"(Unit Cost: ${passport.total_estimated_unit_cost_usd:.2f}, Verified: {verified}) at [green]{out}[/green]"
    )


@app.command("nurbs-step")
def nurbs_step(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    out: Path = typer.Option(Path("duct_nurbs_ap242.stp"), "--out", "-o", help="Output STEP AP242 file path"),
):
    """Export exact ISO-10303-21 STEP AP242 CAD model with bicubic B-spline NURBS surfaces."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    export_exact_nurbs_step(spec, output_path=out)
    console.print(f"[bold green][OK][/bold green] Exported exact AP242 NURBS STEP to [cyan]{out}[/cyan]")


@app.command("hybrid-aalc")
def hybrid_aalc(
    spec_path: Path = typer.Argument(..., help="Path to FORGE duct JSON specification"),
    heat_load_w: float = typer.Option(800.0, "--tdp", help="Accelerator TDP in Watts"),
    fluid: str = typer.Option("water", "--fluid", help="Working fluid: water, methanol, acetone"),
    wick: str = typer.Option("sintered_copper", "--wick", help="Wick structure: sintered_copper, micro_grooved, screen_mesh"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Optional output JSON report path"),
):
    """Evaluate coupled two-phase vapor chamber and heat pipe hybrid cooling (AALC) limits."""
    spec = DuctSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    aalc_spec = HybridAALCSpec(
        heat_load_watts=heat_load_w,
        working_fluid=fluid,  # type: ignore[arg-type]
        wick_structure=wick,  # type: ignore[arg-type]
    )
    rep = solve_hybrid_aalc_cooling(spec, aalc_spec=aalc_spec)
    console.print(f"[bold cyan]Hybrid AALC Two-Phase Thermal Evaluation ({fluid.upper()} / {wick})[/bold cyan]")
    console.print(f"- Capillary Limit:       [green]{rep.capillary_limit_watts:.1f} W[/green]")
    console.print(f"- Boiling CHF Limit:     [green]{rep.boiling_limit_watts:.1f} W[/green]")
    console.print(f"- Entrainment Limit:     [green]{rep.entrainment_limit_watts:.1f} W[/green]")
    console.print(f"- Limiting Mechanism:    [yellow]{rep.limiting_mechanism}[/yellow]")
    console.print(f"- Margin to Dryout:      [bold green]{rep.thermal_margin_to_dryout_pct:.1f}%[/bold green]")
    hotspot_color = "red" if rep.throttling_triggered else "green"
    console.print(
        f"- Hotspot Junction Temp: [bold {hotspot_color}]{rep.hotspot_junction_temp_c:.1f} C[/bold] "
        f"(Margin: {rep.junction_margin_c:.1f} C)"
    )
    console.print(
        f"- Total System Delta-P:  [yellow]{rep.system_total_dp_pa:.1f} Pa[/yellow] "
        f"(Condenser: {rep.condenser_air_dp_pa:.1f} Pa)"
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.model_dump(), indent=2) + "\n", encoding="utf-8")
        console.print(f"[bold green][OK][/bold green] Report saved to [cyan]{out}[/cyan]")


@app.command("batch-screen")
def batch_screen(
    count: int = typer.Option(1000, "--count", "-n", help="Number of Monte Carlo candidate ducts to screen"),
    cfm: float = typer.Option(120.0, "--cfm", help="Volumetric airflow in CFM"),
):
    """Screen thousands of candidate duct configurations in parallel via vectorized PyTorch tensor SIMD."""
    import time
    import torch

    torch.manual_seed(42)
    inlet_w = torch.empty(count).uniform_(60.0, 140.0)
    inlet_h = torch.empty(count).uniform_(40.0, 100.0)
    outlet_w = torch.empty(count).uniform_(60.0, 180.0)
    outlet_h = torch.empty(count).uniform_(40.0, 120.0)
    length = torch.empty(count).uniform_(80.0, 250.0)
    wall = torch.empty(count).uniform_(1.6, 3.2)
    tensor_specs = torch.stack([inlet_w, inlet_h, outlet_w, outlet_h, length, wall], dim=1)

    cond = PhysicsConditions(volumetric_flow_cfm=cfm)
    t0 = time.perf_counter()
    res = evaluate_duct_specs_tensor_batch(tensor_specs, conditions=cond)
    t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    designs_per_sec = count / max(1e-6, (t_elapsed_ms / 1000.0))
    console.print("[bold cyan]Vectorized Tensor Batch Evaluation Complete[/bold cyan]")
    console.print(
        f"- Evaluated [bold green]{count}[/bold green] candidate designs in "
        f"[bold yellow]{t_elapsed_ms:.2f} ms[/bold yellow] ({designs_per_sec:.0f} designs/sec)"
    )
    console.print(
        f"- Mean Pressure Drop:   [green]{float(res.total_pressure_drop_pa.mean()):.2f} Pa[/green] "
        f"(Min: {float(res.total_pressure_drop_pa.min()):.2f} Pa, Max: {float(res.total_pressure_drop_pa.max()):.2f} Pa)"
    )
    console.print(f"- Mean Flow Uniformity: [green]{float(res.velocity_uniformity_index.mean()):.4f}[/green]")
    console.print(f"- Mean Shell Mass:      [green]{float(res.estimated_mass_grams.mean()):.1f} g[/green]")


@app.command()
def demo(
    out_dir: Path = typer.Option(Path("runs/demo"), "--out", "-o", help="Output directory for generated assets"),
):
    """Execute end-to-end multi-disciplinary hardware inverse design showcase."""
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print("[bold cyan]=================================================================[/bold cyan]")
    console.print("[bold cyan]  FORGE: Bleeding-Edge Physical AI Inverse Design Showcase       [/bold cyan]")
    console.print("[bold cyan]=================================================================[/bold cyan]\n")

    # 1. AI Copilot Synthesis
    console.print("[bold yellow]1. AI Copilot Synthesis & Obstacle Bypass...[/bold yellow]")
    prompt = "80mm Delta server fan to NVIDIA H100 PCIe accelerator, bypass 15mm RAM stick on left, PC-CF"
    copilot_res = synthesize_from_prompt(prompt, optimize_profile=True)
    spec = copilot_res.spec
    console.print(
        f"   [green][OK][/green] Synthesized {spec.inlet_width}x{spec.inlet_height} -> "
        f"{spec.outlet_width}x{spec.outlet_height}mm shroud ({spec.material.value})"
    )

    # 2. PyTorch Differentiable Adjoint Co-Design
    console.print("[bold yellow]2. PyTorch Differentiable Adjoint Co-Design...[/bold yellow]")
    adj = run_differentiable_adjoint_codesign(spec, steps=25)
    opt_spec = adj.optimized_spec
    console.print(
        f"   [green][OK][/green] Adjoint dP: {adj.initial_pressure_drop_pa:.2f} Pa -> "
        f"{adj.optimized_pressure_drop_pa:.2f} Pa (-{adj.pressure_drop_reduction_pct}%)"
    )
    console.print(
        f"   [green][OK][/green] Hotspot Die: {adj.initial_hotspot_temp_c:.1f} C -> "
        f"{adj.optimized_hotspot_temp_c:.1f} C (-{adj.hotspot_reduction_c:.1f} C)"
    )

    # 3. TPMS Metamaterial Core
    console.print("[bold yellow]3. TPMS Gyroid Metamaterial Synthesis...[/bold yellow]")
    tpms_spec = TPMSSpec(
        lattice_type="gyroid",
        width_mm=opt_spec.outlet_width,
        height_mm=opt_spec.outlet_height,
        length_mm=30.0,
    )
    tpms_rep = evaluate_tpms_thermo_hydraulics(tpms_spec)
    tpms_path = out_dir / "tpms_gyroid.stl"
    tpms_tris = build_tpms_lattice_mesh(tpms_spec, grid_nx=10, grid_ny=8, grid_nz=8)
    tpms_path.write_bytes(export_binary_stl(tpms_tris))
    console.print(
        f"   [green][OK][/green] Specific surface area: {tpms_rep.specific_surface_area_m2_m3:.1f} m^2/m^3 | "
        f"Heat Cap @40K: {tpms_rep.heat_dissipation_at_40k_delta_w:.1f} W"
    )

    # 4. Universal Open Standards Export
    console.print("[bold yellow]4. Universal Open Standards Digital Twin Export...[/bold yellow]")
    usd_path = out_dir / "forge_twin.usda"
    usd_path.write_text(export_openusd_stage(opt_spec), encoding="utf-8")
    mo_path = out_dir / "forge_component.mo"
    mo_path.write_text(export_modelica_component(opt_spec), encoding="utf-8")
    inp_path = export_calculix_inp(opt_spec, out_dir / "forge_fea.inp")
    fmu_path = export_fmu_package(opt_spec, out_dir / "forge_twin.fmu")
    console.print(f"   [green][OK][/green] OpenUSD Omniverse Stage : {usd_path}")
    console.print(f"   [green][OK][/green] Modelica 1D Component   : {mo_path}")
    console.print(f"   [green][OK][/green] CalculiX 3D FEA Deck    : {inp_path}")
    console.print(f"   [green][OK][/green] FMI 2.0 Digital Twin FMU: {fmu_path}")

    # 5. Cryptographic Hardware Passport
    console.print("[bold yellow]5. Cryptographic SLSA Provenance Passport...[/bold yellow]")
    passport = generate_hardware_provenance_passport(opt_spec)
    verified = verify_hardware_provenance(passport)
    pass_path = out_dir / "passport.json"
    pass_path.write_text(json.dumps(passport.model_dump(mode="json"), indent=2), encoding="utf-8")
    console.print(
        f"   [green][OK][/green] Passport ID: {passport.passport_id} (HMAC: {'VERIFIED [OK]' if verified else 'FAILED'})"
    )
    console.print(
        f"   [green][OK][/green] Unit Cost  : ${passport.total_estimated_unit_cost_usd:.2f} (Filament + Machine + Fasteners)"
    )

    console.print(f"\n[bold green]All showcase deliverables compiled in {out_dir}/[/bold green]")

    console.print(f"\n[bold green]All showcase deliverables compiled in {out_dir}/[/bold green]")


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
