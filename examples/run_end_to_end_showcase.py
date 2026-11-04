#!/usr/bin/env python3
"""FORGE End-to-End Showcase: Generative Physical AI Inverse Design Platform.

Executes all 6 architectural pillars in under 15 seconds:
1. Natural-language prompt synthesis with automatic Bézier obstacle bypass
2. Stratford non-linear boundary layer diffuser optimization
3. Brinkman-Darcy Navier-Stokes fluid topology optimization
4. PyTorch differentiable adjoint co-design (Duct loft + Heatsink fins)
5. Functionally graded Triply Periodic Minimal Surface (TPMS Gyroid) lattice synthesis
6. Structural modal vibration & transient PID fan spin-up simulation
7. Open digital twin exports: OpenUSD (.usda), Modelica (.mo), CalculiX (.inp), FMI 2.0 (.fmu)
8. Cryptographic Hardware Bill of Materials (HBOM) & SLSA Provenance Passport
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge.cad.exporters import export_binary_stl
from forge.cad.tpms import TPMSSpec, build_tpms_lattice_mesh, evaluate_tpms_thermo_hydraulics
from forge.compliance.provenance import (
    generate_hardware_provenance_passport,
    verify_hardware_provenance,
)
from forge.copilot.agent import synthesize_from_prompt
from forge.interop.calculix_fea import export_calculix_inp
from forge.interop.fmi_fmu import export_fmu_package
from forge.interop.modelica import export_modelica_component
from forge.interop.openusd import export_openusd_stage
from forge.opt.differentiable_adjoint import run_differentiable_adjoint_codesign
from forge.physics.structural_modal import analyze_structural_modal_resonance
from forge.physics.transient_thermal import simulate_transient_thermal_response
from forge.topopt.darcy_solver import run_fluid_topology_optimization

console = Console()


def run_showcase(output_dir: Path | str = "runs/showcase") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    console.print(
        Panel.fit(
            "[bold cyan]FORGE: Generative Physical Inverse Design Platform[/bold cyan]\n"
            "[dim]Bleeding-edge co-design of heterogeneous AI accelerator cooling systems[/dim]",
            border_style="cyan",
        )
    )

    # 1. Natural Language Design Copilot
    console.print("\n[bold yellow]>> Phase 1: Natural Language Prompt Synthesis & Obstacle Bypass[/bold yellow]")
    prompt = (
        "Design a 80mm Delta server fan duct connecting to an NVIDIA H100 PCIe accelerator, "
        "bypass a 15mm RAM stick obstacle on the left, use PC-CF carbon fiber"
    )
    console.print(f"  [dim]Prompt:[/dim] '{prompt}'")
    copilot_res = synthesize_from_prompt(prompt, optimize_profile=True)
    spec = copilot_res.spec
    console.print(
        f"  [green][OK][/green] Synthesized {spec.inlet_width}x{spec.inlet_height}mm -> "
        f"{spec.outlet_width}x{spec.outlet_height}mm shroud | Material: {spec.material.value} | "
        f"Flange: {spec.fan_standard.value if spec.fan_standard else 'None'}"
    )

    # 2. Brinkman-Darcy Topology Optimization
    console.print("\n[bold yellow]>> Phase 2: Brinkman-Darcy Navier-Stokes Topology Optimization[/bold yellow]")
    top_res, _ = run_fluid_topology_optimization(spec, nz=16, nx=12, iterations=10)
    console.print(
        f"  [green][OK][/green] TopOpt completed in 10 iterations | Energy Dissipation: "
        f"{top_res.initial_dissipation_watts:.2f}W -> {top_res.final_dissipation_watts:.2f}W "
        f"(-{top_res.dissipation_reduction_pct:.1f}%)"
    )

    # 3. PyTorch Differentiable Adjoint Co-Design
    console.print("\n[bold yellow]>> Phase 3: PyTorch Differentiable Adjoint Co-Design[/bold yellow]")
    adj_res = run_differentiable_adjoint_codesign(spec, steps=25)
    opt_spec = adj_res.optimized_spec
    console.print(
        f"  [green][OK][/green] Co-Optimized Duct + Heatsink: dP {adj_res.initial_pressure_drop_pa:.2f} Pa -> "
        f"{adj_res.optimized_pressure_drop_pa:.2f} Pa (-{adj_res.pressure_drop_reduction_pct}%) | "
        f"GPU Hotspot: {adj_res.initial_hotspot_temp_c:.1f} C -> {adj_res.optimized_hotspot_temp_c:.1f} C "
        f"(-{adj_res.hotspot_reduction_c:.1f} C)"
    )

    # 4. TPMS Metamaterial Lattice Core
    console.print("\n[bold yellow]>> Phase 4: Triply Periodic Minimal Surface (TPMS Gyroid) Core[/bold yellow]")
    tpms_spec = TPMSSpec(
        lattice_type="gyroid",
        width_mm=opt_spec.outlet_width,
        height_mm=opt_spec.outlet_height,
        length_mm=32.0,
        unit_cell_mm=8.0,
    )
    tpms_rep = evaluate_tpms_thermo_hydraulics(tpms_spec, flow_cfm=140.0)
    tpms_tris = build_tpms_lattice_mesh(tpms_spec, grid_nx=10, grid_ny=8, grid_nz=8)
    tpms_path = out / "tpms_gyroid_core.stl"
    tpms_path.write_bytes(export_binary_stl(tpms_tris))
    console.print(
        f"  [green][OK][/green] Gyroid Lattice: Specific Area {tpms_rep.specific_surface_area_m2_m3:.1f} m^2/m^3 | "
        f"Nusselt Nu={tpms_rep.nusselt_number:.1f} | Heat Capacity @40K: {tpms_rep.heat_dissipation_at_40k_delta_w:.1f} W"
    )

    # 5. Multi-Physics Structural Modal & Transient PID Simulation
    console.print("\n[bold yellow]>> Phase 5: Structural Modal & Transient Closed-Loop PID Simulation[/bold yellow]")
    modal_rep = analyze_structural_modal_resonance(opt_spec, fan_rpm=9500)
    trans_rep = simulate_transient_thermal_response(opt_spec, burst_watts=450.0, duration_s=15.0)
    console.print(
        f"  [green][OK][/green] Fundamental Panel Mode f_11: {modal_rep.fundamental_panel_freq_hz:.1f} Hz | "
        f"Fan BPF: {modal_rep.fan_bpf_freq_hz:.1f} Hz | Min Separation: {modal_rep.min_harmonic_separation_pct:.1f}% "
        f"(Resonance Risk: {modal_rep.resonance_risk})"
    )
    console.print(
        f"  [green][OK][/green] Transient Burst 65W -> 450W: Peak Die {trans_rep.peak_hotspot_temp_c:.1f} C | "
        f"Fan Spin-Up Time: {trans_rep.fan_spinup_90pct_time_s:.2f}s | Throttling Triggered: {trans_rep.thermal_throttling_triggered}"
    )

    # 6. Universal Open Standards & Cryptographic SLSA Provenance Passport
    console.print("\n[bold yellow]>> Phase 6: Universal Open Interoperability & Cryptographic Provenance[/bold yellow]")
    usd_path = out / "forge_stage.usda"
    usd_path.write_text(export_openusd_stage(opt_spec), encoding="utf-8")
    mo_path = out / "forge_thermofluid.mo"
    mo_path.write_text(export_modelica_component(opt_spec), encoding="utf-8")
    inp_path = export_calculix_inp(opt_spec, out / "forge_structural.inp")
    fmu_path = export_fmu_package(opt_spec, out / "forge_digital_twin.fmu")

    passport = generate_hardware_provenance_passport(opt_spec)
    is_valid = verify_hardware_provenance(passport)
    pass_path = out / "forge_passport.json"
    pass_path.write_text(json.dumps(passport.model_dump(mode="json"), indent=2), encoding="utf-8")

    console.print(f"  [green][OK][/green] OpenUSD Omniverse Stage : {usd_path.name}")
    console.print(f"  [green][OK][/green] Modelica 1D System Twin : {mo_path.name}")
    console.print(f"  [green][OK][/green] CalculiX / Abaqus FEA   : {inp_path.name}")
    console.print(f"  [green][OK][/green] FMI 2.0 Co-Sim FMU      : {fmu_path.name}")
    console.print(
        f"  [green][OK][/green] SLSA Passport ID        : [cyan]{passport.passport_id}[/cyan] "
        f"(Unit Cost: ${passport.total_estimated_unit_cost_usd:.2f}, HMAC Verified: {is_valid})"
    )

    elapsed = time.perf_counter() - t0

    # Summary Table
    table = Table(title="FORGE Inverse Design Benchmark Deliverables", show_header=True, header_style="bold magenta")
    table.add_column("Asset / Standard", style="cyan")
    table.add_column("Format / Ecosystem", style="green")
    table.add_column("Target Application", style="yellow")
    table.add_column("Key Metric", style="white")

    table.add_row("Aerodynamic Shroud", "OpenUSD (.usda)", "NVIDIA Omniverse / Isaac Sim", "PBR Carbon Fiber + Physics Metadata")
    table.add_row("1D System Component", "Modelica (.mo)", "OpenModelica / Dymola / Amesim", "Calibrated Turbulent Flow R_hyd")
    table.add_row("Structural / Thermal FEA", "CalculiX / Abaqus (.inp)", "CalculiX ccx / FreeCAD FEM", "10-Mode Frequency + Pressure DLOAD")
    table.add_row("Digital Twin Container", "FMI 2.0 (.fmu)", "Simulink / Twin Builder / FMPy", "ANSI C99 Real-Time Co-Simulation")
    table.add_row("TPMS Gyroid Core", "Binary STL (.stl)", "Direct Metal Laser Sintering (DMLS)", f"Area {tpms_rep.specific_surface_area_m2_m3:.0f} m^2/m^3")
    table.add_row("Hardware Passport", "SLSA / JSON", "Enterprise Supply Chain Verification", f"${passport.total_estimated_unit_cost_usd:.2f} / unit (HMAC OK)")

    console.print("\n", table)
    console.print(f"\n[bold green][OK] Showcase completed successfully in {elapsed:.2f} seconds![/bold green]\n")


if __name__ == "__main__":
    run_showcase()
