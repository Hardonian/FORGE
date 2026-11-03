"""Tests for next-generation long-term capabilities:
- Multi-station Bezier diffuser optimization
- Coupled Conjugate Heat Transfer (CHT) finned heatsink solver
- Multi-branch 1-to-N splitter manifold CAD & parallel hydraulic network solver
- Swirl-canceling honeycomb stator mesh & hydraulics
- Helmholtz / Quarter-Wave acoustic metamaterial liner design
- Persistent SQLite experiment registry & hardware preset catalog
"""
from __future__ import annotations
from pathlib import Path
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from forge.api.server import app as fastapi_app
from forge.cad.honeycomb import (
    HoneycombStatorSpec,
    evaluate_honeycomb_hydraulics,
    generate_honeycomb_stator_mesh,
)
from forge.cad.manifold import ManifoldBranch, ManifoldSpec, build_manifold_mesh
from forge.cad.mesh import build_duct_mesh
from forge.cli.main import app as cli_app
from forge.core.models import CrossSectionShape, DuctSpec, PhysicsConditions, ServerFanStandard
from forge.opt.bezier_diffuser import optimize_bezier_diffuser
from forge.physics.acoustics import design_acoustic_metamaterial_liner
from forge.physics.heatsink import (
    HeatsinkSpec,
    evaluate_heatsink_at_cfm,
    solve_coupled_duct_heatsink_system,
)
from forge.physics.network_solver import auto_balance_manifold, solve_manifold_network
from forge.physics.solver_1d import solve_duct_physics
from forge.presets.catalog import get_preset_spec, list_presets
from forge.registry.store import ForgeRegistry


def test_bezier_diffuser_optimization_reduces_loss():
    """Stratford / bell-mouth Bezier diffuser optimization should reduce expansion loss and build a valid mesh."""
    base = DuctSpec(
        inlet_width=60.0,
        inlet_height=60.0,
        outlet_width=140.0,
        outlet_height=100.0,
        length=80.0,
        wall=2.0,
    )
    conds = PhysicsConditions(volumetric_flow_cfm=160.0)
    opt_spec, opt_metrics, summary = optimize_bezier_diffuser(
        base, conds, num_control_points=2, iterations=20
    )

    assert len(opt_spec.width_profile_ctrl) == 2
    assert len(opt_spec.height_profile_ctrl) == 2
    assert opt_metrics.total_pressure_drop_pa <= summary["baseline_dp_pa"]
    assert summary["dp_reduction_pct"] >= 0.0

    tris = build_duct_mesh(opt_spec)
    assert len(tris) > 100


def test_coupled_conjugate_heat_transfer_heatsink():
    """Coupled Fan + Duct + Heatsink CHT solver should find equilibrium CFM and die temperatures."""
    duct = DuctSpec(
        inlet_width=120.0,
        inlet_height=120.0,
        outlet_width=110.0,
        outlet_height=42.0,
        length=95.0,
        wall=2.0,
        inlet_shape=CrossSectionShape.CIRCULAR,
        fan_standard=ServerFanStandard.FAN_120MM,
    )
    hs = HeatsinkSpec(width_mm=110.0, fin_height_mm=38.0, length_mm=180.0, fin_count=54, material="COPPER")
    conds = PhysicsConditions(heat_dissipation_watts=400.0, air_temp_c=25.0)

    rep = solve_coupled_duct_heatsink_system(duct, hs, conds)
    assert 10.0 < rep.equilibrium_flow_cfm < 190.0
    assert rep.duct_pressure_drop_pa > 0.0
    assert rep.heatsink_pressure_drop_pa > 0.0
    assert abs(rep.total_system_pressure_drop_pa - (rep.duct_pressure_drop_pa + rep.heatsink_pressure_drop_pa)) < 0.05
    assert 0.5 < rep.fin_efficiency <= 1.0
    assert rep.hotspot_die_temp_c >= rep.mean_die_temp_c > 25.0

    # Verify that higher flow uniformity reduces hotspot die temperature
    rep_poor_uni = evaluate_heatsink_at_cfm(hs, conds, flow_cfm=100.0, flow_uniformity_index=0.55)
    rep_high_uni = evaluate_heatsink_at_cfm(hs, conds, flow_cfm=100.0, flow_uniformity_index=0.95)
    assert rep_high_uni.hotspot_die_temp_c < rep_poor_uni.hotspot_die_temp_c


def test_manifold_network_solver_and_auto_balancing():
    """1-to-N splitter manifold solver should conserve flow and auto-balance asymmetric GPU heat loads."""
    manifold = ManifoldSpec(
        inlet_width=120.0,
        inlet_height=120.0,
        inlet_shape=CrossSectionShape.CIRCULAR,
        plenum_length=35.0,
        wall=2.0,
        branches=[
            ManifoldBranch(
                name="GPU_0_HighLoad",
                outlet_width=60.0,
                outlet_height=40.0,
                offset_x=-40.0,
                length=85.0,
                heat_load_watts=500.0,
                downstream_k_loss=1.8,
            ),
            ManifoldBranch(
                name="GPU_1_LowLoad",
                outlet_width=60.0,
                outlet_height=40.0,
                offset_x=40.0,
                length=85.0,
                heat_load_watts=200.0,
                downstream_k_loss=1.8,
            ),
        ],
    )

    tris = build_manifold_mesh(manifold)
    assert len(tris) > 200

    net = solve_manifold_network(manifold, total_flow_cfm=180.0)
    total_q = sum(b.flow_cfm for b in net.branches)
    assert abs(total_q - 180.0) < 0.5

    balanced_spec, balanced_net, summary = auto_balance_manifold(manifold, total_flow_cfm=180.0)
    assert balanced_net.thermal_balance_index > net.thermal_balance_index
    assert summary["balanced_thermal_balance_index"] > summary["initial_thermal_balance_index"]
    # High-load GPU_0 should have been widened relative to low-load GPU_1
    assert balanced_spec.branches[0].outlet_width > balanced_spec.branches[1].outlet_width


def test_honeycomb_stator_mesh_and_hydraulics():
    """Honeycomb flow-straightener stator should generate watertight ribs and >80% swirl reduction."""
    spec = HoneycombStatorSpec(width_mm=80.0, height_mm=80.0, axial_length_mm=15.0, cell_pitch_mm=12.0, cell_wall_mm=0.8)
    tris = generate_honeycomb_stator_mesh(spec)
    assert len(tris) > 40

    hyd = evaluate_honeycomb_hydraulics(spec, velocity_m_s=12.0)
    assert 0.5 < hyd["porosity"] < 0.95
    assert hyd["pressure_drop_pa"] > 0.0
    assert hyd["swirl_attenuation_pct"] > 80.0


def test_acoustic_metamaterial_liner_design():
    """Helmholtz / quarter-wave liner should tune to exact fan BPF and achieve positive transmission loss."""
    liner = design_acoustic_metamaterial_liner(fan_rpm=9000, blade_count=7, air_temp_c=30.0, max_cavity_depth_mm=10.0)
    # 9000 RPM / 60 * 7 blades = 1050 Hz
    assert abs(liner.blade_passing_freq_hz - 1050.0) < 0.1
    assert liner.quarter_wave_length_mm > 50.0
    assert liner.peak_transmission_loss_db > 8.0
    assert liner.effective_stopband_hz[0] < 1050.0 < liner.effective_stopband_hz[1]


def test_sqlite_registry_persistence_and_queries():
    """ForgeRegistry should store, deduplicate, query, and export DuctSpecs for surrogate training."""
    reg = ForgeRegistry(":memory:")
    s1 = DuctSpec(inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50, length=100, wall=2.0)
    s2 = DuctSpec(inlet_width=80, inlet_height=80, outlet_width=90, outlet_height=60, length=110, wall=2.0)
    conds = PhysicsConditions()

    h1 = reg.record_evaluation(s1, solve_duct_physics(s1, conds), tag="run_a")
    h2 = reg.record_evaluation(s2, solve_duct_physics(s2, conds), tag="run_a")
    assert reg.count_designs() == 2

    # Deduplication on identical spec
    h1_dup = reg.record_evaluation(s1, solve_duct_physics(s1, conds), tag="run_b")
    assert h1 == h1_dup
    assert reg.count_designs() == 2

    fetched = reg.get_design(h2)
    assert fetched is not None
    assert fetched["spec"]["inlet_width"] == 80.0

    top = reg.list_top_designs(limit=5)
    assert len(top) == 2
    assert top[0]["dp_pa"] <= top[1]["dp_pa"]

    exported = reg.export_surrogate_training_specs()
    assert len(exported) == 2
    reg.close()


def test_presets_and_new_api_cli_endpoints(tmp_path: Path):
    """Hardware presets and new REST API & CLI endpoints should execute cleanly."""
    presets = list_presets()
    assert len(presets) >= 5
    h100 = get_preset_spec("H100_PCIe_1U_SanAce40")
    assert h100["spec"].inlet_width == 40.0

    client = TestClient(fastapi_app)
    r_presets = client.get("/api/v1/presets")
    assert r_presets.status_code == 200
    assert len(r_presets.json()["presets"]) >= 5

    r_acoustics = client.post("/api/v1/acoustics?fan_rpm=8500&blade_count=7")
    assert r_acoustics.status_code == 200
    assert r_acoustics.json()["blade_passing_freq_hz"] > 900.0

    r_hs = client.post("/api/v1/heatsink", json={"duct": h100["spec"].model_dump(mode="json")})
    assert r_hs.status_code == 200
    assert r_hs.json()["equilibrium_flow_cfm"] > 0.0

    runner = CliRunner()
    out_preset = tmp_path / "h100.json"
    res_cli = runner.invoke(cli_app, ["presets", "--export", "H100_PCIe_1U_SanAce40", "--out", str(out_preset)])
    assert res_cli.exit_code == 0
    assert out_preset.exists()

    res_hs_cli = runner.invoke(cli_app, ["heatsink", str(out_preset), "--watts", "350"])
    assert res_hs_cli.exit_code == 0

    res_ac_cli = runner.invoke(cli_app, ["acoustics", "--rpm", "9000", "--blades", "7"])
    assert res_ac_cli.exit_code == 0

