"""End-to-end tests for Phase 1 (Obstacles, Slicer 3MF, NL Copilot),
Phase 2 (Brinkman-Darcy TopOpt, 3D FNO Flow Field Surrogate), and
Phase 3 (Cloud Entitlements, Enterprise Rack Digital Twin & ROI, API/CLI).
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import zipfile
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from forge.api.server import app as fastapi_app
from forge.cad.obstacles import check_obstacle_clearances, route_duct_around_obstacles
from forge.cad.slicer import export_production_slicer_3mf
from forge.cli.main import app as cli_app
from forge.cloud.rack_twin import IPMITelemetryFrame, evaluate_datacenter_rack_roi
from forge.cloud.tenant import CloudEntitlementManager, SubscriptionTier
from forge.copilot.agent import synthesize_from_prompt
from forge.core.models import DuctMaterial, DuctSpec, ObstacleBox, PhysicsConditions
from forge.surrogate.fno_field import predict_flow_field
from forge.topopt.darcy_solver import run_fluid_topology_optimization


def test_obstacle_clearance_and_auto_routing():
    """3D obstacle in mid-span should be detected and automatically bypassed via Bezier centerline routing."""
    obs = ObstacleBox(
        name="PCIe_Power_Cable",
        min_x=-45.0,
        max_x=-10.0,
        min_y=-25.0,
        max_y=25.0,
        min_z=40.0,
        max_z=60.0,
        clearance_mm=3.0,
    )
    spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=60.0,
        outlet_width=80.0,
        outlet_height=60.0,
        length=100.0,
        wall=2.0,
        obstacles=[obs],
    )
    initial_rep = check_obstacle_clearances(spec)
    assert not initial_rep.all_clear
    assert initial_rep.details[0].colliding

    routed_spec, routed_rep = route_duct_around_obstacles(spec)
    assert len(routed_spec.centerline_x_ctrl) == 2 or len(routed_spec.centerline_y_ctrl) == 2
    assert routed_rep.all_clear
    assert routed_rep.min_margin_mm >= 0.0


def test_production_slicer_3mf_and_interlocking_split():
    """Production slicer 3MF should embed filament settings and auto-split oversized ducts with interlocking flanges."""
    oversized = DuctSpec(
        inlet_width=100.0,
        inlet_height=80.0,
        outlet_width=120.0,
        outlet_height=60.0,
        length=290.0,  # Exceeds Bambu_Lab_X1C 256mm Z limit
        wall=2.2,
        material=DuctMaterial.PC_CF,
    )
    archive_bytes = export_production_slicer_3mf(oversized, printer_profile="Bambu_Lab_X1C", auto_split_if_oversized=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        names = zf.namelist()
        assert "3D/3dmodel.model" in names
        assert "Metadata/project_settings.config" in names
        assert "Metadata/model_settings.config" in names

        cfg = json.loads(zf.read("Metadata/project_settings.config").decode("utf-8"))
        assert cfg["auto_split_applied"] is True
        assert cfg["filament_type"] == "PC-CF"
        assert cfg["slicer_parameters"]["nozzle_temp_c"] == 290

        model_xml = zf.read("3D/3dmodel.model").decode("utf-8")
        assert 'object id="1"' in model_xml
        assert 'object id="2"' in model_xml


def test_natural_language_design_copilot():
    """NL Design Copilot should parse fan, GPU, material, CFM, and obstacle constraints end-to-end."""
    prompt = (
        "Design a cooling shroud connecting an 80mm Delta server fan to an RTX 4090 PCIe slot, "
        "bypass a 15mm RAM obstacle on the left, keep pressure drop under 120 Pa at 110 CFM, "
        "and format for Bambu Lab X1-Carbon in PC-CF."
    )
    res = synthesize_from_prompt(prompt)
    assert res.spec.inlet_width == 80.0
    assert res.spec.outlet_width == 136.0
    assert res.spec.material == DuctMaterial.PC_CF
    assert len(res.spec.obstacles) == 1
    assert res.clearance.all_clear
    assert res.physics.total_pressure_drop_pa > 0.0
    assert len(res.reasoning_trace) >= 4


def test_darcy_fluid_topology_optimization():
    """Brinkman-Darcy topology optimization should converge and extract an organic 3D-printable mesh."""
    spec = DuctSpec(
        inlet_width=60.0,
        inlet_height=60.0,
        outlet_width=130.0,
        outlet_height=80.0,
        length=90.0,
        wall=2.0,
    )
    conds = PhysicsConditions(volumetric_flow_cfm=150.0)
    res, tris = run_fluid_topology_optimization(spec, conds, nz=16, nx=12, iterations=12)

    assert len(res.density_field) == 16
    assert len(res.density_field[0]) == 12
    assert res.optimized_dp_pa <= res.baseline_dp_pa
    assert res.dissipation_reduction_pct >= 0.0
    assert len(tris) > 100


def test_fno_spatial_flow_field_surrogate():
    """2D/3D FNO spectral surrogate should predict spatial velocity, pressure, and separation maps."""
    spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=80.0,
        outlet_width=120.0,
        outlet_height=50.0,
        length=95.0,
        wall=2.0,
        offset_x=12.0,
    )
    pred = predict_flow_field(spec, nz=16, nx=12)
    assert pred.nz == 16
    assert pred.nx == 12
    assert len(pred.velocity_magnitude_m_s) == 16
    assert len(pred.velocity_magnitude_m_s[0]) == 12
    assert pred.peak_velocity_m_s > 0.0
    assert 0.0 <= pred.max_separation_risk <= 1.0


def test_cloud_entitlements_and_datacenter_rack_roi():
    """Cloud entitlement manager and enterprise datacenter rack digital twin should compute accurate metrics."""
    mgr = CloudEntitlementManager()
    mgr.register_tenant("startup-free", "Open Hardware Lab", SubscriptionTier.OPEN_CORE)

    assert mgr.verify_feature_access("startup-free", "evaluate_1d") is True
    assert mgr.verify_feature_access("startup-free", "topopt_darcy") is False
    assert mgr.verify_feature_access("default-pro", "topopt_darcy") is True
    assert mgr.verify_feature_access("enterprise-hyperscale", "datacenter_rack_twin") is True

    mgr.record_usage("default-pro", "topopt_darcy", compute_credits=2.5)
    bill = mgr.get_billing_summary("default-pro")
    assert bill["used_topopt_runs"] == 1
    assert bill["estimated_total_invoice_usd"] == 199.0

    spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=80.0,
        outlet_width=112.0,
        outlet_height=40.0,
        length=85.0,
        wall=2.0,
        guide_vanes=2,
        material=DuctMaterial.PC_CF,
    )
    ipmi = IPMITelemetryFrame(rack_id="RACK-H100-07", nodes_per_rack=8, gpus_per_node=8)
    roi = evaluate_datacenter_rack_roi(spec, ipmi=ipmi)
    assert roi.total_gpus_in_rack == 64
    assert roi.fan_power_saved_watts_per_rack > 0.0
    assert roi.cluster_100_rack_annual_savings_usd > 1000.0
    assert roi.payback_period_days > 0.0


def test_phase1_2_3_api_and_cli_integration(tmp_path: Path):
    """Verify REST API and CLI endpoints for Copilot, Slicer 3MF, TopOpt, Flow-Field, and Rack ROI."""
    client = TestClient(fastapi_app)

    r_cop = client.post(
        "/api/v1/copilot",
        params={"prompt": "80mm fan to RTX 4090 in PC-CF at 120 CFM"},
    )
    assert r_cop.status_code == 200
    cop_data = r_cop.json()
    spec_payload = cop_data["spec"]

    r_ff = client.post("/api/v1/flow-field", json={"spec": spec_payload})
    assert r_ff.status_code == 200
    assert r_ff.json()["peak_velocity_m_s"] > 0.0

    r_top = client.post("/api/v1/topopt?nz=12&nx=10&iterations=6", json={"spec": spec_payload})
    assert r_top.status_code == 200
    assert r_top.json()["triangle_count"] > 50

    r_roi = client.post("/api/v1/enterprise/rack-roi", json={"spec": spec_payload})
    assert r_roi.status_code == 200
    assert r_roi.json()["annual_savings_usd_per_rack"] > 0.0

    r_bill = client.get("/api/v1/cloud/billing/default-pro")
    assert r_bill.status_code == 200
    assert r_bill.json()["tier"] == "cloud_pro"

    r_slicer = client.post("/api/v1/export/slicer-3mf", json=spec_payload)
    assert r_slicer.status_code == 200
    assert len(r_slicer.content) > 200

    runner = CliRunner()
    cop_dir = tmp_path / "cop_out"
    res_cli_cop = runner.invoke(
        cli_app,
        ["copilot", "80mm fan to H100 PCIe in PC-CF", "--out", str(cop_dir)],
    )
    assert res_cli_cop.exit_code == 0
    assert (cop_dir / "synthesized_duct.json").exists()
    assert (cop_dir / "production_duct.3mf").exists()

    res_cli_roi = runner.invoke(cli_app, ["rack-roi", str(cop_dir / "synthesized_duct.json")])
    assert res_cli_roi.exit_code == 0

