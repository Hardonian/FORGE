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
from forge.interop.mcp_server import (
    get_openai_tool_schemas,
    handle_mcp_jsonrpc_request,
    list_mcp_tools,
)
from forge.interop.open_schemas import export_all_json_schemas
from forge.cad.tpms import TPMSSpec, build_tpms_lattice_mesh, evaluate_tpms_thermo_hydraulics
from forge.opt.differentiable_adjoint import run_differentiable_adjoint_codesign
from forge.interop.calculix_fea import export_calculix_inp
from forge.interop.fmi_fmu import export_fmu_package
from forge.compliance.provenance import (
    generate_hardware_provenance_passport,
    verify_hardware_provenance,
)


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


def test_cad_plugins_modal_transient_and_compliance_dossier(tmp_path: Path):
    """Verify Onshape/FreeCAD/OpenSCAD codegen, structural modal solver, transient PID thermal sim, and UL94/ASHRAE dossier."""
    spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=80.0,
        outlet_width=110.0,
        outlet_height=42.0,
        length=85.0,
        wall=2.2,
        guide_vanes=1,
        material=DuctMaterial.PC_CF,
    )

    # 1. CAD Ecosystem Plugins
    fs_code = export_onshape_featurescript(spec)
    assert "FeatureScript 2411;" in fs_code
    assert "opLoft" in fs_code

    fc_code = export_freecad_macro(spec)
    assert "Part.makeLoft" in fc_code

    scad_code = export_openscad_script(spec)
    assert "polyhedron(" in scad_code

    # 2. Structural Modal Vibration & Harmonic Resonance
    modal_rep = analyze_structural_modal_resonance(spec, fan_rpm=9000, blade_count=7)
    assert modal_rep.fundamental_panel_freq_hz > 50.0
    assert modal_rep.effective_youngs_modulus_mpa > 1000.0
    assert len(modal_rep.higher_panel_modes_hz) == 3

    # 3. Transient Thermal & Closed-Loop PID Fan PWM Simulator
    trans_rep = simulate_transient_thermal_response(spec, idle_watts=60.0, burst_watts=400.0, duration_s=12.0, dt_s=1.0)
    assert len(trans_rep.time_s) == 13
    assert trans_rep.peak_hotspot_temp_c >= trans_rep.hotspot_die_temp_c[0]
    assert trans_rep.fan_pwm_pct[-1] > trans_rep.fan_pwm_pct[0]

    # 4. Enterprise Compliance Dossier (UL94, ASHRAE, NEBS)
    dossier = generate_compliance_dossier(spec)
    assert dossier.certificate_id.startswith("FORGE-CERT-")
    assert "UL94 V-0" in dossier.ul94_flammability_rating
    assert len(dossier.checks) == 5
    md_cert = export_compliance_dossier_markdown(dossier)
    assert "# FORGE Hardware Qualification & Compliance Dossier" in md_cert

    # 5. API & CLI endpoints
    client = TestClient(fastapi_app)
    spec_json = spec.model_dump(mode="json")

    r_mod = client.post("/api/v1/modal", json={"spec": spec_json})
    assert r_mod.status_code == 200
    assert r_mod.json()["fundamental_panel_freq_hz"] > 0.0

    r_tr = client.post("/api/v1/transient?duration_s=6.0", json=spec_json)
    assert r_tr.status_code == 200
    assert r_tr.json()["peak_hotspot_temp_c"] > 25.0

    r_comp = client.post("/api/v1/compliance", json={"spec": spec_json})
    assert r_comp.status_code == 200
    assert "FORGE-CERT-" in r_comp.json()["dossier"]["certificate_id"]

    r_cad = client.post("/api/v1/export/cad-script?target_cad=onshape", json=spec_json)
    assert r_cad.status_code == 200
    assert "FeatureScript" in r_cad.json()["script"]

    runner = CliRunner()
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec_json), encoding="utf-8")

    assert runner.invoke(cli_app, ["modal", str(spec_file)]).exit_code == 0
    assert runner.invoke(cli_app, ["transient", str(spec_file), "--duration", "6"]).exit_code == 0
    cert_file = tmp_path / "cert.md"
    assert runner.invoke(cli_app, ["compliance", str(spec_file), "--out", str(cert_file)]).exit_code == 0
    assert cert_file.exists()
    fs_file = tmp_path / "duct.fs"
    assert runner.invoke(cli_app, ["cad-script", str(spec_file), "--target", "onshape", "--out", str(fs_file)]).exit_code == 0
    assert fs_file.exists()


def test_open_interoperability_openusd_modelica_mcp_and_schemas(tmp_path: Path):
    """Verify OpenUSD (.usda), Modelica (.mo), MCP JSON-RPC 2.0 server, OpenAI tool schemas, and JSON Schemas."""
    spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=80.0,
        outlet_width=112.0,
        outlet_height=42.0,
        length=90.0,
        wall=2.0,
        material=DuctMaterial.PC_CF,
    )

    # 1. NVIDIA Omniverse / Pixar OpenUSD (.usda) stage
    usda = export_openusd_stage(spec)
    assert "#usda 1.0" in usda
    assert 'def Xform "FORGE_DigitalTwin"' in usda
    assert "forge:physics:totalPressureDropPa" in usda
    assert 'uniform token info:id = "UsdPreviewSurface"' in usda

    # 2. Modelica (.mo) 1D Thermo-Fluid Digital Twin
    mo = export_modelica_component(spec)
    assert "within FORGE.ThermalFluid;" in mo
    assert "K_turbulent" in mo
    assert "delta_p_Pa = p_in_Pa - p_out_Pa;" in mo

    # 3. Model Context Protocol (MCP) JSON-RPC 2.0 & OpenAI Function Schemas
    tools = list_mcp_tools()
    assert len(tools) >= 5
    openai_tools = get_openai_tool_schemas()
    assert openai_tools[0]["type"] == "function"

    init_resp = handle_mcp_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["result"]["serverInfo"]["name"] == "forge-mcp-server"

    list_resp = handle_mcp_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert len(list_resp["result"]["tools"]) >= 5

    res_read = handle_mcp_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "forge://presets/catalog"}}
    )
    assert "H100_PCIe_1U_SanAce40" in res_read["result"]["contents"][0]["text"]

    call_resp = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "forge_evaluate_duct",
                "arguments": {"spec": spec.model_dump(mode="json"), "volumetric_flow_cfm": 120.0},
            },
        }
    )
    assert call_resp["result"]["isError"] is False
    parsed_call = json.loads(call_resp["result"]["content"][0]["text"])
    assert parsed_call["physics"]["total_pressure_drop_pa"] > 0.0

    # 4. Open JSON-Schemas
    all_schemas = export_all_json_schemas()
    assert "DuctSpec" in all_schemas["schemas"]
    assert "ComplianceDossier" in all_schemas["schemas"]

    # 5. REST API & CLI Verification
    client = TestClient(fastapi_app)
    spec_json = spec.model_dump(mode="json")

    r_usda = client.post("/api/v1/export/openusd", json={"spec": spec_json})
    assert r_usda.status_code == 200
    assert "#usda 1.0" in r_usda.text

    r_mo = client.post("/api/v1/export/modelica", json={"spec": spec_json})
    assert r_mo.status_code == 200
    assert "within FORGE.ThermalFluid;" in r_mo.text

    r_mcp = client.post("/api/v1/mcp", json={"jsonrpc": "2.0", "id": 10, "method": "tools/list"})
    assert r_mcp.status_code == 200
    assert len(r_mcp.json()["result"]["tools"]) >= 5

    r_tools = client.get("/api/v1/tools/schema")
    assert r_tools.status_code == 200
    assert len(r_tools.json()["tools"]) >= 5

    r_schemas = client.get("/api/v1/schemas")
    assert r_schemas.status_code == 200
    assert "DuctSpec" in r_schemas.json()["schemas"]

    runner = CliRunner()
    out_sch = tmp_path / "schemas.json"
    assert runner.invoke(cli_app, ["interop", "--target", "schemas", "--out", str(out_sch)]).exit_code == 0
    assert out_sch.exists()


def test_phase6_tpms_adjoint_calculix_fmu_and_provenance(tmp_path: Path):
    """Verify Phase 6: TPMS Gyroid/Diamond metamaterial lattices, PyTorch differentiable adjoint co-design,
    CalculiX (.inp) FEA export, FMI 2.0 (.fmu) digital twin package, and Cryptographic HBOM Provenance Passport.
    """
    # 1. TPMS Gyroid & Diamond Metamaterial Lattice Core
    tpms_spec = TPMSSpec(
        lattice_type="gyroid",
        width_mm=60.0,
        height_mm=40.0,
        length_mm=28.0,
        unit_cell_mm=8.0,
        porosity_inlet=0.74,
        porosity_outlet=0.64,
    )
    tpms_tris = build_tpms_lattice_mesh(tpms_spec, grid_nx=10, grid_ny=8, grid_nz=8)
    assert len(tpms_tris) > 50
    tpms_rep = evaluate_tpms_thermo_hydraulics(tpms_spec, flow_cfm=130.0)
    assert tpms_rep.specific_surface_area_m2_m3 > 100.0
    assert tpms_rep.nusselt_number > 10.0
    assert tpms_rep.heat_dissipation_at_40k_delta_w > 150.0
    assert tpms_rep.self_supporting_printability is True

    # 2. PyTorch End-to-End Differentiable Adjoint Co-Design
    base_spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=80.0,
        outlet_width=136.0,
        outlet_height=62.0,
        length=95.0,
        wall=2.0,
        guide_vanes=0,
        material=DuctMaterial.PC_CF,
    )
    adj_res = run_differentiable_adjoint_codesign(base_spec, steps=35)
    assert len(adj_res.loss_trajectory) == 35
    assert adj_res.loss_trajectory[-1] < adj_res.loss_trajectory[0]
    assert "dLoss_dBezierW1" in adj_res.adjoint_sensitivities
    assert adj_res.optimized_pressure_drop_pa < adj_res.initial_pressure_drop_pa
    assert adj_res.optimized_hotspot_temp_c < adj_res.initial_hotspot_temp_c

    # 3. CalculiX / Abaqus (.inp) 3D Structural & Modal FEA Deck
    inp_file = tmp_path / "forge_duct.inp"
    export_calculix_inp(base_spec, inp_file)
    inp_text = inp_file.read_text(encoding="utf-8")
    assert "*NODE, NSET=Nall" in inp_text
    assert "*ELEMENT, TYPE=S3, ELSET=Eall" in inp_text
    assert "*MATERIAL, NAME=PC_CF" in inp_text
    assert "*FREQUENCY" in inp_text

    # 4. FMI 2.0 Co-Simulation (.fmu) Digital Twin Package
    fmu_file = tmp_path / "forge_duct.fmu"
    export_fmu_package(base_spec, fmu_file)
    assert fmu_file.exists()
    with zipfile.ZipFile(fmu_file, "r") as zf:
        names = set(zf.namelist())
        assert "modelDescription.xml" in names
        assert "sources/forge_duct_fmu.c" in names
        assert "resources/forge_spec.json" in names
        xml_text = zf.read("modelDescription.xml").decode("utf-8")
        assert 'fmiVersion="2.0"' in xml_text
        assert 'name="pressure_drop_pa"' in xml_text

    # 5. Cryptographic Hardware Bill of Materials (HBOM) & SLSA Provenance Passport
    passport = generate_hardware_provenance_passport(base_spec)
    assert passport.passport_id.startswith("FRG-PASS-")
    assert len(passport.bom_items) >= 2
    assert passport.total_estimated_unit_cost_usd > 0.0
    assert verify_hardware_provenance(passport) is True

    # Tampering with unit cost must immediately invalidate the HMAC-SHA256 signature
    tampered = passport.model_copy(
        update={"total_estimated_unit_cost_usd": passport.total_estimated_unit_cost_usd + 100.0}
    )
    assert verify_hardware_provenance(tampered) is False

    # 6. FastAPI & CLI End-to-End Verification
    client = TestClient(fastapi_app)
    spec_json = base_spec.model_dump(mode="json")

    r_tpms = client.post("/api/v1/tpms", json=tpms_spec.model_dump())
    assert r_tpms.status_code == 200
    assert r_tpms.json()["self_supporting_printability"] is True

    r_adj = client.post("/api/v1/opt/adjoint?steps=20", json={"spec": spec_json})
    assert r_adj.status_code == 200
    assert r_adj.json()["pressure_drop_reduction_pct"] > 0.0

    r_inp = client.post("/api/v1/export/calculix", json={"spec": spec_json})
    assert r_inp.status_code == 200
    assert "*ELEMENT, TYPE=S3" in r_inp.text

    r_fmu = client.post("/api/v1/export/fmu", json={"spec": spec_json})
    assert r_fmu.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r_fmu.content), "r") as zf_api:
        assert "modelDescription.xml" in zf_api.namelist()

    r_prov = client.post("/api/v1/compliance/provenance", json={"spec": spec_json})
    assert r_prov.status_code == 200
    assert r_prov.json()["signature_verified"] is True

    runner = CliRunner()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_json), encoding="utf-8")
    tpms_out = tmp_path / "tpms.stl"
    assert runner.invoke(cli_app, ["tpms", "--type", "gyroid", "--out", str(tpms_out)]).exit_code == 0
    assert tpms_out.exists()

    adj_out = tmp_path / "adj.json"
    assert runner.invoke(cli_app, ["adjoint", str(spec_path), "--steps", "15", "--out", str(adj_out)]).exit_code == 0
    assert adj_out.exists()

    prov_out = tmp_path / "passport.json"
    assert runner.invoke(cli_app, ["provenance", str(spec_path), "--out", str(prov_out)]).exit_code == 0
    assert prov_out.exists()




