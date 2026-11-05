"""Comprehensive unit and integration test suite for Hyperscale AI Datacenter systems:
1. Dual Counter-Rotating Fan Staging & N+1 Failure Backflow Damper Dynamics.
2. Heterogeneous Multi-Die CoWoS Package & Air Preheating Cascade Network.
3. Direct-to-Chip (D2C) Liquid Cold Plate Microchannel Hydraulics & CHF Margin.
4. Open Compute Project (OCP) ORV3 & OAM v2.0 Compliance Standards Engine.
5. Bayesian Sim2Real Inverse Parameter Estimation via MCMC.
6. REST API Endpoints & MCP Agentic Tool Integration.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from forge.api.server import app
from forge.compliance.ocp_standards import OCPFormFactor, evaluate_ocp_compliance
from forge.core.models import DuctSpec
from forge.interop.mcp_server import handle_mcp_jsonrpc_request
from forge.interop.open_schemas import export_all_json_schemas
from forge.lab.bayesian_mcmc import LabTelemetryPoint, run_bayesian_mcmc_calibration
from forge.physics.cold_plate import ColdPlateSpec, CoolantType, UQDSize, solve_cold_plate_hydraulics
from forge.physics.cowos_multidie import (
    ComponentDieSpec,
    CoWoSPackageSpec,
    DieType,
    solve_cowos_multidie_thermal_network,
)
from forge.physics.fan_redundancy import (
    AntiBackflowDamperSpec,
    FanBankSpec,
    FanUnitSpec,
    evaluate_fan_bank_redundancy,
    solve_fan_pressure_at_flow,
)

client = TestClient(app)


def test_fan_redundancy_counter_rotating_and_n_plus_one_backflow():
    """Verify dual fan swirl recovery, N+1 backflow leakage, and emergency failover RPM."""
    # 1. Swirl recovery staging boost
    single_fan = FanUnitSpec(is_counter_rotating_pair=False, shutoff_pressure_pa=400.0)
    dual_fan = FanUnitSpec(is_counter_rotating_pair=True, shutoff_pressure_pa=400.0)
    p_single, _ = solve_fan_pressure_at_flow(single_fan, 40.0, 100.0)
    p_dual, swirl_pa = solve_fan_pressure_at_flow(dual_fan, 40.0, 100.0)
    assert p_dual > 1.95 * p_single
    assert swirl_pa > 0.0

    # 2. Bank redundancy & backflow
    bank = FanBankSpec(
        fan=dual_fan,
        damper=AntiBackflowDamperSpec(equipped=True, leakage_fraction=0.03),
        total_fans=5,
        failed_fans=1,
        duty_cycle_pct=75.0,
        target_cooling_cfm=280.0,
    )
    rep = evaluate_fan_bank_redundancy(bank)

    assert rep.staged_dual_fan_pressure_multiplier >= 2.0
    assert rep.healthy_state.net_heatsink_cfm > rep.failed_state_uncompensated.net_heatsink_cfm
    assert rep.failed_state_uncompensated.reverse_backflow_cfm > 0.0
    # Damper limits leakage
    assert rep.failed_state_uncompensated.backflow_leakage_pct < 10.0
    # Failover response boosts RPM
    assert rep.failover_compensation.emergency_duty_pct >= bank.duty_cycle_pct
    assert rep.failover_compensation.emergency_rpm >= rep.healthy_state.fan_rpm


def test_cowos_multidie_heterogeneous_thermal_network():
    """Verify multi-die CoWoS package, air preheating cascade, and independent die throttling."""
    pkg = CoWoSPackageSpec(
        package_name="Test-CoWoS-Package",
        substrate_width_mm=100.0,
        substrate_length_mm=120.0,
        components=[
            ComponentDieSpec(
                name="VRM-Upstream",
                die_type=DieType.VRM_POWER_STAGE,
                power_w=80.0,
                width_mm=10.0,
                length_mm=60.0,
                streamwise_position_mm=10.0,
                max_junction_temp_c=115.0,
            ),
            ComponentDieSpec(
                name="HBM3e-Midstream",
                die_type=DieType.HBM3E_STACK,
                power_w=40.0,
                width_mm=11.0,
                length_mm=11.0,
                streamwise_position_mm=40.0,
                max_junction_temp_c=85.0,
            ),
            ComponentDieSpec(
                name="GPU-Downstream",
                die_type=DieType.GPU_LOGIC,
                power_w=650.0,
                width_mm=30.0,
                length_mm=25.0,
                streamwise_position_mm=75.0,
                max_junction_temp_c=105.0,
            ),
        ],
    )

    rep = solve_cowos_multidie_thermal_network(pkg, airflow_cfm=150.0, ambient_temp_c=35.0)

    assert rep.total_package_power_w == 770.0
    assert rep.bulk_air_exit_temp_c > rep.ambient_air_temp_c
    assert len(rep.components) == 3

    # Upstream air preheat check: downstream dies must see higher inlet air temp
    vrm = next(c for c in rep.components if c.name == "VRM-Upstream")
    hbm = next(c for c in rep.components if c.name == "HBM3e-Midstream")
    gpu = next(c for c in rep.components if c.name == "GPU-Downstream")

    assert vrm.local_air_inlet_temp_c <= hbm.local_air_inlet_temp_c
    assert hbm.local_air_inlet_temp_c <= gpu.local_air_inlet_temp_c
    assert gpu.preheat_delta_t_c > 0.0
    assert rep.critical_bottleneck_component != ""


def test_cold_plate_microchannel_hydraulics_and_chf():
    """Verify skived microchannel liquid cold plate hydraulics, UQD couplings, and CHF margin."""
    cp_spec = ColdPlateSpec(
        coolant=CoolantType.PG25,
        flow_rate_lpm=4.0,
        coolant_inlet_temp_c=30.0,
        heat_load_w=1200.0,
        plate_width_mm=75.0,
        plate_length_mm=80.0,
        fin_thickness_mm=0.20,
        channel_width_mm=0.20,
        channel_height_mm=3.50,
        uqd_coupling=UQDSize.UQD04,
    )

    res = solve_cold_plate_hydraulics(cp_spec)

    assert res.number_of_channels >= 150
    assert res.channel_aspect_ratio == 17.5
    assert res.hydraulic_diameter_um > 300.0  # ~378 um
    assert res.reynolds_number > 50.0
    assert res.heat_transfer_coeff_w_m2k > 8000.0
    assert 0.0 < res.fin_efficiency_pct <= 100.0
    assert res.total_pressure_drop_kpa > 0.0
    assert res.uqd_coupling_pressure_drop_kpa > 0.0
    assert res.heat_flux_w_cm2 == 20.0
    assert res.chf_safety_margin_ratio > 2.0
    assert res.is_chf_safe is True


def test_ocp_orv3_and_oam_v2_compliance_engine():
    """Verify OCP ORV3 1OU/2OU and OAM v2.0 formal compliance verification and attestation."""
    # 1. Compliant 2OU Duct Spec
    spec_compliant = DuctSpec(
        inlet_width=100.0,
        inlet_height=70.0,
        outlet_width=100.0,
        outlet_height=70.0,
        length=120.0,
        wall=2.0,
    )
    rep_2ou = evaluate_ocp_compliance(spec_compliant, form_factor=OCPFormFactor.ORV3_2OU)
    assert rep_2ou.overall_compliance_status == "CERTIFIED_COMPLIANT"
    assert rep_2ou.compliance_score_pct == 100.0
    assert rep_2ou.rules_passed == 6
    assert rep_2ou.certificate_id.startswith("OCP-CERT-")
    assert len(rep_2ou.sha256_attestation) == 64

    # 2. Non-compliant height against 1OU limit (44mm)
    rep_1ou = evaluate_ocp_compliance(spec_compliant, form_factor=OCPFormFactor.ORV3_1OU)
    assert rep_1ou.overall_compliance_status == "REJECTED_NON_COMPLIANT"
    height_rule = next(r for r in rep_1ou.rules if r.rule_id == "OCP-ENV-01")
    assert height_rule.passed is False
    assert height_rule.margin_of_safety_pct < 0.0


def test_bayesian_sim2real_mcmc_calibration():
    """Verify Bayesian MCMC parameter estimation and uncertainty reduction."""
    spec = DuctSpec(
        inlet_width=80.0,
        inlet_height=80.0,
        outlet_width=120.0,
        outlet_height=60.0,
        length=140.0,
        wall=2.0,
    )
    telemetry = [
        LabTelemetryPoint(flow_cfm=80.0, measured_pressure_drop_pa=68.0, measured_hotspot_temp_c=78.5),
        LabTelemetryPoint(flow_cfm=120.0, measured_pressure_drop_pa=145.0, measured_hotspot_temp_c=71.0),
        LabTelemetryPoint(flow_cfm=160.0, measured_pressure_drop_pa=245.0, measured_hotspot_temp_c=67.0),
    ]

    rep = run_bayesian_mcmc_calibration(spec, telemetry=telemetry, iterations=250, burn_in=75, seed=123)

    assert rep.total_mcmc_iterations == 250
    assert rep.burn_in_iterations == 75
    assert 0.0 < rep.acceptance_rate_pct <= 100.0
    assert "roughness_um" in rep.parameters
    assert "r_tim_k_w" in rep.parameters
    assert "fan_install_factor" in rep.parameters

    p_rough = rep.parameters["roughness_um"]
    assert p_rough.ci_95_lower <= p_rough.posterior_mean <= p_rough.ci_95_upper
    assert p_rough.posterior_std > 0.0
    assert rep.post_calibration_rmse_c < rep.pre_calibration_rmse_c


def test_api_endpoints_hyperscale_systems():
    """Verify REST API endpoints for all 5 new hyperscale subsystems."""
    # 1. Fan Redundancy Endpoint
    r_fan = client.post(
        "/api/v1/physics/fan-redundancy",
        json={"total_fans": 5, "failed_fans": 1, "duty_cycle_pct": 75.0, "target_cooling_cfm": 300.0},
    )
    assert r_fan.status_code == 200
    assert "staged_dual_fan_pressure_multiplier" in r_fan.json()

    # 2. CoWoS Multi-Die Endpoint
    r_cowos = client.post(
        "/api/v1/physics/cowos-multidie",
        json={"spec": CoWoSPackageSpec().model_dump(), "airflow_cfm": 140.0, "ambient_temp_c": 35.0},
    )
    assert r_cowos.status_code == 200
    assert "critical_bottleneck_component" in r_cowos.json()

    # 3. Cold Plate Endpoint
    r_cp = client.post(
        "/api/v1/physics/cold-plate",
        json=ColdPlateSpec(coolant=CoolantType.PG25, flow_rate_lpm=3.5, heat_load_w=1000.0).model_dump(),
    )
    assert r_cp.status_code == 200
    assert "chf_safety_margin_ratio" in r_cp.json()

    # 4. OCP Compliance Check Endpoint
    spec_dict = DuctSpec(
        inlet_width=80.0, inlet_height=80.0, outlet_width=100.0, outlet_height=42.0, length=140.0, wall=2.0
    ).model_dump()
    r_ocp = client.post(
        "/api/v1/compliance/ocp-check?form_factor=ocp_oam_v2",
        json={"spec": spec_dict},
    )
    assert r_ocp.status_code == 200
    assert "certificate_id" in r_ocp.json()

    # 5. Bayesian Calibration Endpoint
    r_cal = client.post(
        "/api/v1/lab/bayesian-calibration?iterations=150&burn_in=40",
        json={"spec": spec_dict},
    )
    assert r_cal.status_code == 200
    assert "acceptance_rate_pct" in r_cal.json()


def test_mcp_hyperscale_tools():
    """Verify MCP Agentic tool execution for all 5 new hyperscale tools."""
    spec_dict = DuctSpec(
        inlet_width=80.0, inlet_height=80.0, outlet_width=100.0, outlet_height=42.0, length=140.0, wall=2.0
    ).model_dump()

    # 1. MCP fan redundancy
    mcp_res = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "test-1",
            "method": "tools/call",
            "params": {"name": "forge_fan_redundancy", "arguments": {"total_fans": 6, "failed_fans": 1}},
        }
    )
    assert mcp_res["result"]["isError"] is False

    # 2. MCP cowos multidie
    mcp_res = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "test-2",
            "method": "tools/call",
            "params": {"name": "forge_cowos_multidie", "arguments": {"airflow_cfm": 150.0}},
        }
    )
    assert mcp_res["result"]["isError"] is False

    # 3. MCP cold plate
    mcp_res = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "test-3",
            "method": "tools/call",
            "params": {"name": "forge_cold_plate_hydraulics", "arguments": {"coolant": "pg25", "flow_rate_lpm": 3.0}},
        }
    )
    assert mcp_res["result"]["isError"] is False

    # 4. MCP ocp compliance
    mcp_res = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "test-4",
            "method": "tools/call",
            "params": {"name": "forge_ocp_compliance", "arguments": {"spec": spec_dict, "form_factor": "ocp_oam_v2"}},
        }
    )
    assert mcp_res["result"]["isError"] is False

    # 5. MCP bayesian calibration
    mcp_res = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "test-5",
            "method": "tools/call",
            "params": {"name": "forge_bayesian_calibration", "arguments": {"spec": spec_dict, "iterations": 100}},
        }
    )
    assert mcp_res["result"]["isError"] is False


def test_open_schemas_registry_expansion_to_28():
    """Verify that export_all_json_schemas includes all 28 domain models."""
    schemas = export_all_json_schemas()
    assert len(schemas["schemas"]) >= 28
    assert "FanBankSpec" in schemas["schemas"]
    assert "CoWoSPackageSpec" in schemas["schemas"]
    assert "ColdPlateSpec" in schemas["schemas"]
    assert "OCPComplianceReport" in schemas["schemas"]
    assert "BayesianCalibrationReport" in schemas["schemas"]
