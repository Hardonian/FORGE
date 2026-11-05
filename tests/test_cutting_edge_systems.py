"""Tests for cutting-edge generative CAD, two-phase cooling, vectorized physics, and digital twin streaming."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from forge.api.server import app
from forge.api.websocket_twin import (
    DigitalTwinConnectionManager,
    DigitalTwinOnlineTracker,
    LiveTelemetryFrame,
)
from forge.cad.nurbs_step import export_exact_nurbs_step
from forge.core.models import DuctSpec, PhysicsConditions
from forge.interop.mcp_server import handle_mcp_jsonrpc_request, list_mcp_tools
from forge.interop.open_schemas import export_all_json_schemas
from forge.physics.hybrid_cooling import (
    HybridAALCSpec,
    solve_hybrid_aalc_cooling,
)
from forge.physics.vectorized import (
    BatchPhysicsResult,
    evaluate_duct_specs_tensor_batch,
)


@pytest.fixture
def sample_spec() -> DuctSpec:
    return DuctSpec(
        inlet_width=100.0,
        inlet_height=60.0,
        outlet_width=120.0,
        outlet_height=75.0,
        length=140.0,
        wall=2.4,
        offset_x=15.0,
        offset_y=10.0,
    )


def test_nurbs_step_ap242_exporter(sample_spec: DuctSpec, tmp_path: Path):
    """Verify exact ISO-10303-21 STEP AP242 B-spline NURBS surface generation."""
    out_file = tmp_path / "test_duct_nurbs.stp"
    step_content = export_exact_nurbs_step(sample_spec, output_path=out_file)

    assert out_file.exists()
    assert step_content.startswith("ISO-10303-21;")
    assert "HEADER;" in step_content
    assert "AP242_MANAGED_MODEL_EDITION" in step_content
    assert "DATA;" in step_content
    assert "B_SPLINE_SURFACE_WITH_KNOTS" in step_content
    assert "OuterDuctSkin" in step_content
    assert "InnerDuctBore" in step_content
    assert "ADVANCED_FACE" in step_content
    assert "CLOSED_SHELL" in step_content
    assert "MANIFOLD_SOLID_BREP" in step_content
    assert "ADVANCED_BREP_SHAPE_REPRESENTATION" in step_content
    assert step_content.strip().endswith("END-ISO-10303-21;")


def test_hybrid_aalc_cooling_solver(sample_spec: DuctSpec):
    """Verify two-phase vapor chamber and heat pipe hybrid thermal limits."""
    aalc_spec = HybridAALCSpec(
        heat_load_watts=750.0,
        working_fluid="water",
        wick_structure="sintered_copper",
    )
    rep = solve_hybrid_aalc_cooling(sample_spec, aalc_spec=aalc_spec)

    assert rep.operating_power_watts == 750.0
    assert rep.capillary_limit_watts > 500.0
    assert rep.boiling_limit_watts > 500.0
    assert rep.entrainment_limit_watts > 500.0
    assert rep.thermal_margin_to_dryout_pct > 0.0
    assert not rep.two_phase_dryout_risk
    assert rep.hotspot_junction_temp_c > 25.0
    assert rep.junction_margin_c > 0.0
    assert rep.system_total_dp_pa > 0.0

    # Test extreme thermal overload (1800W) triggering throttling / dryout
    extreme_spec = HybridAALCSpec(
        heat_load_watts=2200.0,
        working_fluid="water",
        wick_structure="micro_grooved",
    )
    rep_extreme = solve_hybrid_aalc_cooling(sample_spec, aalc_spec=extreme_spec)
    assert rep_extreme.throttling_triggered or rep_extreme.two_phase_dryout_risk

    # Test alternate fluids: methanol and acetone
    rep_methanol = solve_hybrid_aalc_cooling(
        sample_spec,
        aalc_spec=HybridAALCSpec(heat_load_watts=400.0, working_fluid="methanol"),
    )
    assert rep_methanol.hotspot_junction_temp_c > 0.0

    rep_acetone = solve_hybrid_aalc_cooling(
        sample_spec,
        aalc_spec=HybridAALCSpec(heat_load_watts=400.0, working_fluid="acetone"),
    )
    assert rep_acetone.hotspot_junction_temp_c > 0.0


def test_vectorized_tensor_batch_physics(sample_spec: DuctSpec):
    """Verify PyTorch SIMD vectorized batch evaluation on CPU tensors."""
    # 1. Test with python list of DuctSpec
    spec_list = [sample_spec, sample_spec]
    res_list = evaluate_duct_specs_tensor_batch(spec_list)
    assert isinstance(res_list, BatchPhysicsResult)
    assert res_list.total_pressure_drop_pa.shape == (2,)
    assert (res_list.total_pressure_drop_pa > 0.0).all()
    assert (res_list.velocity_uniformity_index > 0.35).all()
    assert (res_list.velocity_uniformity_index <= 1.0).all()

    # 2. Test large tensor batch of 2,000 candidate configurations
    torch.manual_seed(123)
    batch_size = 2000
    tensor_data = torch.stack(
        [
            torch.empty(batch_size).uniform_(80.0, 140.0),
            torch.empty(batch_size).uniform_(40.0, 90.0),
            torch.empty(batch_size).uniform_(80.0, 160.0),
            torch.empty(batch_size).uniform_(50.0, 100.0),
            torch.empty(batch_size).uniform_(100.0, 200.0),
            torch.empty(batch_size).uniform_(1.8, 3.0),
        ],
        dim=1,
    )

    cond = PhysicsConditions(volumetric_flow_cfm=140.0)
    batch_res = evaluate_duct_specs_tensor_batch(tensor_data, conditions=cond)

    assert batch_res.total_pressure_drop_pa.shape == (batch_size,)
    assert batch_res.velocity_uniformity_index.shape == (batch_size,)
    assert batch_res.reynolds_number.shape == (batch_size,)
    assert batch_res.estimated_mass_grams.shape == (batch_size,)

    assert not torch.isnan(batch_res.total_pressure_drop_pa).any()
    assert not torch.isnan(batch_res.velocity_uniformity_index).any()
    assert (batch_res.estimated_mass_grams > 5.0).all()


def test_digital_twin_online_tracker_and_connection_manager(sample_spec: DuctSpec):
    """Verify digital twin online tracking of dust loading and TIM degradation."""
    tracker = DigitalTwinOnlineTracker(spec=sample_spec)

    # Clean initial state
    frame0 = LiveTelemetryFrame(
        fan_rpm=9000.0,
        fan_pwm_pct=50.0,
        inlet_ambient_temp_c=25.0,
        gpu_hotspot_temp_c=68.0,
        board_power_watts=400.0,
    )
    s0 = tracker.process_frame(frame0)
    assert s0.filter_dust_loading_factor == 1.0
    assert not s0.filter_service_recommended
    assert not s0.thermal_throttling_risk
    assert s0.health_score_pct >= 90.0

    # Simulate dirty filter / high measured differential pressure
    for _ in range(5):
        frame_dust = LiveTelemetryFrame(
            fan_rpm=9000.0,
            fan_pwm_pct=50.0,
            inlet_ambient_temp_c=25.0,
            gpu_hotspot_temp_c=72.0,
            board_power_watts=400.0,
            measured_differential_pressure_pa=s0.nominal_clean_dp_pa * 1.6,
        )
        s_dust = tracker.process_frame(frame_dust)

    assert s_dust.filter_dust_loading_factor > 1.20
    assert s_dust.filter_service_recommended

    # Simulate thermal degradation (hotspot 84°C)
    for _ in range(5):
        frame_thermal = LiveTelemetryFrame(
            fan_rpm=9000.0,
            fan_pwm_pct=50.0,
            inlet_ambient_temp_c=25.0,
            gpu_hotspot_temp_c=84.0,
            board_power_watts=400.0,
        )
        s_thermal = tracker.process_frame(frame_thermal)

    assert s_thermal.thermal_throttling_risk
    assert s_thermal.health_score_pct < s0.health_score_pct
    assert len(s_thermal.alerts) > 0

    # Connection Manager test
    mgr = DigitalTwinConnectionManager()
    state = mgr.process_telemetry("chassis_42u_node1", frame0)
    assert state.health_score_pct > 80.0


def test_api_cutting_edge_endpoints(sample_spec: DuctSpec):
    """Test FastAPI endpoints for STEP NURBS, hybrid AALC, and batch evaluation."""
    client = TestClient(app)

    # 1. STEP NURBS export endpoint
    resp_nurbs = client.post("/api/v1/export/step-nurbs", json=sample_spec.model_dump())
    assert resp_nurbs.status_code == 200
    assert "B_SPLINE_SURFACE_WITH_KNOTS" in resp_nurbs.text
    assert "ADVANCED_FACE" in resp_nurbs.text

    # 2. Hybrid AALC endpoint
    aalc_payload = {
        "spec": sample_spec.model_dump(),
        "aalc_spec": {
            "heat_load_watts": 850.0,
            "working_fluid": "water",
            "wick_structure": "sintered_copper",
        },
    }
    resp_aalc = client.post("/api/v1/hybrid-aalc", json=aalc_payload)
    assert resp_aalc.status_code == 200
    data_aalc = resp_aalc.json()
    assert "capillary_limit_watts" in data_aalc
    assert "boiling_limit_watts" in data_aalc
    assert data_aalc["operating_power_watts"] == 850.0

    # 3. Batch evaluation endpoint
    batch_payload = {
        "specs": [sample_spec.model_dump(), sample_spec.model_dump()],
    }
    resp_batch = client.post("/api/v1/batch-evaluate", json=batch_payload)
    assert resp_batch.status_code == 200
    data_batch = resp_batch.json()
    assert data_batch["count"] == 2
    assert len(data_batch["total_pressure_drop_pa"]) == 2
    assert len(data_batch["velocity_uniformity_index"]) == 2

    # 4. WebSocket digital twin streaming test
    with client.websocket_connect("/ws/telemetry/chassis_01") as websocket:
        frame = {
            "fan_rpm": 9500.0,
            "fan_pwm_pct": 55.0,
            "inlet_ambient_temp_c": 24.5,
            "exhaust_temp_c": 46.0,
            "gpu_hotspot_temp_c": 69.0,
            "board_power_watts": 420.0,
        }
        websocket.send_json(frame)
        rec = websocket.receive_json()
        assert "health_score_pct" in rec
        assert "filter_dust_loading_factor" in rec


def test_mcp_cutting_edge_tools(sample_spec: DuctSpec):
    """Verify MCP tools discovery, schema coverage, and execution."""
    tools = list_mcp_tools()
    tool_names = [t["name"] for t in tools]
    assert "forge_export_nurbs_step" in tool_names
    assert "forge_hybrid_aalc_cooling" in tool_names
    assert "forge_batch_screen_specs" in tool_names

    # Test MCP tool call: forge_export_nurbs_step
    req_nurbs = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "forge_export_nurbs_step",
            "arguments": {"spec": sample_spec.model_dump()},
        },
    }
    resp_nurbs = handle_mcp_jsonrpc_request(req_nurbs)
    assert resp_nurbs.get("error") is None
    parsed_nurbs = json.loads(resp_nurbs["result"]["content"][0]["text"])
    assert "step_content" in parsed_nurbs
    assert "B_SPLINE_SURFACE_WITH_KNOTS" in parsed_nurbs["step_content"]

    # Test MCP tool call: forge_hybrid_aalc_cooling
    req_aalc = {
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": {
            "name": "forge_hybrid_aalc_cooling",
            "arguments": {
                "spec": sample_spec.model_dump(),
                "heat_load_watts": 800.0,
                "working_fluid": "water",
            },
        },
    }
    resp_aalc = handle_mcp_jsonrpc_request(req_aalc)
    assert resp_aalc.get("error") is None
    parsed_aalc = json.loads(resp_aalc["result"]["content"][0]["text"])
    assert "capillary_limit_watts" in parsed_aalc

    # Test MCP tool call: forge_batch_screen_specs
    req_batch = {
        "jsonrpc": "2.0",
        "id": 103,
        "method": "tools/call",
        "params": {
            "name": "forge_batch_screen_specs",
            "arguments": {
                "specs": [sample_spec.model_dump()],
                "volumetric_flow_cfm": 130.0,
            },
        },
    }
    resp_batch = handle_mcp_jsonrpc_request(req_batch)
    assert resp_batch.get("error") is None
    parsed_batch = json.loads(resp_batch["result"]["content"][0]["text"])
    assert parsed_batch["count"] == 1


def test_open_schemas_registry_expansion():
    """Verify 20 domain models are registered in Draft 2020-12 schema exporter."""
    schemas_dict = export_all_json_schemas()
    schemas = schemas_dict["schemas"]
    assert len(schemas) == 20
    assert "HybridAALCSpec" in schemas
    assert "HybridAALCReport" in schemas
    assert "LiveTelemetryFrame" in schemas
    assert "DigitalTwinHealthState" in schemas
