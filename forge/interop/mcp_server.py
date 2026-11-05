"""Model Context Protocol (MCP) JSON-RPC 2.0 Server and OpenAI/Anthropic Function-Calling Schema Exporter."""
from __future__ import annotations
import json
from typing import Any, Dict, List
from forge import __version__
from forge.cad.dfm import analyze_dfm
from forge.cad.mesh import build_duct_mesh
from forge.cad.nurbs_step import export_exact_nurbs_step
from forge.cad.tpms import TPMSSpec, evaluate_tpms_thermo_hydraulics
from forge.cloud.rack_twin import IPMITelemetryFrame, evaluate_datacenter_rack_roi
from forge.compliance.dossier import generate_compliance_dossier
from forge.compliance.provenance import (
    generate_hardware_provenance_passport,
    verify_hardware_provenance,
)
from forge.copilot.agent import synthesize_from_prompt
from forge.core.models import DuctSpec, PhysicsConditions
from forge.interop.calculix_fea import export_calculix_inp
from forge.interop.fmi_fmu import export_fmu_package
from forge.interop.open_schemas import export_all_json_schemas
from forge.interop.openusd import export_openusd_stage
from forge.opt.differentiable_adjoint import run_differentiable_adjoint_codesign
from forge.physics.heatsink import HeatsinkSpec, solve_coupled_duct_heatsink_system
from forge.physics.hybrid_cooling import HybridAALCSpec, solve_hybrid_aalc_cooling
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.vectorized import evaluate_duct_specs_tensor_batch
from forge.presets.catalog import list_presets
from forge.topopt.darcy_solver import run_fluid_topology_optimization


def list_mcp_tools() -> List[Dict[str, Any]]:
    """Return MCP-compliant tool definitions for AI agent discovery."""
    return [
        {
            "name": "forge_synthesize_copilot",
            "description": "Synthesize an obstacle-routed, Stratford-Bezier-optimized cooling duct from a natural language prompt.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Natural language engineering prompt"},
                    "optimize_profile": {"type": "boolean", "default": True},
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "forge_evaluate_duct",
            "description": "Evaluate aerodynamic pressure drop, flow uniformity, aeroacoustics, structural deflection, and DFM printability.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "volumetric_flow_cfm": {"type": "number", "default": 150.0},
                    "heat_dissipation_watts": {"type": "number", "default": 350.0},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_optimize_topopt",
            "description": "Run Navier-Stokes Brinkman-Darcy fluid topology optimization to synthesize organic diffuser channels.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "iterations": {"type": "integer", "default": 12},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_coupled_heatsink",
            "description": "Solve coupled Fan + Duct + Finned Heatsink conjugate heat transfer and GPU hotspot junction temperatures.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "heat_dissipation_watts": {"type": "number", "default": 400.0},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_compliance_dossier",
            "description": "Generate an enterprise UL94 / ASHRAE / NEBS hardware qualification certificate and compliance dossier.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "fan_rpm": {"type": "integer", "default": 9000},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_tpms_lattice",
            "description": "Synthesize a 3D functionally graded TPMS (Gyroid, Schwarz-P, Diamond) metamaterial core and evaluate Darcy-Forchheimer hydraulics and Nusselt CHT.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": TPMSSpec.model_json_schema(),
                    "flow_cfm": {"type": "number", "default": 120.0},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_differentiable_adjoint",
            "description": "Run PyTorch end-to-end differentiable adjoint co-design to simultaneously optimize duct loft, corner radius, and heatsink fins.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "steps": {"type": "integer", "default": 35},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_provenance_passport",
            "description": "Generate a cryptographic Hardware Bill of Materials (HBOM) and SLSA digital manufacturing provenance passport with HMAC-SHA256 signature.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_rack_twin_roi",
            "description": "Evaluate 42U datacenter rack digital twin fan affinity power reduction, GPU hotspot relief, and annual financial payback.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "fans_rpm": {"type": "number", "default": 11500.0},
                    "gpu_temp_c": {"type": "number", "default": 78.5},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_export_openusd",
            "description": "Export Pixar / AOUSD OpenUSD (.usda) stage for NVIDIA Omniverse, Isaac Sim, and Blender with embedded physics digital twin attributes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_export_calculix_inp",
            "description": "Export CalculiX / Abaqus (.inp) finite element analysis deck with clamped inlet flange nodes, pressure DLOAD, and pre-stressed frequency extraction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_export_fmu",
            "description": "Export FMI 2.0 Co-Simulation and Model Exchange (.fmu) digital twin package with self-contained ANSI C99 kernel and XML description.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_export_nurbs_step",
            "description": "Export exact ISO-10303-21 STEP AP242 CAD model with bicubic B-spline NURBS surfaces (B_SPLINE_SURFACE_WITH_KNOTS, ADVANCED_FACE).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_hybrid_aalc_cooling",
            "description": "Evaluate two-phase vapor chamber and heat pipe hybrid cooling limits (capillary, boiling CHF, entrainment, and condensation) for 600W-1200W AI accelerators.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": DuctSpec.model_json_schema(),
                    "heat_load_watts": {"type": "number", "default": 800.0},
                    "working_fluid": {"type": "string", "enum": ["water", "methanol", "acetone"], "default": "water"},
                    "wick_structure": {"type": "string", "enum": ["sintered_copper", "micro_grooved", "screen_mesh"], "default": "sintered_copper"},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "forge_batch_screen_specs",
            "description": "Evaluate candidate duct specifications in parallel via vectorized PyTorch tensor SIMD/GPU pass.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "specs": {
                        "type": "array",
                        "items": DuctSpec.model_json_schema(),
                    },
                    "volumetric_flow_cfm": {"type": "number", "default": 120.0},
                },
                "required": ["specs"],
            },
        },
    ]


def get_openai_tool_schemas() -> List[Dict[str, Any]]:
    """Return OpenAI / Anthropic / Gemini compatible function-calling tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in list_mcp_tools()
    ]


def handle_mcp_jsonrpc_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a Model Context Protocol (MCP) JSON-RPC 2.0 request and return a JSON-RPC 2.0 response."""
    req_id = request.get("id", 1)
    method = request.get("method", "")
    params = request.get("params") or {}

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "forge-mcp-server", "version": __version__},
                    "capabilities": {"tools": {}, "resources": {}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": list_mcp_tools()},
            }

        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "resources": [
                        {
                            "uri": "forge://presets/catalog",
                            "name": "FORGE Hardware Accelerator Presets",
                            "mimeType": "application/json",
                        },
                        {
                            "uri": "forge://schemas/all",
                            "name": "FORGE Domain JSON Schemas",
                            "mimeType": "application/json",
                        },
                    ]
                },
            }

        if method == "resources/read":
            uri = params.get("uri", "")
            if uri == "forge://presets/catalog":
                payload = {"presets": list_presets()}
            elif uri == "forge://schemas/all":
                payload = export_all_json_schemas()
            else:
                raise ValueError(f"Unknown MCP resource URI: {uri}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(payload),
                        }
                    ]
                },
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments") or {}

            if tool_name == "forge_synthesize_copilot":
                res = synthesize_from_prompt(
                    prompt=args["prompt"],
                    optimize_profile=bool(args.get("optimize_profile", True)),
                )
                out_obj = res.model_dump(mode="json")

            elif tool_name == "forge_evaluate_duct":
                spec = DuctSpec(**args["spec"])
                cond = PhysicsConditions(
                    volumetric_flow_cfm=float(args.get("volumetric_flow_cfm", 150.0)),
                    heat_dissipation_watts=float(args.get("heat_dissipation_watts", 350.0)),
                )
                tris = build_duct_mesh(spec)
                phys = solve_duct_physics(spec, cond)
                dfm = analyze_dfm(spec, tris)
                out_obj = {"physics": phys.model_dump(), "dfm": dfm.model_dump()}

            elif tool_name == "forge_optimize_topopt":
                spec = DuctSpec(**args["spec"])
                top_res, _tris = run_fluid_topology_optimization(
                    spec, iterations=int(args.get("iterations", 12))
                )
                out_obj = top_res.model_dump(mode="json")

            elif tool_name == "forge_coupled_heatsink":
                spec = DuctSpec(**args["spec"])
                hs = HeatsinkSpec(width_mm=spec.outlet_width, fin_height_mm=max(10.0, spec.outlet_height - 4.0))
                cond = PhysicsConditions(heat_dissipation_watts=float(args.get("heat_dissipation_watts", 400.0)))
                rep = solve_coupled_duct_heatsink_system(spec, hs, cond)
                out_obj = rep.model_dump()

            elif tool_name == "forge_compliance_dossier":
                spec = DuctSpec(**args["spec"])
                dossier = generate_compliance_dossier(spec, fan_rpm=int(args.get("fan_rpm", 9000)))
                out_obj = dossier.model_dump(mode="json")

            elif tool_name == "forge_tpms_lattice":
                tpms_spec = TPMSSpec(**args["spec"])
                flow_cfm = float(args.get("flow_cfm", 120.0))
                rep = evaluate_tpms_thermo_hydraulics(tpms_spec, flow_cfm=flow_cfm)
                out_obj = rep.model_dump()

            elif tool_name == "forge_differentiable_adjoint":
                spec = DuctSpec(**args["spec"])
                steps = int(args.get("steps", 35))
                res = run_differentiable_adjoint_codesign(spec, steps=steps)
                out_obj = res.model_dump(mode="json")

            elif tool_name == "forge_provenance_passport":
                spec = DuctSpec(**args["spec"])
                passport = generate_hardware_provenance_passport(spec)
                verified = verify_hardware_provenance(passport)
                out_obj = {
                    "passport": passport.model_dump(mode="json"),
                    "signature_verified": verified,
                }

            elif tool_name == "forge_rack_twin_roi":
                spec = DuctSpec(**args["spec"])
                ipmi = IPMITelemetryFrame(
                    fans_rpm=float(args.get("fans_rpm", 11500.0)),
                    gpu_hotspot_temp_c=float(args.get("gpu_temp_c", 78.5)),
                )
                roi = evaluate_datacenter_rack_roi(spec, ipmi=ipmi)
                out_obj = roi.model_dump()

            elif tool_name == "forge_export_openusd":
                spec = DuctSpec(**args["spec"])
                usda = export_openusd_stage(spec)
                out_obj = {"stage_usda": usda}

            elif tool_name == "forge_export_calculix_inp":
                spec = DuctSpec(**args["spec"])
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as tmp:
                    inp_p = export_calculix_inp(spec, Path(tmp) / "deck.inp")
                    deck = inp_p.read_text(encoding="utf-8")
                out_obj = {"calculix_deck_inp": deck}

            elif tool_name == "forge_export_fmu":
                spec = DuctSpec(**args["spec"])
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as tmp:
                    fmu_p = export_fmu_package(spec, Path(tmp) / "model.fmu")
                    fmu_size = fmu_p.stat().st_size
                out_obj = {"fmu_filename": "model.fmu", "size_bytes": fmu_size, "status": "packaged"}

            elif tool_name == "forge_export_nurbs_step":
                spec = DuctSpec(**args["spec"])
                step_text = export_exact_nurbs_step(spec)
                out_obj = {"step_content": step_text, "char_count": len(step_text), "format": "AP242"}

            elif tool_name == "forge_hybrid_aalc_cooling":
                spec = DuctSpec(**args["spec"])
                aalc_spec = HybridAALCSpec(
                    heat_load_watts=float(args.get("heat_load_watts", 800.0)),
                    working_fluid=args.get("working_fluid", "water"),
                    wick_structure=args.get("wick_structure", "sintered_copper"),
                )
                rep = solve_hybrid_aalc_cooling(spec, aalc_spec=aalc_spec)
                out_obj = rep.model_dump()

            elif tool_name == "forge_batch_screen_specs":
                specs = [DuctSpec(**s) for s in args["specs"]]
                cond = PhysicsConditions(volumetric_flow_cfm=float(args.get("volumetric_flow_cfm", 120.0)))
                batch_res = evaluate_duct_specs_tensor_batch(specs, conditions=cond)
                out_obj = {
                    "count": len(specs),
                    "total_pressure_drop_pa": [round(float(v), 2) for v in batch_res.total_pressure_drop_pa],
                    "velocity_uniformity_index": [round(float(v), 4) for v in batch_res.velocity_uniformity_index],
                    "estimated_mass_grams": [round(float(v), 1) for v in batch_res.estimated_mass_grams],
                }

            else:
                raise ValueError(f"Unknown MCP tool: '{tool_name}'")

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(out_obj)}],
                    "isError": False,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": str(exc)},
        }

