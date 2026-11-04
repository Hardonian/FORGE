"""Model Context Protocol (MCP) JSON-RPC 2.0 Server and OpenAI/Anthropic Function-Calling Schema Exporter."""
from __future__ import annotations
import json
from typing import Any, Dict, List
from forge import __version__
from forge.cad.dfm import analyze_dfm
from forge.cad.mesh import build_duct_mesh
from forge.compliance.dossier import generate_compliance_dossier
from forge.copilot.agent import synthesize_from_prompt
from forge.core.models import DuctSpec, PhysicsConditions
from forge.interop.open_schemas import export_all_json_schemas
from forge.physics.heatsink import HeatsinkSpec, solve_coupled_duct_heatsink_system
from forge.physics.solver_1d import solve_duct_physics
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

