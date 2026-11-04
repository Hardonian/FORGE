"""CalculiX / Abaqus (.inp) 3D Structural & Thermal Finite Element Analysis Exporter.

Exports FORGE 3D meshes and physics boundary conditions into ready-to-run
CalculiX (`ccx`), Abaqus, and FreeCAD FEM `.inp` finite element decks with
clamped inlet flange nodes, temperature-dependent polymer properties, internal
static aerodynamic pressure loads, convective film coefficients, and modal steps.
"""

from __future__ import annotations

from pathlib import Path

from forge.cad.mesh import build_duct_mesh
from forge.core.hashes import sha256_hash
from forge.core.models import DuctSpec, MATERIAL_PROPERTIES, PhysicsConditions
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.structural_modal import MATERIAL_CTE_PER_K


def export_calculix_inp(
    spec: DuctSpec,
    output_path: str | Path,
    conditions: PhysicsConditions | None = None,
) -> Path:
    """Export a CalculiX / Abaqus `.inp` thermo-structural and modal FEA deck."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cond = conditions or PhysicsConditions()

    tris = build_duct_mesh(spec)
    physics = solve_duct_physics(spec, cond)
    spec_hash = sha256_hash(spec.to_canonical_dict())[:16]

    mat_key = spec.material.value
    mat_props = MATERIAL_PROPERTIES.get(mat_key, MATERIAL_PROPERTIES["PC-CF"])
    e_mpa = mat_props.get("youngs_modulus_mpa", 6500.0)
    hdt_c = mat_props.get("hdt_c", 140.0)
    nu = 0.36
    cte_1_k = MATERIAL_CTE_PER_K.get(mat_key, 25e-6)
    # CalculiX mm-tonne-s-N-MPa unit system: density in tonne/mm^3 = g/cm^3 * 1e-9
    rho_tonne_mm3 = mat_props.get("density", 1.20) * 1e-9

    # Deduplicate vertices (1-based indexing for Abaqus/CalculiX)
    vertex_map: dict[tuple[float, float, float], int] = {}
    nodes: list[tuple[float, float, float]] = []
    elements: list[tuple[int, int, int]] = []

    for tri in tris:
        tri_ids: list[int] = []
        for v in (tri.a, tri.b, tri.c):
            key = (round(v.x, 4), round(v.y, 4), round(v.z, 4))
            idx = vertex_map.get(key)
            if idx is None:
                idx = len(nodes) + 1
                vertex_map[key] = idx
                nodes.append(key)
            tri_ids.append(idx)
        if len({tri_ids[0], tri_ids[1], tri_ids[2]}) == 3:
            elements.append((tri_ids[0], tri_ids[1], tri_ids[2]))

    # Identify InletFlange nodes (z <= 1.0 mm) for fixed mounting constraint
    flange_nodes = [
        idx for idx, (_, _, z) in enumerate(nodes, start=1) if z <= 1.0
    ]
    if not flange_nodes:
        flange_nodes = [1]

    # Internal aerodynamic static pressure in MPa (1 Pa = 1e-6 MPa)
    internal_pressure_mpa = max(1e-6, physics.total_pressure_drop_pa * 1e-6)
    exhaust_temp_c = cond.air_temp_c + physics.estimated_temp_rise_c
    mat_safe_name = mat_key.replace("-", "_")

    lines: list[str] = [
        "** ===================================================================",
        "** FORGE Generative Hardware Platform - CalculiX / Abaqus FEA Deck",
        f"** Spec Hash : {spec_hash}",
        f"** Material  : {mat_key} (E={e_mpa:.0f} MPa, nu={nu:.2f}, HDT={hdt_c:.1f} C)",
        f"** Aero Load : {physics.total_pressure_drop_pa:.2f} Pa ({internal_pressure_mpa:.6e} MPa)",
        f"** Exhaust T : {exhaust_temp_c:.2f} C",
        "** Units     : mm, N, tonne, s, MPa, mJ, C",
        "** ===================================================================",
        "*HEADING",
        f"FORGE Cooling Duct Thermo-Structural & Modal FEA ({mat_key})",
        "*NODE, NSET=Nall",
    ]

    for idx, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"{idx}, {x:.4f}, {y:.4f}, {z:.4f}")

    lines.append("*ELEMENT, TYPE=S3, ELSET=Eall")
    for el_id, (n1, n2, n3) in enumerate(elements, start=1):
        lines.append(f"{el_id}, {n1}, {n2}, {n3}")

    lines.append("*NSET, NSET=InletFlange")
    for i in range(0, len(flange_nodes), 10):
        chunk = ", ".join(str(nid) for nid in flange_nodes[i : i + 10])
        lines.append(chunk)

    lines.extend(
        [
            f"*MATERIAL, NAME={mat_safe_name}",
            "*ELASTIC",
            f"{e_mpa:.1f}, {nu:.3f}, 25.0",
            f"{e_mpa * 0.72:.1f}, {nu:.3f}, {hdt_c:.1f}",
            "*DENSITY",
            f"{rho_tonne_mm3:.6e}",
            "*EXPANSION, ZERO=25.0",
            f"{cte_1_k:.6e}",
            "*CONDUCTIVITY",
            "0.28, 25.0",
            "*SPECIFIC HEAT",
            "1.20e+09",
            f"*SHELL SECTION, ELSET=Eall, MATERIAL={mat_safe_name}",
            f"{spec.wall:.3f}",
            "*INITIAL CONDITIONS, TYPE=TEMPERATURE",
            "Nall, 25.0",
            "** -------------------------------------------------------------------",
            "** STEP 1: Static Aerodynamic Pressure & Thermal Expansion Stress",
            "** -------------------------------------------------------------------",
            "*STEP",
            "*STATIC",
            "*BOUNDARY",
            "InletFlange, 1, 6, 0.0",
            "*DLOAD",
            f"Eall, P, {internal_pressure_mpa:.6e}",
            "*TEMPERATURE",
            f"Nall, {exhaust_temp_c:.2f}",
            "*NODE PRINT, NSET=InletFlange",
            "RF",
            "*EL PRINT, ELSET=Eall",
            "S, E",
            "*NODE FILE",
            "U, NT",
            "*EL FILE",
            "S, E",
            "*END STEP",
            "** -------------------------------------------------------------------",
            "** STEP 2: Pre-Stressed Eigenfrequency Modal Extraction (10 Modes)",
            "** -------------------------------------------------------------------",
            "*STEP",
            "*FREQUENCY",
            "10",
            "*NODE FILE",
            "U",
            "*END STEP",
        ]
    )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
